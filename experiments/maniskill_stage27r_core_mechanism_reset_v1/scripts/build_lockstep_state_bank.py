#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from common import PROTOCOL_ID, atomic_json
from stage27r_runtime import TASKS, choose_tile, load_agent, make_env, object_pose, query, quat_distance

PHASES=("free_space_approach","pre_contact_or_pre_grasp","object_in_hand_pre_placement","contact_placement_near_completion")
FRACTIONS=(0.18,0.38,0.62,0.82)


def rgb_array(obs): return obs["rgb"].detach().cpu().numpy()


def fidelity(task, seed, prefix):
    # CPU PhysX forbids num_envs > 1.  Use two independently reset simulator
    # instances and broadcast the exact same action to both shadow branches.
    left=make_env(task,1); right=make_env(task,1)
    try:
      left_obs,left_info=left.reset(seed=[seed]); right_obs,right_info=right.reset(seed=[seed]); max_action=max_trans=max_rot=0.0; max_rgb=0; categorical=True
      for raw in prefix:
        left_action=torch.as_tensor(raw,dtype=torch.float32,device=left.base_env.device).reshape(1,-1)
        right_action=torch.as_tensor(raw,dtype=torch.float32,device=right.base_env.device).reshape(1,-1)
        max_action=max(max_action,float(torch.max(torch.abs(left_action.cpu()-right_action.cpu())).item()))
        left_obs,_,_,_,left_info=left.step(left_action); right_obs,_,_,_,right_info=right.step(right_action)
        left_pos,left_quat=object_pose(left.base_env,task); right_pos,right_quat=object_pose(right.base_env,task)
        max_trans=max(max_trans,float(np.linalg.norm(left_pos[0]-right_pos[0]))); max_rot=max(max_rot,quat_distance(left_quat[0],right_quat[0])); max_rgb=max(max_rgb,int(np.max(np.abs(rgb_array(left_obs)[0].astype(int)-rgb_array(right_obs)[0].astype(int))))); categorical &= bool(left_info["success"][0]==right_info["success"][0])
      branch_success=bool(left_info["success"][0].item())
      passed=max_action==0 and max_trans<=1e-5 and max_rot<=1e-4 and max_rgb<=1 and categorical
      return {"broadcast_action_max_abs":max_action,"translation_m":max_trans,"rotation_rad":max_rot,"rgb_max_lsb":max_rgb,"categorical_agreement":categorical,"branch_success":branch_success,"pass":passed}
    finally: left.close(); right.close()


def expert_rows(h5_path, count_per_phase, offset):
    rows=[]
    with h5py.File(h5_path,"r") as source:
      keys=sorted(source,key=lambda key:int(key.removeprefix("traj_")))
      needed=count_per_phase*4
      if len(keys)<offset+needed: raise RuntimeError(f"expert source has {len(keys)}, need offset+count {offset+needed}")
      for index,key in enumerate(keys[offset:offset+needed]):
        phase_index=index//count_per_phase; actions=np.asarray(source[f"{key}/actions"],dtype=np.float32); branch=max(1,min(len(actions)-1,round(len(actions)*FRACTIONS[phase_index])))
        successes=np.asarray(source[f"{key}/success"],dtype=bool) if f"{key}/success" in source else np.zeros(len(actions),dtype=bool)
        if successes.any(): branch=min(branch,max(1,int(np.flatnonzero(successes)[0])))
        rows.append({"source_type":"expert","source_episode":key,"phase":PHASES[phase_index],"episode_seed":None,"branch_step":branch,"prefix_actions":actions[:branch].tolist()})
    return rows


def onpolicy_rows(env,agent,task,model_seed,device,count_per_phase,seed_base):
    rows=[]
    for phase_index,phase in enumerate(PHASES):
      for offset in range(count_per_phase):
        seed=seed_base+phase_index*100+offset; obs,_=env.reset(seed=[seed]); prefix=[]; branch=max(1,round(TASKS[task][1]*FRACTIONS[phase_index])); cached=None
        for step in range(branch):
          if step%4==0 or cached is None: cached=query(agent,obs,device,"fine",choose_tile(obs),{"global_encoder_calls":0,"fine_encoder_calls":0,"visual_tokens":0,"policy_forward_calls":0,"policy_forward_rows":0,"gpu_latency_ms":0})
          action=cached[:,step%4]; prefix.append(action[0].detach().cpu().tolist()); obs,_,_,_,info=env.step(action)
          if bool(info["success"][0].item()): prefix.pop(); break
        rows.append({"source_type":"on_policy","source_episode":f"seed_{seed}","phase":phase,"episode_seed":seed,"branch_step":len(prefix),"prefix_actions":prefix})
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=TASKS,required=True); p.add_argument("--model-seed",type=int,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--expert-h5",type=Path,required=True); p.add_argument("--expert-offset",type=int,default=0); p.add_argument("--bank",choices=["calibration","confirmatory","negative"],required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda"); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    total=48 if a.bank in ("calibration","negative") else 96; per_phase=total//8; device=torch.device(a.device); env=make_env(a.task); agent,_=load_agent(env,a.task,a.model_seed,a.checkpoint,device)
    rows=expert_rows(a.expert_h5,per_phase,a.expert_offset)+onpolicy_rows(env,agent,a.task,a.model_seed,device,per_phase,2703000 if a.bank=="calibration" else 2704000)
    env.close(); rows=sorted(rows,key=lambda row:(PHASES.index(row["phase"]),row["source_type"],hashlib.sha256(f"{row['source_episode']}".encode()).hexdigest()))
    for index,row in enumerate(rows):
      if row["episode_seed"] is None:
        # The replay metadata seed is required for reset; source episode ids are
        # resolved from the adjacent JSON in the same order.
        meta=json.loads(a.expert_h5.with_suffix(".json").read_text())["episodes"][int(row["source_episode"].removeprefix("traj_"))]; row["episode_seed"]=int(meta["episode_seed"])
      row["bank_id"]=f"{a.task}-{a.model_seed}-{a.bank}-{index:03d}"; row["prefix_sha256"]=hashlib.sha256(np.asarray(row["prefix_actions"],np.float32).tobytes()).hexdigest(); row["fidelity"]=fidelity(a.task,int(row["episode_seed"]),row["prefix_actions"])
    if any(row["fidelity"]["branch_success"] for row in rows): raise RuntimeError("post-success state admitted to primary bank")
    pass_rate=float(np.mean([row["fidelity"]["pass"] for row in rows])); atomic_json(a.output,{"protocol_id":PROTOCOL_ID,"task":a.task,"model_seed":a.model_seed,"bank":a.bank,"states":rows,"count":len(rows),"fidelity_pass_rate":pass_rate,"post_success_states":0,"causal_gate_pass":pass_rate>=0.95})

if __name__=="__main__": main()
