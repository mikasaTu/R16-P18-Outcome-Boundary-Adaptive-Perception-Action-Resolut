#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import PROTOCOL_ID, append_jsonl, sha256_file, write_json_new
from predictor import CompletionModel, FeatureShape, flatten_feature
from stage26_runtime import make_env, load_policy_from_checkpoint, neutral_from_last, policy_chunk, public_snapshot, temporal_action_for_indices, visual_latent

MODES = ("fixed_horizon", "learned_counterfactual_completion_gate", "learned_success_only_classifier", "privileged_neutral_after_hold5", "privileged_terminate_first_success", "fixed_time_matched_stop", "random_matched_stop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-seed", type=int, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed-bank", type=Path, required=True); parser.add_argument("--predictor", type=Path, required=True); parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--num-envs", type=int, default=20); parser.add_argument("--matched-profile", type=Path); parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def load_predictor(path: Path, device: torch.device) -> CompletionModel:
    payload = torch.load(path, map_location=device, weights_only=False); shape = FeatureShape(**payload["shape"]); model = CompletionModel(payload["architecture"], shape).to(device); model.load_state_dict(payload["state_dict"]); model.eval(); return model


def pad(rows: deque, count: int, fallback: list[float]) -> list[list[float]]:
    values = list(rows) or [fallback]
    while len(values) < count: values.insert(0, values[0])
    return values[-count:]


def matched_schedule(seeds: list[int], profile_path: Path, mode: str) -> dict[int, int]:
    profile = json.loads(profile_path.read_text()); stopped = [int(row["stop_step"]) for row in profile["stops"] if row["stop_step"] is not None]
    count = len(stopped)
    ranked = sorted(seeds, key=lambda seed: hashlib.sha256(f"{PROTOCOL_ID}|matched|{mode}|{seed}".encode()).hexdigest())[:count]
    if not stopped: return {}
    if mode == "fixed_time_matched_stop":
        step = int(round(float(np.mean(stopped)))); return {seed: step for seed in ranked}
    ordered_steps = sorted(stopped, key=lambda step: hashlib.sha256(f"{PROTOCOL_ID}|random-step|{step}".encode()).hexdigest())
    return {seed: ordered_steps[index % len(ordered_steps)] for index, seed in enumerate(ranked)}


def run_batch(env: Any, agent: torch.nn.Module, predictor: CompletionModel, seeds: list[int], device: torch.device, mode: str, calibration: dict[str, Any], schedule: dict[int,int]) -> list[dict[str, Any]]:
    random.seed(16018); np.random.seed(16018); torch.manual_seed(16018); torch.cuda.manual_seed_all(16018)
    obs, _ = env.reset(seed=seeds); count, horizon, queries = len(seeds), 200, 30; dim = int(env.action_space.shape[-1])
    table = torch.zeros(count,horizon,horizon+queries,dim,device=device); last_action = torch.zeros(count,dim,device=device)
    active = torch.ones(count,dtype=torch.bool,device=device); terminated = torch.zeros_like(active); gate_streak = torch.zeros(count,dtype=torch.int64,device=device)
    success_once = torch.zeros_like(active); hold5 = torch.zeros_like(active); streak = torch.zeros(count,dtype=torch.int64,device=device); longest = torch.zeros_like(streak); terminal_success = torch.zeros_like(active)
    first_success = torch.full((count,),-1,dtype=torch.int64,device=device); stop_step = torch.full((count,),-1,dtype=torch.int64,device=device)
    policy_calls = torch.zeros(count,dtype=torch.int64,device=device); policy_latency = torch.zeros(count,dtype=torch.float64,device=device); head_latency = torch.zeros(count,dtype=torch.float64,device=device)
    histories_latent=[deque(maxlen=4) for _ in seeds]; histories_proprio=[deque(maxlen=4) for _ in seeds]; histories_action=[deque(maxlen=4) for _ in seeds]; traces=[[] for _ in seeds]
    info={"success":torch.zeros(count,dtype=torch.bool,device=device)}
    for step in range(horizon):
        action = neutral_from_last(last_action)
        # Controls whose stop decision is already known at the beginning of
        # the step must not make an unreported policy call.
        pre_stop=torch.zeros(count,dtype=torch.bool,device=device)
        if mode=="privileged_neutral_after_hold5": pre_stop=(streak>=5)&active
        elif mode in {"fixed_time_matched_stop","random_matched_stop"}:
            for index,seed in enumerate(seeds): pre_stop[index]=active[index] and schedule.get(seed,-1)==step
        stop_step[pre_stop]=step; active &= ~pre_stop
        policy_indices=torch.nonzero(active & ~terminated,as_tuple=False).flatten(); chunks={}; called_this_step=torch.zeros(count,dtype=torch.bool,device=device)
        if policy_indices.numel():
            subset={k:v[policy_indices] for k,v in obs.items()}; started=time.perf_counter(); chunk=policy_chunk(agent,subset,device); torch.cuda.synchronize(device); elapsed=time.perf_counter()-started
            called_this_step[policy_indices]=True; policy_calls[policy_indices]+=1; policy_latency[policy_indices]+=elapsed/int(policy_indices.numel())
            latent=visual_latent(agent,obs,policy_indices).detach().cpu().numpy()
            for local,index_tensor in enumerate(policy_indices):
                index=int(index_tensor.item()); histories_latent[index].append(latent[local].astype(float).tolist()); histories_proprio[index].append(obs["state"][index].detach().cpu().float().tolist()); chunks[index]=chunk[local]
            should_stop=torch.zeros(count,dtype=torch.bool,device=device)
            if mode in {"learned_counterfactual_completion_gate","learned_success_only_classifier"}:
                feature_rows=[]; active_list=policy_indices.tolist()
                for index in active_list:
                    feature={"visual":np.asarray(pad(histories_latent[index],4,[0.0]*predictor.shape.visual),dtype=np.float32),"proprio":np.asarray(pad(histories_proprio[index],4,[0.0]*predictor.shape.proprio),dtype=np.float32),"actions":np.asarray(pad(histories_action[index],4,last_action[index].detach().cpu().tolist()),dtype=np.float32),"predicted":chunk[active_list.index(index),:5].detach().cpu().numpy().reshape(-1).astype(np.float32),"consistency":np.zeros(predictor.shape.consistency,dtype=np.float32)}
                    actions=feature["actions"]; feature["consistency"][:4]=[float(np.mean(np.linalg.norm(np.diff(actions,axis=0),axis=1))),float(np.std(actions)),float(np.linalg.norm(actions[-1]-actions[-2])),float(last_action[index,-1].item())]
                    feature_rows.append(flatten_feature(feature))
                started_head=time.perf_counter(); logits=predictor(torch.from_numpy(np.stack(feature_rows)).to(device)); torch.cuda.synchronize(device); head_elapsed=time.perf_counter()-started_head
                probs=torch.sigmoid(logits/float(calibration["temperature"]))
                for local,index in enumerate(active_list):
                    if mode=="learned_counterfactual_completion_gate": signal=bool(probs[local,0]>=calibration["tau_hold"] and probs[local,0]-probs[local,1]>=calibration["tau_advantage"])
                    else: signal=bool(probs[local,2]>=calibration["success_only_threshold"])
                    gate_streak[index]=gate_streak[index]+1 if signal else 0; should_stop[index]=gate_streak[index]>=2; head_latency[index]+=head_elapsed/len(active_list)
            newly_stop=should_stop & active; stop_step[newly_stop]=step; active &= ~newly_stop
            remaining=torch.nonzero(active & ~terminated,as_tuple=False).flatten()
            if remaining.numel():
                local_map={int(global_index.item()):local for local,global_index in enumerate(policy_indices)}
                chosen_chunk=torch.stack([chunk[local_map[int(index.item())]] for index in remaining])
                chosen=temporal_action_for_indices(table,chosen_chunk,step,remaining); action[remaining]=chosen; last_action[remaining]=chosen
        obs,_,_,_,info=env.step(action); success=info["success"].to(device=device,dtype=torch.bool)
        newly=success & ~success_once & ~terminated; first_success[newly]=step+1; success_once |= success & ~terminated; streak=torch.where(success & ~terminated,streak+1,torch.where(~terminated,torch.zeros_like(streak),streak)); longest=torch.maximum(longest,streak); hold5 |= streak>=5
        if mode=="privileged_terminate_first_success":
            stop=success & ~terminated; terminal_success[stop]=True; terminated |= stop; active &= ~stop; stop_step[stop]=step+1
        snap=public_snapshot(env.base_env)
        for index in range(count):
            histories_action[index].append(action[index].detach().cpu().float().tolist()); traces[index].append({"step":step+1,"success":bool(success[index].item()),"success_streak":int(streak[index].item()),"policy_called":bool(called_this_step[index].item()),"object_position":snap["object_position"][index].astype(float).tolist(),"object_quaternion":snap["object_quaternion"][index].astype(float).tolist(),"executed_action":action[index].detach().cpu().float().tolist()})
    terminal_success[~terminated]=info["success"].to(device=device,dtype=torch.bool)[~terminated]
    return [{"protocol_id":PROTOCOL_ID,"model_seed":int(0),"episode_seed":int(seed),"mode":mode,"success_once":bool(success_once[i].item()),"success_hold5":bool(hold5[i].item()),"success_at_end":bool(terminal_success[i].item()),"post_success_loss":bool(success_once[i].item() and not terminal_success[i].item()),"longest_success_streak":int(longest[i].item()),"first_success_step":int(first_success[i].item()),"stop_step":None if stop_step[i].item()<0 else int(stop_step[i].item()),"policy_calls":int(policy_calls[i].item()),"policy_latency_seconds":float(policy_latency[i].item()),"completion_head_latency_seconds":float(head_latency[i].item()),"trace":traces[i]} for i,seed in enumerate(seeds)]


def main() -> None:
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); marker=args.output_dir/"CLOSED_LOOP_COMPLETE.json"
    if marker.exists(): print("CLOSED_LOOP_ALREADY_COMPLETE"); return
    seeds=[int(x) for x in json.loads(args.seed_bank.read_text())["banks"]["confirmatory"]]
    if args.max_episodes is not None: seeds=seeds[:args.max_episodes]
    if len(seeds)%args.num_envs: raise ValueError("num-envs must divide episodes")
    freeze=json.loads(args.calibration_freeze.read_text()); calibration=freeze["selected_checkpoints"][str(args.model_seed)]["calibration"]
    schedule={}
    if args.mode in {"fixed_time_matched_stop","random_matched_stop"}:
        if args.matched_profile is None: raise ValueError("matched profile required")
        schedule=matched_schedule(seeds,args.matched_profile,args.mode)
    device=torch.device("cuda"); env=make_env("StackCube-v1",args.num_envs,sim_backend="physx_cuda"); agent,_=load_policy_from_checkpoint(env,"StackCube-v1",args.model_seed,args.checkpoint,device,args.checkpoint_sha256); predictor=load_predictor(args.predictor,device)
    output=args.output_dir/"episodes.jsonl"; started=time.time(); records=[]
    try:
        for offset in range(0,len(seeds),args.num_envs):
            batch=run_batch(env,agent,predictor,seeds[offset:offset+args.num_envs],device,args.mode,calibration,schedule)
            for row in batch: row["model_seed"]=args.model_seed; append_jsonl(output,row)
            records+=batch; print(f"STAGE26_CLOSED_LOOP_PROGRESS mode={args.mode} seed={args.model_seed} episodes={len(records)}/{len(seeds)}",flush=True)
    finally: env.close()
    stops=[{"episode_seed":row["episode_seed"],"stop_step":row["stop_step"]} for row in records]
    summary={"protocol_id":PROTOCOL_ID,"status":"CLOSED_LOOP_ARM_COMPLETE","mode":args.mode,"model_seed":args.model_seed,"episodes":len(records),"success_once":float(np.mean([r["success_once"] for r in records])),"success_at_end":float(np.mean([r["success_at_end"] for r in records])),"post_success_loss":float(np.mean([r["post_success_loss"] for r in records])),"stop_rate":float(np.mean([r["stop_step"] is not None for r in records])),"stops":stops,"policy_calls":int(sum(r["policy_calls"] for r in records)),"policy_latency_seconds":float(sum(r["policy_latency_seconds"] for r in records)),"completion_head_latency_seconds":float(sum(r["completion_head_latency_seconds"] for r in records)),"episodes_sha256":sha256_file(output),"wall_seconds":time.time()-started}
    write_json_new(args.output_dir/"summary.json",summary); write_json_new(marker,{"protocol_id":PROTOCOL_ID,"status":"CLOSED_LOOP_ARM_COMPLETE","summary_sha256":sha256_file(args.output_dir/"summary.json")})


if __name__=="__main__": main()
