#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import PROTOCOL_ID, atomic_json
from stage27r_runtime import TASKS, checkpoint_candidates, evaluate_episode, load_agent, make_env

MODEL_SEEDS = (16018, 16019, 16020)
CANDIDATE_ORDER = ("PegInsertionSide-v1", "PlugCharger-v1", "PullCubeTool-v1", "PushT-v1")


def summarize(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in ["success_once","success_hold5","success_at_end","post_success_loss"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--training-run",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    device=torch.device(args.device); all_groups={}
    for task in ("StackCube-v1", *CANDIDATE_ORDER, "PushCube-v1"):
      for seed in MODEL_SEEDS:
        candidates=checkpoint_candidates(args.training_run,task,seed); env=make_env(task)
        screened=[]
        try:
          for candidate in candidates:
            agent,_=load_agent(env,task,seed,Path(candidate["path"]),device)
            rows=[]
            for episode_seed in range(2701000,2701040):
              rows.append(evaluate_episode(env,agent,episode_seed,device,"fine","fine"))
            screened.append({**candidate,"screen":summarize(rows)})
          top=sorted(screened,key=lambda row:(-row["screen"]["success_hold5"],-row["screen"]["success_at_end"],row["screen"]["post_success_loss"],row["step"]))[:2]
          validated=[]
          for candidate in top:
            agent,_=load_agent(env,task,seed,Path(candidate["path"]),device); modes={}
            for name,visual,action in [("CC","coarse","coarse"),("FF","fine","fine")]:
              rows=[evaluate_episode(env,agent,s,device,visual,action) for s in range(2702000,2702100)]
              modes[name]=summarize(rows)
            validated.append({**candidate,"validation":modes})
          selected=sorted(validated,key=lambda row:(-row["validation"]["FF"]["success_hold5"],-row["validation"]["FF"]["success_at_end"],row["validation"]["FF"]["post_success_loss"],row["step"]))[0]
          all_groups[f"{task}/seed_{seed}"]={"screened":screened,"validated_top2":validated,"selected":selected}
        finally: env.close()
    task_gate={}
    for task in ("StackCube-v1",*CANDIDATE_ORDER):
      ff=[all_groups[f"{task}/seed_{seed}"]["selected"]["validation"]["FF"]["success_hold5"] for seed in MODEL_SEEDS]
      cc=[all_groups[f"{task}/seed_{seed}"]["selected"]["validation"]["CC"]["success_hold5"] for seed in MODEL_SEEDS]
      passed=0.30<=np.mean(ff)<=0.80 and (max(ff)-min(ff))<=0.20 and sum(v>=0.25 for v in ff)>=2 and np.mean(ff)<=0.90 and np.mean(cc)>=0.5*np.mean(ff)
      task_gate[task]={"FF_success_hold5_by_seed":ff,"CC_success_hold5_by_seed":cc,"pass":bool(passed)}
    selected_positive=next((task for task in CANDIDATE_ORDER if task_gate[task]["pass"]),None)
    push=[all_groups[f"PushCube-v1/seed_{seed}"]["selected"]["validation"]["FF"]["success_hold5"] for seed in MODEL_SEEDS]
    result={"protocol_id":PROTOCOL_ID,"groups":all_groups,"task_gate":task_gate,"selected_positive":selected_positive,"pushcube_success_hold5":push,"selected_negative":"PushCube-v1" if np.mean(push)>=0.70 else None}
    atomic_json(args.output,result); print(json.dumps({k:v for k,v in result.items() if k!="groups"},indent=2))

if __name__=="__main__": main()
