#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import PROTOCOL_ID, atomic_json
from stage27r_runtime import evaluate_episode, load_agent, make_env

WEIGHTS={"balanced":(100,20,5,-10,-5),"success_dominant":(120,10,3,-12,-6),"progress_dominant":(80,35,5,-10,-5)}


def utility(row,w): return w[0]*row["success_hold5"]+w[1]*row["normalized_progress"]+w[2]*row["recoverable"]+w[3]*row["dropped_or_slipped"]+w[4]*row["collision"]


def main():
 p=argparse.ArgumentParser(); p.add_argument("--state-bank",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--model-seed",type=int,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda"); p.add_argument("--tile-grid",type=int,choices=[2,4],default=2); p.add_argument("--state-start",type=int,default=0); p.add_argument("--state-stop",type=int); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 bank=json.loads(a.state_bank.read_text()); states=bank["states"][a.state_start:a.state_stop]; task=bank["task"]; seed=a.model_seed; repeats=3 if bank["bank"]=="calibration" else 5; device=torch.device(a.device); env=make_env(task); agent,_=load_agent(env,task,seed,a.checkpoint,device); rows=[]; tiles=a.tile_grid*a.tile_grid; conditions=[("CC","coarse","coarse",None),("CF","coarse","fine",None),*[(f"FC_tile{i}","fine","coarse",i) for i in range(tiles)],*[(f"FF_tile{i}","fine","fine",i) for i in range(tiles)]]
 try:
  for state in states:
   for name,visual,action,tile in conditions:
    for repeat in range(repeats):
     result=evaluate_episode(env,agent,int(state["episode_seed"]),device,visual,action,tile,horizon=28,prefix_actions=state["prefix_actions"],treatment_steps=8,continuation_steps=20,tile_grid=a.tile_grid)
     result.update(protocol_id=PROTOCOL_ID,task=task,model_seed=seed,bank=bank["bank"],bank_id=state["bank_id"],source_episode=state["source_episode"],phase=state["phase"],source_type=state["source_type"],condition=name,repeat=repeat,causal_fidelity_pass=state["fidelity"]["pass"])
     result["utilities"]={key:utility(result,value) for key,value in WEIGHTS.items()}; rows.append(result)
 finally: env.close()
 atomic_json(a.output,{"protocol_id":PROTOCOL_ID,"rows":rows,"row_count":len(rows),"conditions":[x[0] for x in conditions],"repeats":repeats,"tile_grid":a.tile_grid})

if __name__=="__main__": main()
