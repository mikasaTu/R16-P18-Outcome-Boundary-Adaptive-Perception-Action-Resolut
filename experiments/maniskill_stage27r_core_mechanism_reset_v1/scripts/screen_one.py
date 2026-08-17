#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from common import PROTOCOL_ID,atomic_json
from stage27r_runtime import checkpoint_candidates,evaluate_episode,load_agent,make_env

def summary(rows): return {k:float(np.mean([r[k] for r in rows])) for k in ["success_once","success_hold5","success_at_end","post_success_loss"]}
def main():
 p=argparse.ArgumentParser();p.add_argument("--training-run",type=Path,required=True);p.add_argument("--task",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda");a=p.parse_args()
 if a.output.exists(): print(a.output.read_text());return
 device=torch.device(a.device);env=make_env(a.task);screen=[]
 try:
  for c in checkpoint_candidates(a.training_run,a.task,a.seed):
   agent,_=load_agent(env,a.task,a.seed,Path(c["path"]),device);rows=[evaluate_episode(env,agent,s,device,"fine","fine") for s in range(2701000,2701040)];screen.append({**c,"screen":summary(rows)})
  top=sorted(screen,key=lambda r:(-r["screen"]["success_hold5"],-r["screen"]["success_at_end"],r["screen"]["post_success_loss"],r["step"]))[:2];validated=[]
  for c in top:
   agent,_=load_agent(env,a.task,a.seed,Path(c["path"]),device);modes={}
   for name,v,act in [("CC","coarse","coarse"),("FF","fine","fine")]: modes[name]=summary([evaluate_episode(env,agent,s,device,v,act) for s in range(2702000,2702100)])
   validated.append({**c,"validation":modes})
  selected=sorted(validated,key=lambda r:(-r["validation"]["FF"]["success_hold5"],-r["validation"]["FF"]["success_at_end"],r["validation"]["FF"]["post_success_loss"],r["step"]))[0]
 finally:env.close()
 atomic_json(a.output,{"protocol_id":PROTOCOL_ID,"task":a.task,"seed":a.seed,"screened":screen,"validated_top2":validated,"selected":selected})
if __name__=="__main__":main()
