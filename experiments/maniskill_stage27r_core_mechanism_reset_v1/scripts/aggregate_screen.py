#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from common import PROTOCOL_ID,atomic_json
TASKS=("StackCube-v1","PegInsertionSide-v1","PlugCharger-v1","PullCubeTool-v1","PushT-v1","PushCube-v1"); POS=TASKS[1:5];SEEDS=(16018,16019,16020)
def main():
 p=argparse.ArgumentParser();p.add_argument("--shard-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();groups={}
 for t in TASKS:
  for s in SEEDS:groups[f"{t}/seed_{s}"]=json.loads((a.shard_dir/f"{t}-seed{s}.json").read_text())
 gates={}
 for t in TASKS[:-1]:
  ff=[groups[f"{t}/seed_{s}"]["selected"]["validation"]["FF"]["success_hold5"] for s in SEEDS];cc=[groups[f"{t}/seed_{s}"]["selected"]["validation"]["CC"]["success_hold5"] for s in SEEDS]
  gates[t]={"FF":ff,"CC":cc,"pass":bool(.30<=np.mean(ff)<=.80 and max(ff)-min(ff)<=.20 and sum(x>=.25 for x in ff)>=2 and np.mean(ff)<=.90 and np.mean(cc)>=.5*np.mean(ff))}
 selected=next((t for t in POS if gates[t]["pass"]),None);push=[groups[f"PushCube-v1/seed_{s}"]["selected"]["validation"]["FF"]["success_hold5"] for s in SEEDS]
 atomic_json(a.output,{"protocol_id":PROTOCOL_ID,"groups":groups,"task_gates":gates,"selected_positive":selected,"pushcube_success_hold5":push,"selected_negative":"PushCube-v1" if np.mean(push)>=.70 else None,"fresh_reset_replay_agreement_pending_lockstep_bank":True})
if __name__=="__main__":main()
