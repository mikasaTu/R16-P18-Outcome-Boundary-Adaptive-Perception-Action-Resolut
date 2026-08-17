#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py

from common import PROTOCOL_ID, atomic_json, sha256_file

CONTROL={"StackCube-v1":"pd_ee_delta_pos","PegInsertionSide-v1":"pd_ee_delta_pose","PlugCharger-v1":"pd_ee_delta_pose","PullCubeTool-v1":"pd_ee_delta_pose","PushT-v1":"pd_ee_delta_pose","PushCube-v1":"pd_ee_delta_pos"}


def main():
 p=argparse.ArgumentParser(); p.add_argument("--task",choices=CONTROL,required=True); p.add_argument("--official-h5",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--python",type=Path,required=True); a=p.parse_args(); root=a.output_root/a.task; complete=root/"EXPERT_POOL_COMPLETE.json"
 if complete.exists(): print(complete.read_text()); return
 meta=json.loads(a.official_h5.with_suffix(".json").read_text()); episodes=sorted(meta["episodes"],key=lambda r:int(r["episode_id"]))[500:720]; raw=root/"source"/"trajectory.h5"; raw.parent.mkdir(parents=True,exist_ok=True)
 if raw.exists(): raise FileExistsError(raw)
 with h5py.File(a.official_h5,"r") as src,h5py.File(raw,"x") as dst:
  for i,row in enumerate(episodes): src.copy(src[f"traj_{row['episode_id']}"],dst,name=f"traj_{i}"); row["episode_id"]=i
 meta["episodes"]=episodes; raw.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
 command=[str(a.python),"-m","mani_skill.trajectory.replay_trajectory","--traj-path",str(raw),"--sim-backend","physx_cpu","--obs-mode","rgb","--target-control-mode",CONTROL[a.task],"--save-traj","--use-first-env-state","--max-retry","9","--num-envs","8"]
 with (root/"replay.log").open("x") as log: rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
 if rc: raise RuntimeError(f"expert replay failed {rc}")
 output=raw.with_name(f"trajectory.rgb.{CONTROL[a.task]}.physx_cpu.h5"); count=len(json.loads(output.with_suffix('.json').read_text())["episodes"])
 if count<72: raise RuntimeError(f"expert pool only {count}, need 72")
 atomic_json(complete,{"protocol_id":PROTOCOL_ID,"status":"PASS","task":a.task,"successful":count,"h5":str(output),"h5_sha256":sha256_file(output)})

if __name__=="__main__": main()
