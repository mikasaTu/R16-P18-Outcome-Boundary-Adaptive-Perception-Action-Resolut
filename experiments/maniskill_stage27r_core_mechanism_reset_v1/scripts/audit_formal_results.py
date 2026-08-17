#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from common import PROTOCOL_ID, atomic_json, sha256_file

PREDECESSORS=["experiments/maniskill_act_boundary_screen_v1","experiments/maniskill_stage25_repair_oracle_v1","experiments/maniskill_stage26_counterfactual_completion_v1"]


def main():
 p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--formal-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); checks={}
 freeze=json.loads((a.repo/"experiments/maniskill_stage27r_core_mechanism_reset_v1/manifests/predecessor_tree_freeze.json").read_text())["trees"]
 current={path:subprocess.check_output(["git","rev-parse",f"HEAD:{path}"],cwd=a.repo,text=True).strip() for path in PREDECESSORS}; checks["predecessor_immutability"]={"pass":current==freeze,"frozen":freeze,"current":current}
 model=(a.repo/"experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/multires_policy.py").read_text(); tree=ast.parse(model); obs_keys=set()
 for node in ast.walk(tree):
  if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id in {"obs","data"} and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str): obs_keys.add(node.slice.value)
 checks["no_privileged_model_input"]={"pass":not bool(obs_keys-{"state","rgb","_visual_mode","_tile_id","_tile_grid"}),"observed_keys":sorted(obs_keys)}
 raw_files=sorted(a.formal_root.glob("oracle/**/*.json")); row_count=0; utility_mismatch=0; accounting_missing=0
 weights={"balanced":(100,20,5,-10,-5),"success_dominant":(120,10,3,-12,-6),"progress_dominant":(80,35,5,-10,-5)}
 for path in raw_files:
  value=json.loads(path.read_text())
  for row in value.get("rows",[]):
   row_count+=1; required={"global_encoder_calls","fine_encoder_calls","policy_forward_calls","policy_forward_rows","visual_tokens","action_opportunities","executed_steps","gpu_latency_ms","simulator_latency_ms","estimated_flops","peak_memory_bytes"}; accounting_missing+=not required.issubset(row["accounting"])
   for name,w in weights.items():
    expected=w[0]*row["success_hold5"]+w[1]*row["normalized_progress"]+w[2]*row["recoverable"]+w[3]*row["dropped_or_slipped"]+w[4]*row["collision"]
    utility_mismatch+=abs(expected-row["utilities"][name])>1e-9
 checks["raw_outcome_recompute"]={"pass":row_count>0 and utility_mismatch==0,"rows":row_count,"utility_mismatches":utility_mismatch}; checks["compute_accounting_recompute"]={"pass":row_count>0 and accounting_missing==0,"missing_rows":accounting_missing}
 files=sorted(path for path in a.formal_root.rglob("*") if path.is_file() and path!=a.output); manifest=[{"path":str(path.relative_to(a.formal_root)),"sha256":sha256_file(path),"bytes":path.stat().st_size} for path in files]
 checks["scientific_sha256_manifest"]={"pass":len(manifest)>0,"files":len(manifest)}; checks["all_pass"]=all(value["pass"] for value in checks.values() if isinstance(value,dict) and "pass" in value)
 atomic_json(a.output,{"protocol_id":PROTOCOL_ID,"checks":checks,"manifest":manifest}); print(json.dumps(checks,indent=2))

if __name__=="__main__": main()
