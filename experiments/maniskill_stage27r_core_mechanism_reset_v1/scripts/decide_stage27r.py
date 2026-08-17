#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import PROTOCOL_ID, atomic_json


def main():
 p=argparse.ArgumentParser(); p.add_argument("--analysis",type=Path,required=True); p.add_argument("--state-banks",type=Path,nargs="+",required=True); p.add_argument("--positive-tasks",nargs=2,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); analysis=json.loads(a.analysis.read_text()); stats=analysis["statistics"]["balanced"]
 fidelity=[json.loads(path.read_text())["fidelity_pass_rate"] for path in a.state_banks]; causal=min(fidelity)>=.95
 visual=[stats[t]["visual"] for t in a.positive_tasks]; action=[stats[t]["action"] for t in a.positive_tasks]; joint=[stats[t]["joint"] for t in a.positive_tasks]
 visual_pass=all(x["mean"]>0 and x["ci95"][0]>0 for x in visual)
 action_strong=sum(x["mean"]>0 and x["ci95"][0]>0 for x in action)>=1 and all(x["mean"]>=0 for x in action)
 joint_pass=all(x["ci95"][0]>0 for x in joint)
 joint_fraction=all(stats[t]["positive_joint_state_fraction"]>=.10 for t in a.positive_tasks)
 budget=analysis["budgets"]["balanced"]["0.5"]; jo=budget["joint_oracle"]; fixed=budget["strongest_equal_cost_fixed_axis"]
 success_gain=jo["success_hold5"]-fixed["success_hold5"]; compute_reduction=1-jo["cost"]/max(fixed["cost"],1)
 budget_joint=(success_gain>=.05) or (abs(success_gain)<=.02 and compute_reduction>=.25)
 state_axis=budget["state_axis_oracle"]["utility"]>fixed["utility"]
 visual_budget=budget["visual_only_oracle"]["utility"]>max(budget["random_state"]["utility"],budget["all_coarse"]["utility"])
 if not causal: status="NO_GO_CAUSAL_BACKEND"
 elif not visual_pass and not action_strong and not state_axis: status="NO_GO_CORE_MECHANISM"
 elif visual_pass and not action_strong and not joint_pass and visual_budget: status="REVISE_VISUAL_ONLY"
 elif visual_pass and action_strong and (not joint_pass) and state_axis: status="REVISE_SHARED_AXIS_ROUTER"
 elif visual_pass and action_strong and joint_pass and joint_fraction and budget_joint: status="GO_FULL_JOINT"
 else: status="NO_GO_CORE_MECHANISM"
 result={"protocol_id":PROTOCOL_ID,"final_status":status,"precedence_applied":True,"causal_backend_pass":causal,"fidelity_pass_rates":fidelity,"visual_effect_pass_both":visual_pass,"action_effect_pass":action_strong,"joint_effect_pass":joint_pass,"positive_joint_state_fraction_pass":joint_fraction,"budget_50_success_gain":success_gain,"budget_50_compute_reduction":compute_reduction,"budget_joint_gate":budget_joint,"state_axis_beats_fixed":state_axis,"visual_budget_gate":visual_budget,"downstream_cannot_reverse_upstream_failure":True}
 atomic_json(a.output,result); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
