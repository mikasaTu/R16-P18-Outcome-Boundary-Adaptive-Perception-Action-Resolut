#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR=Path(__file__).resolve().parent; sys.path.insert(0,str(SCRIPT_DIR))
from common import MODEL_SEEDS, PROTOCOL_ID, read_jsonl, write_json_new


def bootstrap(values:list[float],seed:int=16018)->list[float]:
    data=np.asarray(values,dtype=float); rng=np.random.default_rng(seed); means=np.empty(10000)
    for start in range(0,10000,1000):
        idx=rng.integers(0,len(data),size=(1000,len(data))); means[start:start+1000]=data[idx].mean(axis=1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def paired(root:Path,first:str,second:str)->dict[str,Any]:
    per_seed={}; episode=defaultdict(list)
    for seed in MODEL_SEEDS:
        a={r["episode_seed"]:r for r in read_jsonl(root/f"seed_{seed}"/first/"episodes.jsonl")}; b={r["episode_seed"]:r for r in read_jsonl(root/f"seed_{seed}"/second/"episodes.jsonl")}
        values=[float(b[k]["success_at_end"])-float(a[k]["success_at_end"]) for k in sorted(a)]; per_seed[str(seed)]={"gain":float(np.mean(values)),"ci":bootstrap(values)}
        for k,v in zip(sorted(a),values): episode[k].append(v)
    clustered=[float(np.mean(episode[k])) for k in sorted(episode)]
    return {"gain":float(np.mean(clustered)),"ci":bootstrap(clustered),"per_model_seed":per_seed,"positive_model_seeds":sum(v["gain"]>0 for v in per_seed.values()),"nonnegative_model_seeds":sum(v["gain"]>=0 for v in per_seed.values())}


def arm_metrics(root:Path,mode:str)->dict[str,Any]:
    result={"per_model_seed":{}}
    rows=[]
    for seed in MODEL_SEEDS:
        current=read_jsonl(root/f"seed_{seed}"/mode/"episodes.jsonl"); rows+=current
        result["per_model_seed"][str(seed)]={key:float(np.mean([r[key] for r in current])) for key in ("success_once","success_at_end","post_success_loss")}
    result["aggregate"]={key:float(np.mean([r[key] for r in rows])) for key in ("success_once","success_at_end","post_success_loss")}
    result["aggregate"].update({"stop_rate":float(np.mean([r["stop_step"] is not None for r in rows])),"policy_calls":int(sum(r["policy_calls"] for r in rows)),"policy_latency_seconds":float(sum(r["policy_latency_seconds"] for r in rows)),"completion_head_latency_seconds":float(sum(r["completion_head_latency_seconds"] for r in rows))})
    return result


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--result-root",type=Path,required=True); parser.add_argument("--predictor-freeze",type=Path,required=True); parser.add_argument("--fidelity-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    modes=("fixed_horizon","fixed_time_matched_stop","random_matched_stop","learned_success_only_classifier","learned_counterfactual_completion_gate","privileged_neutral_after_hold5","privileged_terminate_first_success")
    arms={mode:arm_metrics(args.result_root,mode) for mode in modes}; causal=paired(args.result_root,"fixed_horizon","privileged_terminate_first_success"); learned=paired(args.result_root,"fixed_horizon","learned_counterfactual_completion_gate"); success_only=paired(args.result_root,"fixed_horizon","learned_success_only_classifier")
    fixed=arms["fixed_horizon"]["aggregate"]; learned_arm=arms["learned_counterfactual_completion_gate"]["aggregate"]
    recovery=learned["gain"]/causal["gain"] if causal["gain"]>0 else 0.0
    once_loss=fixed["success_once"]-learned_arm["success_once"]
    loss_reduction=(fixed["post_success_loss"]-learned_arm["post_success_loss"])/fixed["post_success_loss"] if fixed["post_success_loss"]>0 else 0.0
    latency_overhead=learned_arm["completion_head_latency_seconds"]/fixed["policy_latency_seconds"] if fixed["policy_latency_seconds"]>0 else float("inf")
    causal_gate=bool(causal["gain"]>=.10 and causal["ci"][0]>0 and causal["positive_model_seeds"]==3)
    freeze=json.loads(args.predictor_freeze.read_text()); offline_gate=bool(freeze["offline_learnability_gate_pass"])
    learned_gate=bool((learned["gain"]>=.08 or recovery>=.50) and learned["ci"][0]>0 and once_loss<=.02 and loss_reduction>=.30 and learned["nonnegative_model_seeds"]==3 and sum(v["gain"]>0 and v["ci"][0]>0 for v in learned["per_model_seed"].values())>=2 and learned_arm["policy_calls"]<=fixed["policy_calls"] and latency_overhead<=.10)
    fidelity={str(seed):json.loads((args.fidelity_root/f"seed_{seed}"/"SHARED_PREFIX_FIDELITY.json").read_text()) for seed in MODEL_SEEDS}; fidelity_gate=all(v["pass"] for v in fidelity.values())
    selected_metrics=[freeze["selected_checkpoints"][str(seed)]["calibration"] for seed in MODEL_SEEDS]; false_stop_high=any(v["not_done_false_stop"]>.05 for v in selected_metrics)
    if not fidelity_gate: final="NO_GO_SHARED_PREFIX_FIDELITY"
    elif not causal_gate: final="NO_GO_STOPPING_NOT_CAUSAL"
    elif not offline_gate: final="NO_GO_COMPLETION_NOT_LEARNABLE"
    elif learned["gain"]>0 and false_stop_high: final="REVISE_EARLY_STOP_FALSE_POSITIVE"
    elif learned_gate: final="GO_STOP_NORMALIZED_BASELINE"
    else: final="NO_GO_COMPLETION_NOT_LEARNABLE"
    write_json_new(args.output,{"protocol_id":PROTOCOL_ID,"status":"STAGE26_SUMMARY_COMPLETE","final_status":final,"all_user_mandated_experiments_executed_despite_gates":True,"shared_prefix_fidelity":fidelity,"gates":{"shared_prefix":fidelity_gate,"causal":causal_gate,"offline_learnability":offline_gate,"learned_closed_loop":learned_gate},"arms":arms,"comparisons":{"privileged_terminate_vs_fixed":causal,"learned_counterfactual_vs_fixed":learned,"learned_success_only_vs_fixed":success_only},"learned_recovery_fraction_of_privileged_gain":recovery,"success_once_reduction":once_loss,"post_success_loss_relative_reduction":loss_reduction,"completion_head_latency_overhead_fraction":latency_overhead,"predictor":freeze,"decision_precedence":["NO_GO_SHARED_PREFIX_FIDELITY","NO_GO_STOPPING_NOT_CAUSAL","NO_GO_COMPLETION_NOT_LEARNABLE","REVISE_EARLY_STOP_FALSE_POSITIVE","GO_STOP_NORMALIZED_BASELINE"],"stage27_draft_required":final=="GO_STOP_NORMALIZED_BASELINE"})


if __name__=="__main__": main()
