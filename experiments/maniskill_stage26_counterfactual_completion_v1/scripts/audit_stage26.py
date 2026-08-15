#!/usr/bin/env python3
from __future__ import annotations

import argparse,json,sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from common import MODEL_SEEDS,PROTOCOL_ID,read_jsonl,sha256_file,write_json_new


def ci(values:list[float])->list[float]:
    x=np.asarray(values,float); rng=np.random.default_rng(16018); means=[]
    for _ in range(10):
        idx=rng.integers(0,len(x),size=(1000,len(x))); means.append(x[idx].mean(1))
    all_means=np.concatenate(means); return [float(np.quantile(all_means,.025)),float(np.quantile(all_means,.975))]


def comparison(root:Path,arm:str)->dict[str,Any]:
    clustered=defaultdict(list); seed_gain={}
    for seed in MODEL_SEEDS:
        fixed={r["episode_seed"]:r for r in read_jsonl(root/f"seed_{seed}"/"fixed_horizon"/"episodes.jsonl")}; other={r["episode_seed"]:r for r in read_jsonl(root/f"seed_{seed}"/arm/"episodes.jsonl")}
        values=[]
        for episode in sorted(fixed):
            value=float(other[episode]["success_at_end"])-float(fixed[episode]["success_at_end"]); values.append(value); clustered[episode].append(value)
        seed_gain[str(seed)]=float(np.mean(values))
    values=[float(np.mean(clustered[k])) for k in sorted(clustered)]
    return {"gain":float(np.mean(values)),"ci":ci(values),"seed_gain":seed_gain}


def raw_arm(root:Path,arm:str)->dict[str,float]:
    rows=[]
    for seed in MODEL_SEEDS: rows+=read_jsonl(root/f"seed_{seed}"/arm/"episodes.jsonl")
    return {"once":float(np.mean([r["success_once"] for r in rows])),"end":float(np.mean([r["success_at_end"] for r in rows])),"loss":float(np.mean([r["post_success_loss"] for r in rows])),"calls":float(sum(r["policy_calls"] for r in rows)),"policy_latency":float(sum(r["policy_latency_seconds"] for r in rows)),"head_latency":float(sum(r["completion_head_latency_seconds"] for r in rows))}


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--result-root",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--predictor-freeze",type=Path,required=True); p.add_argument("--fidelity-root",type=Path,required=True); p.add_argument("--summary",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    problems=[]; label_rows=0
    for seed in MODEL_SEEDS:
        for bank in ("train_source","calibration"):
            manifest=read_jsonl(a.data_root/f"seed_{seed}"/bank/"capsules.jsonl"); branches=read_jsonl(a.data_root/f"seed_{seed}"/bank/"branches.jsonl"); by={(r["capsule_id"],r["branch"]):r for r in branches}
            for row in manifest:
                for field,branch in (("hold_success_20","neutral_hold"),("continue_success_20","continue_policy"),("reobserve_success_20","hold_then_reobserve")):
                    if bool(row[field])!=bool(by[(row["capsule_id"],branch)]["success_at_horizon"]): problems.append(f"branch_label:{seed}:{bank}:{row['capsule_id']}:{field}")
                label_rows+=1
    fidelity={};
    for seed in MODEL_SEEDS:
        rows=read_jsonl(a.fidelity_root/f"seed_{seed}"/"shared_prefix_fidelity_raw.jsonl"); fidelity[str(seed)]=bool(len(rows)==640 and max(r["action_max_abs"] for r in rows)<=1e-6 and max(r["object_translation_m"] for r in rows)<=1e-5 and max(r["object_rotation_rad"] for r in rows)<=1e-4 and all(r["categorical_agreement"] for r in rows))
    causal=comparison(a.result_root,"privileged_terminate_first_success"); learned=comparison(a.result_root,"learned_counterfactual_completion_gate"); fixed=raw_arm(a.result_root,"fixed_horizon"); learned_arm=raw_arm(a.result_root,"learned_counterfactual_completion_gate")
    freeze=json.loads(a.predictor_freeze.read_text()); offline=bool(freeze["offline_learnability_gate_pass"]); recovery=learned["gain"]/causal["gain"] if causal["gain"]>0 else 0.; once_loss=fixed["once"]-learned_arm["once"]; loss_reduction=(fixed["loss"]-learned_arm["loss"])/fixed["loss"] if fixed["loss"] else 0.; latency=learned_arm["head_latency"]/fixed["policy_latency"] if fixed["policy_latency"] else 999.
    causal_gate=causal["gain"]>=.10 and causal["ci"][0]>0 and all(v>0 for v in causal["seed_gain"].values()); learned_gate=(learned["gain"]>=.08 or recovery>=.5) and learned["ci"][0]>0 and once_loss<=.02 and loss_reduction>=.3 and all(v>=0 for v in learned["seed_gain"].values()) and sum(v>0 for v in learned["seed_gain"].values())>=2 and learned_arm["calls"]<=fixed["calls"] and latency<=.1
    false_high=any(freeze["selected_checkpoints"][str(seed)]["calibration"]["not_done_false_stop"]>.05 for seed in MODEL_SEEDS)
    if not all(fidelity.values()): final="NO_GO_SHARED_PREFIX_FIDELITY"
    elif not causal_gate: final="NO_GO_STOPPING_NOT_CAUSAL"
    elif not offline: final="NO_GO_COMPLETION_NOT_LEARNABLE"
    elif learned["gain"]>0 and false_high: final="REVISE_EARLY_STOP_FALSE_POSITIVE"
    elif learned_gate: final="GO_STOP_NORMALIZED_BASELINE"
    else: final="NO_GO_COMPLETION_NOT_LEARNABLE"
    reported=json.loads(a.summary.read_text()); checks={"final":final==reported["final_status"],"causal_gain":np.isclose(causal["gain"],reported["comparisons"]["privileged_terminate_vs_fixed"]["gain"]),"learned_gain":np.isclose(learned["gain"],reported["comparisons"]["learned_counterfactual_vs_fixed"]["gain"]),"fidelity":all(fidelity.values())==reported["gates"]["shared_prefix"],"offline":offline==reported["gates"]["offline_learnability"]}
    passed=all(checks.values()) and not problems
    write_json_new(a.output,{"protocol_id":PROTOCOL_ID,"status":"INDEPENDENT_STAGE26_AUDIT_PASS" if passed else "INDEPENDENT_STAGE26_AUDIT_FAIL","audit_pass":passed,"decision_logic_independent_from_summarizer":True,"branch_label_rows_recomputed":label_rows,"problems":problems,"checks":checks,"independent":{"fidelity":fidelity,"causal":causal,"offline":offline,"learned":learned,"recovery_fraction":recovery,"once_loss":once_loss,"loss_reduction":loss_reduction,"latency_overhead":latency,"final_status":final},"summary_sha256":sha256_file(a.summary)})

if __name__=="__main__":main()
