#!/usr/bin/env python3
from __future__ import annotations

import argparse,json,os,queue,subprocess,sys,threading,time
from dataclasses import dataclass
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; sys.path.insert(0,str(HERE))
from common import MODEL_SEEDS,PROTOCOL_ID,sha256_file,write_json_new

@dataclass(frozen=True)
class Job: name:str; command:tuple[str,...]; marker:Path

def valid(path:Path)->bool:
    if not path.is_file(): return False
    try: value=json.loads(path.read_text())
    except Exception:return False
    return value.get("protocol_id")==PROTOCOL_ID and any(word in str(value.get("status","")) for word in ("COMPLETE","FROZEN","_PASS","NO_GO"))

def execute(job:Job,gpu:int,logs:Path)->None:
    if valid(job.marker): print(f"STAGE26_RESUME_SKIP gpu={gpu} job={job.name}",flush=True);return
    logs.mkdir(parents=True,exist_ok=True); log=logs/f"{job.name}.log"; env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);env["PYTHONUNBUFFERED"]="1"
    print(f"STAGE26_JOB_START gpu={gpu} job={job.name}",flush=True)
    with log.open("ab",buffering=0) as handle: result=subprocess.run(job.command,env=env,stdout=handle,stderr=subprocess.STDOUT,check=False)
    if result.returncode or not valid(job.marker): raise RuntimeError(f"job failed rc={result.returncode} name={job.name} log={log}")
    print(f"STAGE26_JOB_COMPLETE gpu={gpu} job={job.name}",flush=True)

def parallel(jobs:list[Job],gpus:int,logs:Path)->None:
    pending:queue.Queue[Job]=queue.Queue();[pending.put(j) for j in jobs]; errors=[];lock=threading.Lock()
    def worker(gpu:int)->None:
        while True:
            with lock:
                if errors:return
            try:job=pending.get_nowait()
            except queue.Empty:return
            try:execute(job,gpu,logs)
            except BaseException as exc:
                with lock:errors.append(exc)
            finally:pending.task_done()
    threads=[threading.Thread(target=worker,args=(i,)) for i in range(gpus)];[t.start() for t in threads];[t.join() for t in threads]
    if errors:raise errors[0]

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--result-root",type=Path,required=True);p.add_argument("--gpu-count",type=int,choices=range(2,9),required=True);a=p.parse_args()
    root=a.result_root;root.mkdir(parents=True,exist_ok=True);complete=root/"FORMAL_COMPLETE.json"
    if valid(complete):print("STAGE26_FORMAL_ALREADY_COMPLETE");return
    bindings=json.loads((ROOT/"manifests/checkpoint_bindings.json").read_text());seed_bank=ROOT/"manifests/seed_banks.json";logs=root/"logs"
    manifest=root/"FORMAL_RUN_MANIFEST.json"
    if not manifest.exists():write_json_new(manifest,{"protocol_id":PROTOCOL_ID,"status":"FORMAL_INPUTS_FROZEN","run_id":a.run_id,"gpu_count":a.gpu_count,"uid":os.getuid(),"gid":os.getgid(),"source_commit":subprocess.check_output(["git","-C",str(ROOT.parents[1]),"rev-parse","HEAD"],text=True).strip(),"source_tree":subprocess.check_output(["git","-C",str(ROOT.parents[1]),"rev-parse","HEAD^{tree}"],text=True).strip(),"protocol_freeze_sha256":sha256_file(ROOT/"PROTOCOL_FREEZE.json"),"seed_bank_sha256":sha256_file(seed_bank),"checkpoint_bindings_sha256":sha256_file(ROOT/"manifests/checkpoint_bindings.json"),"started_at_unix":time.time()})
    collect=[]
    for binding in bindings["checkpoints"]:
        seed=int(binding["model_seed"])
        for bank in ("train_source","calibration"):
            out=root/"counterfactual_data"/f"seed_{seed}"/bank
            collect.append(Job(f"collect-s{seed}-{bank}",(sys.executable,str(HERE/"collect_counterfactual_data.py"),"--model-seed",str(seed),"--checkpoint",binding["path"],"--checkpoint-sha256",binding["sha256"],"--seed-bank",str(seed_bank),"--bank",bank,"--output-dir",str(out),"--num-envs","16"),out/"COLLECTION_COMPLETE.json"))
    parallel(collect,a.gpu_count,logs/"collection")
    if not (root/"FIRST_REAL_WORK.json").exists():write_json_new(root/"FIRST_REAL_WORK.json",{"protocol_id":PROTOCOL_ID,"status":"FIRST_COMMITTED_COLLECTION_SHARD_PROVEN","evidence":str(collect[0].marker),"sha256":sha256_file(collect[0].marker),"uid":os.getuid(),"gid":os.getgid()})
    fidelity=[]
    for binding in bindings["checkpoints"]:
        seed=int(binding["model_seed"]);out=root/"fidelity"/f"seed_{seed}"
        fidelity.append(Job(f"fidelity-s{seed}",(sys.executable,str(HERE/"audit_shared_prefix.py"),"--model-seed",str(seed),"--checkpoint",binding["path"],"--checkpoint-sha256",binding["sha256"],"--capsule-manifest",str(root/"counterfactual_data"/f"seed_{seed}"/"calibration"/"capsules.jsonl"),"--output-dir",str(out),"--states","64"),out/"SHARED_PREFIX_FIDELITY.json"))
    parallel(fidelity,a.gpu_count,logs/"fidelity")
    predictors=root/"predictors";predictor_job=Job("train-predictors",(sys.executable,str(HERE/"train_completion_predictors.py"),"--data-root",str(root/"counterfactual_data"),"--output-dir",str(predictors),"--epochs","50"),predictors/"PREDICTOR_CALIBRATION_FREEZE.json");execute(predictor_job,0,logs/"predictor")
    initial_modes=("fixed_horizon","learned_counterfactual_completion_gate","learned_success_only_classifier","privileged_neutral_after_hold5","privileged_terminate_first_success")
    jobs=[]
    for binding in bindings["checkpoints"]:
        seed=int(binding["model_seed"])
        for mode in initial_modes:
            out=root/"closed_loop"/f"seed_{seed}"/mode
            jobs.append(Job(f"closed-s{seed}-{mode}",(sys.executable,str(HERE/"evaluate_closed_loop.py"),"--model-seed",str(seed),"--checkpoint",binding["path"],"--checkpoint-sha256",binding["sha256"],"--seed-bank",str(seed_bank),"--predictor",str(predictors/f"seed_{seed}"/"predictor.pt"),"--calibration-freeze",str(predictors/"PREDICTOR_CALIBRATION_FREEZE.json"),"--mode",mode,"--output-dir",str(out),"--num-envs","20"),out/"CLOSED_LOOP_COMPLETE.json"))
    parallel(jobs,a.gpu_count,logs/"closed_initial")
    matched=[]
    for binding in bindings["checkpoints"]:
        seed=int(binding["model_seed"]);profile=root/"closed_loop"/f"seed_{seed}"/"learned_counterfactual_completion_gate"/"summary.json"
        for mode in ("fixed_time_matched_stop","random_matched_stop"):
            out=root/"closed_loop"/f"seed_{seed}"/mode
            matched.append(Job(f"closed-s{seed}-{mode}",(sys.executable,str(HERE/"evaluate_closed_loop.py"),"--model-seed",str(seed),"--checkpoint",binding["path"],"--checkpoint-sha256",binding["sha256"],"--seed-bank",str(seed_bank),"--predictor",str(predictors/f"seed_{seed}"/"predictor.pt"),"--calibration-freeze",str(predictors/"PREDICTOR_CALIBRATION_FREEZE.json"),"--mode",mode,"--matched-profile",str(profile),"--output-dir",str(out),"--num-envs","20"),out/"CLOSED_LOOP_COMPLETE.json"))
    parallel(matched,a.gpu_count,logs/"closed_matched")
    summary=root/"key-results/stage26_summary.json";summary.parent.mkdir(parents=True,exist_ok=True)
    execute(Job("summarize",(sys.executable,str(HERE/"summarize_stage26.py"),"--result-root",str(root/"closed_loop"),"--predictor-freeze",str(predictors/"PREDICTOR_CALIBRATION_FREEZE.json"),"--fidelity-root",str(root/"fidelity"),"--output",str(summary)),summary),0,logs/"audit")
    audit=root/"key-results/independent_stage26_audit.json"
    execute(Job("audit",(sys.executable,str(HERE/"audit_stage26.py"),"--result-root",str(root/"closed_loop"),"--data-root",str(root/"counterfactual_data"),"--predictor-freeze",str(predictors/"PREDICTOR_CALIBRATION_FREEZE.json"),"--fidelity-root",str(root/"fidelity"),"--summary",str(summary),"--output",str(audit)),audit),0,logs/"audit")
    summary_value=json.loads(summary.read_text());write_json_new(complete,{"protocol_id":PROTOCOL_ID,"status":"ALL_PREREGISTERED_STAGE26_EXPERIMENTS_COMPLETE","run_id":a.run_id,"final_status":summary_value["final_status"],"summary_sha256":sha256_file(summary),"independent_audit_sha256":sha256_file(audit),"all_planned_experiments_executed":True,"completed_at_unix":time.time()})

if __name__=="__main__":main()
