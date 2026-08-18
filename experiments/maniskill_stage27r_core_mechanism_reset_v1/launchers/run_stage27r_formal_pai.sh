#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ "$(id -u):$(id -g)" == "2254:2254" ]] || exit 73
readonly source_root="${R16P18_SOURCE_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage26-formal-source-v4}"
readonly exp="${source_root}/experiments/maniskill_stage27r_core_mechanism_reset_v1"
readonly py="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python"
readonly ms="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/ManiSkill-r16p18-v3.0.1"
readonly overlay="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r16p18-maniskill-act-v301-overlay/site-packages"
readonly training="${R16P18_TRAINING_ROOT:-/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage27r-core-reset-v1/stage27r-data-train-v12}"
readonly data_root="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-stage27r-core-reset-v2"
readonly old_data="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-act-boundary-screen-v1/official_demos"
readonly expert_root="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-stage27r-expert-pool-v1"
readonly producer_registry_run="${R16P18_PRODUCER_REGISTRY_RUN:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry/runs/stage27r-formal-idle-v9}"
readonly producer_registry_evidence="${R16P18_PRODUCER_REGISTRY_EVIDENCE:-${producer_registry_run}/resolved.json}"
readonly producer_source_root="${R16P18_PRODUCER_SOURCE_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage27r-formal-source-v8}"
readonly producer_launcher="${producer_source_root}/experiments/maniskill_stage27r_core_mechanism_reset_v1/launchers/run_stage27r_formal_pai.sh"
readonly producer_job_id="dlc9nkd8q7u4szm3"
test "${R16P18_PRODUCER_JOB_ID:-${producer_job_id}}" = "${producer_job_id}"
readonly producer_run_id="stage27r-formal-idle-v9"
test "${R16P18_PRODUCER_RUN_ID:-${producer_run_id}}" = "${producer_run_id}"
readonly old_producer_terminal="${OLD_PRODUCER_TERMINAL:-${R16P18_OLD_PRODUCER_TERMINAL:?external OLD_PRODUCER_TERMINAL/no-overlap JSON is required}}"
readonly continuation_registry_run="${R16P18_CONTINUATION_REGISTRY_RUN:?external continuation registry run directory is required}"
readonly continuation_registry_evidence="${R16P18_CONTINUATION_REGISTRY_EVIDENCE:-${continuation_registry_run}/resolved.json}"
readonly pai_source_manifest="${R16P18_PAI_SOURCE_MANIFEST:?external source manifest is required}"
readonly run_id="${PAI_CANARY_RUN_ID:?}"
readonly pai_job_hint="${PAI_CANARY_JOB_ID:-}"
readonly gpu_count="${PAI_CANARY_EXPECTED_GPUS:?}"
readonly result="${R16P18_FORMAL_RESULT_ROOT:-/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage27r-formal-v1/${run_id}}"
readonly artifact="${PAI_CANARY_RUN_DIR:?}"
readonly expected_project_commit="${R16P18_EXPECTED_PROJECT_COMMIT:?}"
readonly expected_project_tree="${R16P18_EXPECTED_PROJECT_TREE:?}"
test "${continuation_registry_run}" != "${producer_registry_run}" || { echo "continuation registry must be distinct from old producer registry" >&2; exit 74; }
test -f "${pai_source_manifest}" || { echo "continuation source manifest is missing" >&2; exit 74; }

# This is the first executable gate.  It only reads externally-created
# evidence, and obtains the new JobId from resolved.json after PAI creation;
# PAI_CANARY_JOB_ID is merely an optional cross-check, never a source of
# lineage.  In particular this block must fail before creating result/output
# directories when the old v9 job is still Running or evidence is absent.
continuation_job_args=()
if [[ -n "${pai_job_hint}" ]]; then continuation_job_args=(--expected-job-id "${pai_job_hint}"); fi
readonly pai_job_id="$("${py}" "${exp}/scripts/validate_continuation_evidence.py" \
  --old-terminal "${old_producer_terminal}" --old-job-id "${producer_job_id}" --old-run-id "${producer_run_id}" \
  --registry-run "${continuation_registry_run}" --registry-evidence "${continuation_registry_evidence}" \
  --expected-run-id "${run_id}" --expected-source-commit "${expected_project_commit}" \
  --expected-source-tree "${expected_project_tree}" \
  --expected-launcher "${exp}/launchers/run_stage27r_formal_pai.sh" \
  --expected-source-manifest "${pai_source_manifest}" "${continuation_job_args[@]}" --print-job-id)"
test -f "${training}/DATA_AND_TRAINING_COMPLETE.json"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_project_commit}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_project_tree}"
test -z "$(git -C "${source_root}" status --porcelain)"
mkdir -p "${result}" "${expert_root}"; cd "${artifact}"
exec > >(tee -a "${artifact}/formal-runtime.log") 2>&1
export PYTHONPATH="${exp}/scripts:${overlay}:${ms}/examples/baselines/act:${ms}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -f "${result}/PRECHECKS.json" ]]; then
  "${py}" -m compileall -q "${exp}/scripts"
  "${py}" -m pytest -q "${exp}/tests" | tee "${artifact}/preflight-pytest.log"
  "${py}" "${exp}/scripts/deterministic_lockstep_smoke.py" --output "${result}/DETERMINISTIC_LOCKSTEP_SMOKE.json"
  "${py}" -c 'import json,sys; smoke=json.load(open(sys.argv[1])); json.dump({"compileall":True,"unit_tests":True,"deterministic_smoke":bool(smoke["pass"]),"deterministic_smoke_evidence":sys.argv[1],"fail_on_overwrite":True,"all_pass":bool(smoke["pass"])},open(sys.argv[2],"x"),indent=2)' "${result}/DETERMINISTIC_LOCKSTEP_SMOKE.json" "${result}/PRECHECKS.json"
else
  "${py}" - "${result}/PRECHECKS.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text(encoding="utf-8"))
if d.get("all_pass") is not True or d.get("compileall") is not True or d.get("unit_tests") is not True or d.get("fail_on_overwrite") is not True:
    raise SystemExit(f"stale/invalid PRECHECKS.json: {p}")
print(f"PRECHECKS_VALID PASS {p}")
PY
fi
if [[ -f "${result}/EXACT_DATASET_AUDIT.json" ]]; then
  "${py}" - "${result}/EXACT_DATASET_AUDIT.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text(encoding="utf-8"))
if d.get("status") != "PASS" or not isinstance(d.get("task_checks"), dict): raise SystemExit(f"stale/invalid EXACT_DATASET_AUDIT.json: {p}")
print(f"EXACT_DATASET_AUDIT_VALID PASS {p}")
PY
else
  "${py}" "${exp}/scripts/audit_exact_dataset.py" --dataset-root "${data_root}" --output "${result}/EXACT_DATASET_AUDIT.json"
fi

screen_dir="${result}/screen"; mkdir -p "${screen_dir}"
tasks=(StackCube-v1 PegInsertionSide-v1 PlugCharger-v1 PullCubeTool-v1 PushT-v1 PushCube-v1); seeds=(16018 16019 16020)
jobs=(); for task in "${tasks[@]}"; do for seed in "${seeds[@]}"; do jobs+=("${task} ${seed}"); done; done
screen_worker(){ local gpu=$1 i task seed out; for ((i=gpu;i<${#jobs[@]};i+=gpu_count)); do read -r task seed <<<"${jobs[$i]}"; out="${screen_dir}/${task}-seed${seed}.json"; if [[ -f "${out}" ]]; then :; else CUDA_VISIBLE_DEVICES=${gpu} "${py}" "${exp}/scripts/screen_one.py" --training-run "${training}" --task "${task}" --seed "${seed}" --output "${out}"; fi; "${py}" "${exp}/scripts/validate_screen_shard.py" --path "${out}" --task "${task}" --seed "${seed}"; done; }
pids=(); for ((g=0;g<gpu_count;g++)); do screen_worker "$g" & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done
if [[ ! -f "${screen_dir}/TASK_SELECTION.json" ]]; then
  "${py}" "${exp}/scripts/aggregate_screen.py" --shard-dir "${screen_dir}" --output "${screen_dir}/TASK_SELECTION.json"
fi
"${py}" - "${screen_dir}/TASK_SELECTION.json" <<'PY'
import json, sys
from pathlib import Path
from common import PROTOCOL_ID

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("protocol_id") != PROTOCOL_ID:
    raise SystemExit(f"screen selection protocol mismatch: {path}")
expected = {f"{task}/seed_{seed}" for task in ("StackCube-v1", "PegInsertionSide-v1", "PlugCharger-v1", "PullCubeTool-v1", "PushT-v1", "PushCube-v1") for seed in (16018, 16019, 16020)}
if set(payload.get("groups", {})) != expected:
    raise SystemExit(f"screen selection group set mismatch: {path}")
for key, group in payload["groups"].items():
    if group.get("protocol_id") != PROTOCOL_ID or group.get("selected") is None:
        raise SystemExit(f"screen selection group incomplete: {key}")
    if len(group.get("validated_top2", [])) < 2:
        raise SystemExit(f"screen selection top2 incomplete: {key}")
print(f"SCREEN_SELECTION_VALID PASS {path}")
PY
positive2=$("${py}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_positive"] or "PegInsertionSide-v1")' "${screen_dir}/TASK_SELECTION.json")
negative=$("${py}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_negative"] or "")' "${screen_dir}/TASK_SELECTION.json")
formal_tasks=(StackCube-v1 "${positive2}"); test -z "${negative}" || formal_tasks+=("${negative}")

official_h5(){ if [[ "$1" == PushT-v1 ]]; then echo "${old_data}/$1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"; else echo "${old_data}/$1/motionplanning/trajectory.h5"; fi; }
pids=(); for task in "${formal_tasks[@]}"; do "${py}" "${exp}/scripts/prepare_expert_pool.py" --task "$task" --official-h5 "$(official_h5 "$task")" --output-root "${expert_root}" --python "${py}" & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done
for task in "${formal_tasks[@]}"; do
  "${py}" - "${expert_root}/${task}/EXPERT_POOL_COMPLETE.json" "$task" <<'PY'
import hashlib, json, sys
from pathlib import Path
from common import PROTOCOL_ID
p, task = Path(sys.argv[1]), sys.argv[2]
d = json.loads(p.read_text(encoding="utf-8"))
h5 = Path(d.get("h5", ""))
if d.get("protocol_id") != PROTOCOL_ID or d.get("status") != "PASS" or d.get("task") != task or int(d.get("successful", 0)) < 72 or not h5.is_file():
    raise SystemExit(f"invalid expert-pool completion: {p}")
h = hashlib.sha256()
with h5.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
if d.get("h5_sha256") != h.hexdigest(): raise SystemExit(f"expert-pool H5 hash mismatch: {p}")
print(f"EXPERT_POOL_VALID PASS {p}")
PY
done
checkpoint(){ "${py}" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["groups"][f"{sys.argv[2]}/seed_{sys.argv[3]}"]["selected"]["path"])' "${screen_dir}/TASK_SELECTION.json" "$1" "$2"; }
expert_h5(){ "${py}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["h5"])' "${expert_root}/$1/EXPERT_POOL_COMPLETE.json"; }

bank_dir="${result}/state_banks"; mkdir -p "${bank_dir}"
for task in "${formal_tasks[@]}"; do
  ck=$(checkpoint "$task" 16018); eh=$(expert_h5 "$task")
  if [[ "$task" == "$negative" && -n "$negative" ]]; then bank=negative; count_offset=0; else bank=calibration; count_offset=0; fi
  if [[ "$bank" == calibration ]]; then
    if [[ -f "${bank_dir}/${task}-calibration.json" ]]; then
      :
    else
      CUDA_VISIBLE_DEVICES=0 "${py}" "${exp}/scripts/build_lockstep_state_bank.py" --task "$task" --model-seed 16018 --checkpoint "$ck" --expert-h5 "$eh" --expert-offset 0 --bank calibration --output "${bank_dir}/${task}-calibration.json"
    fi
    "${py}" "${exp}/scripts/validate_state_bank.py" --path "${bank_dir}/${task}-calibration.json" --task "$task" --bank calibration
    if [[ -f "${bank_dir}/${task}-confirmatory.json" ]]; then
      :
    else
      CUDA_VISIBLE_DEVICES=0 "${py}" "${exp}/scripts/build_lockstep_state_bank.py" --task "$task" --model-seed 16018 --checkpoint "$ck" --expert-h5 "$eh" --expert-offset 24 --bank confirmatory --output "${bank_dir}/${task}-confirmatory.json"
    fi
    "${py}" "${exp}/scripts/validate_state_bank.py" --path "${bank_dir}/${task}-confirmatory.json" --task "$task" --bank confirmatory
  else
    if [[ -f "${bank_dir}/${task}-negative.json" ]]; then
      :
    else
      CUDA_VISIBLE_DEVICES=0 "${py}" "${exp}/scripts/build_lockstep_state_bank.py" --task "$task" --model-seed 16018 --checkpoint "$ck" --expert-h5 "$eh" --expert-offset 0 --bank negative --output "${bank_dir}/${task}-negative.json"
    fi
    "${py}" "${exp}/scripts/validate_state_bank.py" --path "${bank_dir}/${task}-negative.json" --task "$task" --bank negative
  fi
done

cal="${result}/calibration"; mkdir -p "${cal}"; jobs=()
for task in "${formal_tasks[@]:0:2}"; do for seed in "${seeds[@]}"; do for grid in 2 4; do jobs+=("${task} ${seed} ${grid}"); done; done; done
cal_worker(){ local gpu=$1 i task seed grid out; for ((i=gpu;i<${#jobs[@]};i+=gpu_count)); do read -r task seed grid <<<"${jobs[$i]}"; out="${cal}/${task}-seed${seed}-grid${grid}.json"; if [[ -f "$out" ]]; then :; else CUDA_VISIBLE_DEVICES=$gpu "${py}" "${exp}/scripts/run_factorial_oracle.py" --state-bank "${bank_dir}/${task}-calibration.json" --checkpoint "$(checkpoint "$task" "$seed")" --model-seed "$seed" --tile-grid "$grid" --output "$out"; fi; "${py}" "${exp}/scripts/validate_oracle_shard.py" --oracle "$out" --task "$task" --model-seed "$seed" --bank calibration --grid "$grid" --state-bank "${bank_dir}/${task}-calibration.json"; done; }
pids=(); for ((g=0;g<gpu_count;g++)); do cal_worker "$g" & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done
for task in "${formal_tasks[@]:0:2}"; do if [[ -f "${cal}/${task}-CROP_GRID_FREEZE.json" ]]; then "${py}" - "${cal}/${task}-CROP_GRID_FREEZE.json" <<'PY'
import json, sys
from pathlib import Path
from common import PROTOCOL_ID
p = Path(sys.argv[1]); d = json.loads(p.read_text(encoding="utf-8"))
if d.get("protocol_id") != PROTOCOL_ID or int(d.get("selected_grid", -1)) not in (2, 4): raise SystemExit(f"invalid crop-grid freeze: {p}")
if not all(field in d for field in ("grid2_oracle_gain", "grid4_oracle_gain", "recovery_ratio", "threshold")): raise SystemExit(f"crop-grid evidence missing: {p}")
print(f"CROP_GRID_FREEZE_VALID PASS {p}")
PY
else "${py}" "${exp}/scripts/select_crop_grid.py" --grid2 "${cal}/${task}"-seed*-grid2.json --grid4 "${cal}/${task}"-seed*-grid4.json --output "${cal}/${task}-CROP_GRID_FREEZE.json"; fi; done
for task in "${formal_tasks[@]:0:2}"; do "${py}" - "${cal}/${task}-CROP_GRID_FREEZE.json" <<'PY'
import json, sys
from pathlib import Path
from common import PROTOCOL_ID
p = Path(sys.argv[1]); d = json.loads(p.read_text(encoding="utf-8"))
if d.get("protocol_id") != PROTOCOL_ID or int(d.get("selected_grid", -1)) not in (2, 4): raise SystemExit(f"invalid crop-grid freeze: {p}")
if not all(field in d for field in ("grid2_oracle_gain", "grid4_oracle_gain", "recovery_ratio", "threshold")): raise SystemExit(f"crop-grid evidence missing: {p}")
print(f"CROP_GRID_FREEZE_VALID PASS {p}")
PY
done

oracle="${result}/oracle"; mkdir -p "${oracle}"
"${py}" "${exp}/scripts/record_oracle_inputs.py" --formal-root "${result}" --output "${result}/ORACLE_INPUT_SNAPSHOT.json" --state-bank-dir "${bank_dir}" --continuation-registry-run "${continuation_registry_run}" --continuation-registry-evidence "${continuation_registry_evidence}" --continuation-run-id "${run_id}" --continuation-job-id "${pai_job_id}" --continuation-source-commit "${expected_project_commit}" --continuation-source-tree "${expected_project_tree}" --continuation-launcher "${exp}/launchers/run_stage27r_formal_pai.sh" --continuation-source-manifest "${pai_source_manifest}" --old-producer-terminal "${old_producer_terminal}" --old-producer-job-id "${producer_job_id}" --old-producer-run-id "${producer_run_id}" --expected-task StackCube-v1 --expected-task "${positive2}" --model-seed 16018 --model-seed 16019 --model-seed 16020
jobs=()
for task in "${formal_tasks[@]:0:2}"; do grid=$("${py}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_grid"])' "${cal}/${task}-CROP_GRID_FREEZE.json"); for seed in "${seeds[@]}"; do jobs+=("${task} ${seed} ${grid} confirmatory"); done; done
if [[ -n "$negative" ]]; then for seed in "${seeds[@]}"; do jobs+=("${negative} ${seed} 2 negative"); done; fi
oracle_worker(){ local gpu=$1 i task seed grid bank out; for ((i=gpu;i<${#jobs[@]};i+=gpu_count)); do read -r task seed grid bank <<<"${jobs[$i]}"; out="${oracle}/${task}-seed${seed}-${bank}.json"; if [[ -f "$out" ]]; then :; else CUDA_VISIBLE_DEVICES=$gpu "${py}" "${exp}/scripts/run_factorial_oracle.py" --state-bank "${bank_dir}/${task}-${bank}.json" --checkpoint "$(checkpoint "$task" "$seed")" --model-seed "$seed" --tile-grid "$grid" --output "$out"; fi; "${py}" "${exp}/scripts/validate_oracle_shard.py" --oracle "$out" --task "$task" --model-seed "$seed" --bank "$bank" --grid "$grid" --state-bank "${bank_dir}/${task}-${bank}.json"; done; }
pids=(); for ((g=0;g<gpu_count;g++)); do oracle_worker "$g" & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done
# The first snapshot above is immutable.  This second invocation records each
# shard that was absent at that instant only after its state-bank semantic
# validation has completed; a restart revalidates the same per-shard hashes.
"${py}" "${exp}/scripts/record_oracle_inputs.py" --formal-root "${result}" --output "${result}/ORACLE_INPUT_SNAPSHOT.json" --state-bank-dir "${bank_dir}" --continuation-registry-run "${continuation_registry_run}" --continuation-registry-evidence "${continuation_registry_evidence}" --continuation-run-id "${run_id}" --continuation-job-id "${pai_job_id}" --continuation-source-commit "${expected_project_commit}" --continuation-source-tree "${expected_project_tree}" --continuation-launcher "${exp}/launchers/run_stage27r_formal_pai.sh" --continuation-source-manifest "${pai_source_manifest}" --old-producer-terminal "${old_producer_terminal}" --old-producer-job-id "${producer_job_id}" --old-producer-run-id "${producer_run_id}" --expected-task StackCube-v1 --expected-task "${positive2}" --model-seed 16018 --model-seed 16019 --model-seed 16020

derived_output(){
  local kind="$1" target="$2"; shift 2
  "${py}" "${exp}/scripts/resume_derived_output.py" --target "$target" \
    --stale-after-seconds 0 \
    --validator="${py}" \
    --validator="${exp}/scripts/validate_derived_output.py" \
    --validator=--path --validator=__TARGET__ --validator=--kind --validator="$kind" -- "$@"
}
lineage_source_manifest_args=()
if [[ -n "${pai_source_manifest}" ]]; then
  lineage_source_manifest_args=(--pai-source-manifest "${pai_source_manifest}")
fi

derived_output oracle_validation "${result}/ORACLE_VALIDATION.json" \
  "${py}" "${exp}/scripts/validate_oracle_collection.py" --formal-root "${result}" --state-bank-dir "${bank_dir}" --expected-task StackCube-v1 --expected-task "${positive2}" --output __OUTPUT__
derived_output lineage "${result}/ORACLE_LINEAGE_MANIFEST.json" \
  "${py}" "${exp}/scripts/oracle_lineage_manifest.py" --formal-root "${result}" --repo "${source_root}" --launcher "${exp}/launchers/run_stage27r_formal_pai.sh" --state-bank-dir "${bank_dir}" --pai-run-id "${run_id}" --pai-job-id "${pai_job_id}" --oracle-input-snapshot "${result}/ORACLE_INPUT_SNAPSHOT.json" --producer-registry-evidence "${producer_registry_evidence}" --producer-registry-run "${producer_registry_run}" --producer-source-root "${producer_source_root}" --producer-launcher "${producer_launcher}" --producer-job-id "${producer_job_id}" --old-producer-terminal "${old_producer_terminal}" --continuation-registry-run "${continuation_registry_run}" --continuation-registry-evidence "${continuation_registry_evidence}" --continuation-job-id "${pai_job_id}" "${lineage_source_manifest_args[@]}" --output __OUTPUT__
derived_output statistics "${result}/statistics.json" \
  "${py}" "${exp}/scripts/analyze_stage27r.py" --inputs "${oracle}"/*.json --output __OUTPUT__
derived_output mechanism "${result}/MECHANISM_AUDIT.json" \
  "${py}" "${exp}/scripts/mechanism_audit.py" --inputs "${oracle}"/*.json --output __OUTPUT__
bank_args=("${bank_dir}/StackCube-v1-confirmatory.json" "${bank_dir}/${positive2}-confirmatory.json"); test -z "$negative" || bank_args+=("${bank_dir}/${negative}-negative.json")
derived_output result "${result}/RESULT_VECTOR.json" \
  "${py}" "${exp}/scripts/decide_stage27r.py" --analysis "${result}/statistics.json" --task-selection "${screen_dir}/TASK_SELECTION.json" --state-banks "${bank_args[@]}" --positive-tasks StackCube-v1 "${positive2}" --output __OUTPUT__
derived_output official_audit "${result}/INDEPENDENT_AUDIT.json" \
  "${py}" "${exp}/scripts/audit_formal_results.py" --repo "${source_root}" --formal-root "${result}" --training-root "${training}" --dataset-root "${data_root}" --output __OUTPUT__
derived_output posthoc_audit "${result}/POSTHOC_INDEPENDENT_AUDIT.json" \
  "${py}" "${exp}/scripts/posthoc_independent_audit.py" --formal-root "${result}" --dataset-root "${data_root}" --maniskill-root "${ms}" --bootstrap-replicates 10000 --output __OUTPUT__
"${py}" "${exp}/scripts/install_formal_complete.py" --marker "${result}/FORMAL_COMPLETE.json" --official-audit "${result}/INDEPENDENT_AUDIT.json" --posthoc-audit "${result}/POSTHOC_INDEPENDENT_AUDIT.json" --result-vector "${result}/RESULT_VECTOR.json" --oracle-validation "${result}/ORACLE_VALIDATION.json"
