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
readonly run_id="${PAI_CANARY_RUN_ID:?}"
readonly gpu_count="${PAI_CANARY_EXPECTED_GPUS:?}"
readonly result="${R16P18_FORMAL_RESULT_ROOT:-/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage27r-formal-v1/${run_id}}"
readonly artifact="${PAI_CANARY_RUN_DIR:?}"
test -f "${training}/DATA_AND_TRAINING_COMPLETE.json"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${R16P18_EXPECTED_PROJECT_COMMIT:?}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${R16P18_EXPECTED_PROJECT_TREE:?}"
test -z "$(git -C "${source_root}" status --porcelain)"
mkdir -p "${result}" "${expert_root}"; cd "${artifact}"
exec > >(tee -a "${artifact}/formal-runtime.log") 2>&1
export PYTHONPATH="${exp}/scripts:${ms}/examples/baselines/act:${ms}:${overlay}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -f "${result}/PRECHECKS.json" ]]; then
  "${py}" -m compileall -q "${exp}/scripts"
  "${py}" -m pytest -q "${exp}/tests" | tee "${artifact}/preflight-pytest.log"
  "${py}" "${exp}/scripts/deterministic_lockstep_smoke.py" --output "${result}/DETERMINISTIC_LOCKSTEP_SMOKE.json"
  "${py}" -c 'import json,sys; smoke=json.load(open(sys.argv[1])); json.dump({"compileall":True,"unit_tests":True,"deterministic_smoke":bool(smoke["pass"]),"deterministic_smoke_evidence":sys.argv[1],"fail_on_overwrite":True,"all_pass":bool(smoke["pass"])},open(sys.argv[2],"x"),indent=2)' "${result}/DETERMINISTIC_LOCKSTEP_SMOKE.json" "${result}/PRECHECKS.json"
fi
test -f "${result}/EXACT_DATASET_AUDIT.json" || "${py}" "${exp}/scripts/audit_exact_dataset.py" --dataset-root "${data_root}" --output "${result}/EXACT_DATASET_AUDIT.json"

screen_dir="${result}/screen"; mkdir -p "${screen_dir}"
tasks=(StackCube-v1 PegInsertionSide-v1 PlugCharger-v1 PullCubeTool-v1 PushT-v1 PushCube-v1); seeds=(16018 16019 16020)
jobs=(); for task in "${tasks[@]}"; do for seed in "${seeds[@]}"; do jobs+=("${task} ${seed}"); done; done
screen_worker(){ local gpu=$1 i task seed out; for ((i=gpu;i<${#jobs[@]};i+=gpu_count)); do read -r task seed <<<"${jobs[$i]}"; out="${screen_dir}/${task}-seed${seed}.json"; test -f "${out}" || CUDA_VISIBLE_DEVICES=${gpu} "${py}" "${exp}/scripts/screen_one.py" --training-run "${training}" --task "${task}" --seed "${seed}" --output "${out}"; done; }
pids=(); for ((g=0;g<gpu_count;g++)); do screen_worker "$g" & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done
test -f "${screen_dir}/TASK_SELECTION.json" || "${py}" "${exp}/scripts/aggregate_screen.py" --shard-dir "${screen_dir}" --output "${screen_dir}/TASK_SELECTION.json"
positive2=$("${py}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_positive"] or "PegInsertionSide-v1")' "${screen_dir}/TASK_SELECTION.json")
negative=$("${py}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_negative"] or "")' "${screen_dir}/TASK_SELECTION.json")
formal_tasks=(StackCube-v1 "${positive2}"); test -z "${negative}" || formal_tasks+=("${negative}")

official_h5(){ if [[ "$1" == PushT-v1 ]]; then echo "${old_data}/$1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"; else echo "${old_data}/$1/motionplanning/trajectory.h5"; fi; }
pids=(); for task in "${formal_tasks[@]}"; do "${py}" "${exp}/scripts/prepare_expert_pool.py" --task "$task" --official-h5 "$(official_h5 "$task")" --output-root "${expert_root}" --python "${py}" & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done
checkpoint(){ "${py}" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["groups"][f"{sys.argv[2]}/seed_{sys.argv[3]}"]["selected"]["path"])' "${screen_dir}/TASK_SELECTION.json" "$1" "$2"; }
expert_h5(){ "${py}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["h5"])' "${expert_root}/$1/EXPERT_POOL_COMPLETE.json"; }

bank_dir="${result}/state_banks"; mkdir -p "${bank_dir}"
for task in "${formal_tasks[@]}"; do
  ck=$(checkpoint "$task" 16018); eh=$(expert_h5 "$task")
  if [[ "$task" == "$negative" && -n "$negative" ]]; then bank=negative; count_offset=0; else bank=calibration; count_offset=0; fi
  if [[ "$bank" == calibration ]]; then
    test -f "${bank_dir}/${task}-calibration.json" || CUDA_VISIBLE_DEVICES=0 "${py}" "${exp}/scripts/build_lockstep_state_bank.py" --task "$task" --model-seed 16018 --checkpoint "$ck" --expert-h5 "$eh" --expert-offset 0 --bank calibration --output "${bank_dir}/${task}-calibration.json"
    test -f "${bank_dir}/${task}-confirmatory.json" || CUDA_VISIBLE_DEVICES=0 "${py}" "${exp}/scripts/build_lockstep_state_bank.py" --task "$task" --model-seed 16018 --checkpoint "$ck" --expert-h5 "$eh" --expert-offset 24 --bank confirmatory --output "${bank_dir}/${task}-confirmatory.json"
  else
    test -f "${bank_dir}/${task}-negative.json" || CUDA_VISIBLE_DEVICES=0 "${py}" "${exp}/scripts/build_lockstep_state_bank.py" --task "$task" --model-seed 16018 --checkpoint "$ck" --expert-h5 "$eh" --expert-offset 0 --bank negative --output "${bank_dir}/${task}-negative.json"
  fi
done

cal="${result}/calibration"; mkdir -p "${cal}"; jobs=()
for task in "${formal_tasks[@]:0:2}"; do for seed in "${seeds[@]}"; do for grid in 2 4; do jobs+=("${task} ${seed} ${grid}"); done; done; done
cal_worker(){ local gpu=$1 i task seed grid out; for ((i=gpu;i<${#jobs[@]};i+=gpu_count)); do read -r task seed grid <<<"${jobs[$i]}"; out="${cal}/${task}-seed${seed}-grid${grid}.json"; test -f "$out" || CUDA_VISIBLE_DEVICES=$gpu "${py}" "${exp}/scripts/run_factorial_oracle.py" --state-bank "${bank_dir}/${task}-calibration.json" --checkpoint "$(checkpoint "$task" "$seed")" --model-seed "$seed" --tile-grid "$grid" --output "$out"; done; }
pids=(); for ((g=0;g<gpu_count;g++)); do cal_worker "$g" & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done
for task in "${formal_tasks[@]:0:2}"; do test -f "${cal}/${task}-CROP_GRID_FREEZE.json" || "${py}" "${exp}/scripts/select_crop_grid.py" --grid2 "${cal}/${task}"-seed*-grid2.json --grid4 "${cal}/${task}"-seed*-grid4.json --output "${cal}/${task}-CROP_GRID_FREEZE.json"; done

oracle="${result}/oracle"; mkdir -p "${oracle}"; jobs=()
for task in "${formal_tasks[@]:0:2}"; do grid=$("${py}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_grid"])' "${cal}/${task}-CROP_GRID_FREEZE.json"); for seed in "${seeds[@]}"; do jobs+=("${task} ${seed} ${grid} confirmatory"); done; done
if [[ -n "$negative" ]]; then for seed in "${seeds[@]}"; do jobs+=("${negative} ${seed} 2 negative"); done; fi
oracle_worker(){ local gpu=$1 i task seed grid bank out; for ((i=gpu;i<${#jobs[@]};i+=gpu_count)); do read -r task seed grid bank <<<"${jobs[$i]}"; out="${oracle}/${task}-seed${seed}-${bank}.json"; test -f "$out" || CUDA_VISIBLE_DEVICES=$gpu "${py}" "${exp}/scripts/run_factorial_oracle.py" --state-bank "${bank_dir}/${task}-${bank}.json" --checkpoint "$(checkpoint "$task" "$seed")" --model-seed "$seed" --tile-grid "$grid" --output "$out"; done; }
pids=(); for ((g=0;g<gpu_count;g++)); do oracle_worker "$g" & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done

"${py}" "${exp}/scripts/analyze_stage27r.py" --inputs "${oracle}"/*.json --output "${result}/statistics.json"
"${py}" "${exp}/scripts/mechanism_audit.py" --inputs "${oracle}"/*.json --output "${result}/MECHANISM_AUDIT.json"
bank_args=("${bank_dir}/StackCube-v1-confirmatory.json" "${bank_dir}/${positive2}-confirmatory.json"); test -z "$negative" || bank_args+=("${bank_dir}/${negative}-negative.json")
"${py}" "${exp}/scripts/decide_stage27r.py" --analysis "${result}/statistics.json" --task-selection "${screen_dir}/TASK_SELECTION.json" --state-banks "${bank_args[@]}" --positive-tasks StackCube-v1 "${positive2}" --output "${result}/RESULT_VECTOR.json"
"${py}" "${exp}/scripts/audit_formal_results.py" --repo "${source_root}" --formal-root "${result}" --training-root "${training}" --dataset-root "${data_root}" --output "${result}/INDEPENDENT_AUDIT.json"
"${py}" "${exp}/scripts/posthoc_independent_audit.py" --formal-root "${result}" --output "${result}/POSTHOC_INDEPENDENT_AUDIT.json"
printf '{"protocol_id":"R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1","status":"FORMAL_COMPLETE"}\n' >"${result}/FORMAL_COMPLETE.json"
