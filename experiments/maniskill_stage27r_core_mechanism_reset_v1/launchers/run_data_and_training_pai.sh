#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ "$(id -u):$(id -g)" == "2254:2254" ]] || { echo "expected runtime 2254:2254" >&2; exit 73; }
readonly source_root="${R16P18_SOURCE_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage26-formal-source-v4}"
readonly exp_root="${source_root}/experiments/maniskill_stage27r_core_mechanism_reset_v1"
readonly runtime_python="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python"
readonly overlay="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r16p18-maniskill-act-v301-overlay/site-packages"
readonly maniskill_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/ManiSkill-r16p18-v3.0.1"
readonly old_data="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-act-boundary-screen-v1/official_demos"
readonly data_root="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-stage27r-core-reset-v1"
readonly run_id="${PAI_CANARY_RUN_ID:?run id required}"
readonly gpu_count="${PAI_CANARY_EXPECTED_GPUS:?gpu count required}"
readonly result_root="/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage27r-core-reset-v1/${run_id}"
readonly artifact_dir="${PAI_CANARY_RUN_DIR:?artifact dir required}"
readonly expected_commit="${R16P18_EXPECTED_PROJECT_COMMIT:?expected commit required}"
readonly expected_tree="${R16P18_EXPECTED_PROJECT_TREE:?expected tree required}"

[[ "${gpu_count}" =~ ^[2-8]$ ]]
test "$(git -C "${source_root}" rev-parse HEAD)" = "${expected_commit}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${expected_tree}"
test -z "$(git -C "${source_root}" status --porcelain)"
test "$(git -C "${maniskill_root}" rev-parse HEAD)" = "a4a4f9272ad64b1564035874b605ceb687b63ed8"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = "${gpu_count}"
mkdir -p "${result_root}" "${data_root}" "${artifact_dir}"
test "$(stat -c '%u:%g' "${result_root}")" = "2254:2254"
exec > >(tee -a "${artifact_dir}/runtime.log") 2>&1
cd "${artifact_dir}"
test "$(pwd -P)" = "${artifact_dir}"

export PYTHONPATH="${exp_root}/scripts:${maniskill_root}/examples/baselines/act:${maniskill_root}:${overlay}"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY="chen_jian-cj-workspace"
export WANDB_PROJECT="R16-P18-ManiSkill-Stage27R"
export WANDB__SERVICE_WAIT=300
child_pids=()
handle_signal() {
  printf '{"protocol_id":"R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1","status":"PREEMPTION_SIGNAL_RECEIVED"}\n' >"${result_root}/PREEMPTION_SIGNAL.json"
  for pid in "${child_pids[@]:-}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  wait || true; sync "${result_root}"; exit 99
}
trap handle_signal TERM INT

tasks=(StackCube-v1 PegInsertionSide-v1 PlugCharger-v1 PullCubeTool-v1 PushT-v1 PushCube-v1)
for task in "${tasks[@]}"; do
  source_h5="${old_data}/${task}/motionplanning/trajectory.h5"
  if [[ "${task}" == "PushT-v1" ]]; then
    source_h5="${old_data}/${task}/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
  fi
  "${runtime_python}" "${exp_root}/scripts/prepare_exact_replay_data.py" \
    --task-id "${task}" \
    --official-h5 "${source_h5}" \
    --output-root "${data_root}" --python "${runtime_python}" &
  child_pids+=("$!")
done
for pid in "${child_pids[@]}"; do wait "${pid}"; done
child_pids=()

control_for() {
  case "$1" in StackCube-v1|PushCube-v1) echo pd_ee_delta_pos;; *) echo pd_ee_delta_pose;; esac
}

mapfile -t jobs < <(for task in "${tasks[@]}"; do for seed in 16018 16019 16020; do echo "${task} ${seed}"; done; done)
worker() {
  local gpu="$1" index task seed control output worker_tmp wandb_dir
  for ((index=gpu; index<${#jobs[@]}; index+=gpu_count)); do
    read -r task seed <<<"${jobs[$index]}"
    control="$(control_for "${task}")"
    output="${result_root}/training/${task}/seed_${seed}"
    worker_tmp="${artifact_dir}/tmp/gpu_${gpu}/${task}/seed_${seed}"
    wandb_dir="${output}/wandb"
    mkdir -p "${worker_tmp}" "${wandb_dir}"
    # W&B starts a child service after Python's tempfile context is created.
    # Give every concurrent run a persistent, non-shared directory so one
    # process cannot invalidate another process's service port file.
    CUDA_VISIBLE_DEVICES="${gpu}" TMPDIR="${worker_tmp}" WANDB_DIR="${wandb_dir}" \
      "${runtime_python}" "${exp_root}/scripts/train_multires_act.py" \
      --task-id "${task}" --seed "${seed}" --control-mode "${control}" \
      --train-h5 "${data_root}/${task}/splits/train/trajectory.h5" \
      --validation-h5 "${data_root}/${task}/splits/validation/trajectory.h5" \
      --output-dir "${output}" --total-iterations 30000 --checkpoint-interval 5000 \
      --batch-size 256 --validation-batch-size 256 --device cuda --track \
      --wandb-project "${WANDB_PROJECT}" --run-name "stage27r-${task}-seed${seed}"
  done
}
for ((gpu=0; gpu<gpu_count; gpu++)); do worker "${gpu}" & child_pids+=("$!"); done
for pid in "${child_pids[@]}"; do wait "${pid}"; done
printf '{"protocol_id":"R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1","status":"DATA_AND_TRAINING_COMPLETE","run_id":"%s"}\n' "${run_id}" >"${result_root}/DATA_AND_TRAINING_COMPLETE.json"
sync "${result_root}"
