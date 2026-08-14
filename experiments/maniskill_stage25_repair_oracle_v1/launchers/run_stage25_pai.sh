#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" != "2254" || "$(id -g)" != "2254" ]]; then
  echo "runtime identity must remain 2254:2254" >&2
  exit 73
fi

readonly source_root="$1"
readonly result_root="$2"
readonly run_id="$3"
readonly gpu_count="$4"
readonly runtime_python="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python"
readonly overlay="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r16p18-maniskill-act-v301-overlay/site-packages"
readonly maniskill_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/ManiSkill-r16p18-v3.0.1"
readonly old_data="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-act-boundary-screen-v1"

export PYTHONPATH="${source_root}:${maniskill_root}/examples/baselines/act:${maniskill_root}:${overlay}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

mkdir -p "${result_root}"

child_pid=""
handle_signal() {
  printf '{"protocol_id":"R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1","status":"PREEMPTION_SIGNAL_RECEIVED","run_id":"%s"}\n' "${run_id}" > "${result_root}/PREEMPTION_SIGNAL.json"
  sync "${result_root}"
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" || true
  fi
  exit 99
}
trap handle_signal TERM INT

"${runtime_python}" "${source_root}/experiments/maniskill_stage25_repair_oracle_v1/scripts/run_stage25_formal.py" \
  --run-id "${run_id}" \
  --result-root "${result_root}" \
  --candidate-manifest "${source_root}/experiments/maniskill_stage25_repair_oracle_v1/manifests/checkpoint_candidates.json" \
  --screen-seed-bank "${source_root}/experiments/maniskill_stage25_repair_oracle_v1/manifests/checkpoint_screen_seed_bank.json" \
  --final-val-seed-bank "${source_root}/experiments/maniskill_stage25_repair_oracle_v1/manifests/checkpoint_final_val_seed_bank.json" \
  --confirmatory-seed-bank "${source_root}/experiments/maniskill_stage25_repair_oracle_v1/manifests/confirmatory_test_seed_bank.json" \
  --oracle-seed-bank "${source_root}/experiments/maniskill_stage25_repair_oracle_v1/manifests/oracle_source_seed_bank.json" \
  --official-stack-h5 "${old_data}/official_demos/StackCube-v1/motionplanning/trajectory.h5" \
  --training-stack-h5 "${old_data}/selected_raw/StackCube-v1/train/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5" \
  --gpu-count "${gpu_count}" &
child_pid="$!"
wait "${child_pid}"
