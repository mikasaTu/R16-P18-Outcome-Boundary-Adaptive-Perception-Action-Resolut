#!/usr/bin/env bash
set -euo pipefail

umask 077

if [[ "$(id -u)" != "2254" || "$(id -g)" != "2254" ]]; then
  echo "runtime identity must remain 2254:2254" >&2
  exit 73
fi

readonly source_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage25-formal-source"
readonly run_id="${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}"
readonly result_root="/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage25-repair-oracle-v1/${run_id}"
readonly gpu_count="${PAI_CANARY_EXPECTED_GPUS:?PAI_CANARY_EXPECTED_GPUS is required}"
readonly artifact_dir="${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}"
readonly runtime_python="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python"
readonly overlay="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r16p18-maniskill-act-v301-overlay/site-packages"
readonly maniskill_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/ManiSkill-r16p18-v3.0.1"
readonly old_data="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-act-boundary-screen-v1"
readonly launcher_path="${source_root}/experiments/maniskill_stage25_repair_oracle_v1/launchers/run_stage25_pai.sh"

for required in git sha256sum nvidia-smi realpath stat tee sync; do
  command -v "${required}" >/dev/null
done
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]]
test "${gpu_count}" = "8"
test "$(realpath -e "${source_root}")" = "${source_root}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${R16P18_EXPECTED_PROJECT_COMMIT:?}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${R16P18_EXPECTED_PROJECT_TREE:?}"
test -z "$(git -C "${source_root}" status --porcelain)"
test "$(sha256sum "${launcher_path}" | awk '{print $1}')" = "${R16P18_EXPECTED_LAUNCHER_SHA256:?}"
test "$(sha256sum "${source_root}/experiments/maniskill_stage25_repair_oracle_v1/PROTOCOL_FREEZE.json" | awk '{print $1}')" = "${R16P18_EXPECTED_PROTOCOL_FREEZE_SHA256:?}"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = "${gpu_count}"
case "${result_root}" in
  /mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage25-repair-oracle-v1/*) ;;
  *) echo "result root escaped the frozen experiment root" >&2; exit 71 ;;
esac
case "${artifact_dir}" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16-p18-maniskill-stage25-repair-oracle-v1/pai/*) ;;
  *) echo "artifact root escaped the frozen PAI log root" >&2; exit 72 ;;
esac

export PYTHONPATH="${source_root}:${maniskill_root}/examples/baselines/act:${maniskill_root}:${overlay}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

mkdir -p "${result_root}"
test "$(stat -c '%u:%g' "${result_root}")" = "2254:2254"

exec > >(tee -a "${artifact_dir}/runtime.log") 2>&1

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
