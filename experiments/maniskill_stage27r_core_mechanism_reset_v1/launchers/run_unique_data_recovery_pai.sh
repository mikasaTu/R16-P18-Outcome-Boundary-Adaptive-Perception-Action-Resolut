#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ "$(id -u):$(id -g)" == "2254:2254" ]] || exit 73
readonly source_root="${R16P18_SOURCE_ROOT:?}"
readonly exp="${source_root}/experiments/maniskill_stage27r_core_mechanism_reset_v1"
readonly py="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python"
readonly overlay="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r16p18-maniskill-act-v301-overlay/site-packages"
readonly ms="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/ManiSkill-r16p18-v3.0.1"
readonly old_data="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-act-boundary-screen-v1/official_demos"
readonly data_root="/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-stage27r-core-reset-v2"
readonly result_root="/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage27r-core-reset-v1/stage27r-data-train-v20-unique-recovery"
readonly artifact="${PAI_CANARY_RUN_DIR:?}"
readonly gpu_count="${PAI_CANARY_EXPECTED_GPUS:?}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${R16P18_EXPECTED_PROJECT_COMMIT:?}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${R16P18_EXPECTED_PROJECT_TREE:?}"
test -z "$(git -C "${source_root}" status --porcelain)"
test "${gpu_count}" = 8
test -f "${result_root}/RECOVERY_SEED_COMPLETE.json"
mkdir -p "${artifact}"; cd "${artifact}"
exec > >(tee -a "${artifact}/runtime.log") 2>&1
export PYTHONPATH="${exp}/scripts:${ms}/examples/baselines/act:${ms}:${overlay}"
export PYTHONUNBUFFERED=1 WANDB_ENTITY="chen_jian-cj-workspace" WANDB_PROJECT="R16-P18-ManiSkill-Stage27R" WANDB__SERVICE_WAIT=300

pids=()
for task in PegInsertionSide-v1 PullCubeTool-v1; do
  "${py}" "${exp}/scripts/prepare_exact_replay_data.py" --task-id "${task}" \
    --official-h5 "${old_data}/${task}/motionplanning/trajectory.h5" \
    --output-root "${data_root}" --python "${py}" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
"${py}" "${exp}/scripts/audit_exact_dataset.py" --dataset-root "${data_root}" --output "${result_root}/EXACT_DATASET_AUDIT.json"

jobs=(); for task in PegInsertionSide-v1 PullCubeTool-v1; do for seed in 16018 16019 16020; do jobs+=("${task} ${seed}"); done; done
pids=()
for index in "${!jobs[@]}"; do
  read -r task seed <<<"${jobs[$index]}"
  output="${result_root}/training/${task}/seed_${seed}"
  worker_tmp="/tmp/r27r-recovery-gpu-${index}"
  mkdir -p "${worker_tmp}" "${output}/wandb"
  CUDA_VISIBLE_DEVICES="${index}" TMPDIR="${worker_tmp}" WANDB_DIR="${output}/wandb" \
    "${py}" "${exp}/scripts/train_multires_act.py" --task-id "${task}" --seed "${seed}" \
    --control-mode pd_ee_delta_pose --train-h5 "${data_root}/${task}/splits/train/trajectory.h5" \
    --validation-h5 "${data_root}/${task}/splits/validation/trajectory.h5" --output-dir "${output}" \
    --total-iterations 30000 --checkpoint-interval 5000 --batch-size 256 --validation-batch-size 256 \
    --num-workers 8 --device cuda --track --wandb-project "${WANDB_PROJECT}" \
    --run-name "stage27r-unique-recovery-${task}-seed${seed}" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
test "$(find "${result_root}/training" -path '*/checkpoints/step_*/COMPLETE.json' -type f | wc -l)" = 108
test "$(find "${result_root}/training" -name TRAINING_COMPLETE.json -type f | wc -l)" = 18
printf '{"protocol_id":"R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1","status":"DATA_AND_TRAINING_COMPLETE","run_id":"%s","recovered_tasks":["PegInsertionSide-v1","PullCubeTool-v1"]}\n' "${PAI_CANARY_RUN_ID:?}" >"${result_root}/DATA_AND_TRAINING_COMPLETE.json"
