#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?run id is required}
leon_uid=2254
leon_gid=2254
new_root=/mnt/cpfs/zbl-cpfs-new
user_root=$new_root/USERS/leon
project_dir=$user_root/code/r16-p18-libero-stage1-20260812
python_bin=$user_root/envs/libero_sft/bin/python
python_overlay=$user_root/envs/r22p10-libero-pai-overlay/site-packages
dataset_root=$new_root/dataset/leon/embodied_benchmark/datasets/LIBERO
checkpoint_base=$new_root/CKPT/leon/torch/r16-p18-libero-stage1
log_base=$user_root/logs/r16-p18-libero-stage1
config=$project_dir/configs/r16_p18_libero_stage1.yaml

[[ "$(id -u):$(id -g)" == "$leon_uid:$leon_gid" ]] || {
  echo "[owner-preflight] expected $leon_uid:$leon_gid, got $(id -u):$(id -g)" >&2
  exit 70
}
[[ "$(git -C "$project_dir" rev-parse HEAD)" == "${R16_EXPECTED_GIT_COMMIT:?}" ]] || {
  echo "[preflight] source commit changed after privilege drop" >&2
  exit 71
}
test -z "$(git -C "$project_dir" status --porcelain)" || {
  echo "[preflight] source worktree is dirty after privilege drop" >&2
  exit 72
}
[[ "$(sha256sum "$config" | awk '{print $1}')" == "${R16_EXPECTED_CONFIG_SHA256:?}" ]] || {
  echo "[preflight] config hash changed after privilege drop" >&2
  exit 73
}
test -x "$python_bin" && test -d "$dataset_root" || {
  echo "[preflight] pinned Python or dataset root is missing" >&2
  exit 74
}
test "$(sha256sum "$(realpath -e "$python_bin")" | awk '{print $1}')" = \
  89b2f5166fb529c259aedd43e5f718c60e35d58e630cb40ae6accb48fc4f961a || {
  echo "[preflight] pinned Python hash mismatch" >&2
  exit 80
}
overlay_manifest_sha=$(find "$python_overlay" -type f -printf '%P\0' | sort -z | while IFS= read -r -d '' relative; do printf '%s\0' "$relative"; cat "$python_overlay/$relative"; done | sha256sum | awk '{print $1}')
test "$overlay_manifest_sha" = 64dfffdaf464d1a37be19b038cca919a252dba573eb2d0f8aa442b91a4099459 || {
  echo "[preflight] pinned Python overlay hash mismatch" >&2
  exit 81
}
test "$(find "$python_overlay" -type f | wc -l)" = 1688 || {
  echo "[preflight] pinned Python overlay file count mismatch" >&2
  exit 82
}
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2 || {
  echo "[preflight] expected the verified 2xA800 carrier" >&2
  exit 77
}
test "${WANDB_ENTITY:-}" = chen_jian-cj-workspace && test -n "${WANDB_API_KEY:-}" || {
  echo "[preflight] exact W&B secret contract was not injected" >&2
  exit 78
}
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 2 || {
  echo "[preflight] exact 2xA800 runtime inventory is absent" >&2
  exit 79
}

legacy_re='/mnt/data/'"x2robot_v2|/mnt/data/"'checkpoint|/mnt/'"checkpoint|/mnt/cpfs/"'leon|(^|[^[:alnum:]_])/('"x2robot_v2|x2robot_data|open_data"')(/|$)'
if rg -n "$legacy_re" "$config" "$project_dir/scripts" "$project_dir/boundarybc"; then
  echo "[preflight] legacy storage path detected" >&2
  exit 75
fi

mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"
export PYTHONPATH="$python_overlay:$project_dir"
"$python_bin" -m boundarybc.preflight \
  --config "$config" \
  --run-id "$run_id" \
  --checkpoint-run-dir "${R16_CHECKPOINT_RUN_DIR:?}" \
  --log-run-dir "${R16_LOG_RUN_DIR:?}" \
  --cache-run-dir "${R16_CACHE_RUN_DIR:?}"

first_artifact=$R16_LOG_RUN_DIR/launcher_preflight.json
[[ "$(stat -c '%u:%g' "$first_artifact")" == "$leon_uid:$leon_gid" ]] || {
  echo "[owner-check] first persistent artifact is not Leon-owned" >&2
  exit 76
}

echo "FINAL_COMMAND=$python_bin -m boundarybc.pipeline --config $config --run-id $run_id"
exec "$python_bin" -m boundarybc.pipeline \
  --config "$config" \
  --run-id "$run_id" \
  --dataset-root "$dataset_root" \
  --checkpoint-root "$checkpoint_base" \
  --log-root "$log_base" \
  --devices cuda:0,cuda:1
