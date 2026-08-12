#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

PROJECT_DIR=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16-p18-libero-stage1-20260812
PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python
DATASET_ROOT=/mnt/cpfs/zbl-cpfs-new/dataset/leon/embodied_benchmark/datasets/LIBERO
CHECKPOINT_BASE=/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-libero-stage1
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
NONCE=${PAI_CANARY_NONCE:?PAI_CANARY_NONCE is required}

EXPECTED_SOURCE_COMMIT=20d65a9ab7ff42662496c093e14531904cf1fe31
EXPECTED_SOURCE_TREE=b87468210362a81a5d3754630f1ff804f79577e4
EXPECTED_CONFIG_SHA256=d4344c056e1c7682cec9deabd0c82888bd6ec29d6d473b9a1870a262aeaa64bf
EXPECTED_PROJECT_LAUNCHER_SHA256=506f6a98490c3eca7dad8096a550c46e694629192a1deb7a03c8c102e116574f
EXPECTED_SMOKE_EVIDENCE_SHA256=f9968365ec092f9e83e14c2789ad45809fddae7e025e45c5dd3f7bb3f3c592da
EXPECTED_SMOKE_RESULT_SHA256=8a5cb0c1c3f5b68f45f34c7e75227086d61e34bd709a54003875b2c2a6658fb2
SMOKE_RESULT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16-p18-libero-stage1/dev14-smoke-20260812-v5/smoke_result.json

on_error() {
  local exit_code=$?
  printf 'R16P18_BASELINE_COMMAND_FAILED line=%s exit_code=%s command=%q\n' \
    "${BASH_LINENO[0]:-unknown}" "$exit_code" "$BASH_COMMAND" >&2
  return "$exit_code"
}
trap on_error ERR

for required in git sha256sum nvidia-smi stat realpath awk grep; do
  command -v "$required" >/dev/null
done
test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2
test "${WANDB_ENTITY:-}" = chen_jian-cj-workspace
test -n "${WANDB_API_KEY:-}"
[[ "$RUN_ID" =~ ^[a-z0-9][a-z0-9.-]{2,63}$ ]]
[[ "$NONCE" =~ ^[a-f0-9]{32}$ ]]
case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16-p18-libero-stage1/*) ;;
  *) printf 'artifact directory escaped the R16-P18 output root\n' >&2; exit 71 ;;
esac
test "$(realpath -e "$ARTIFACT_DIR")" = "$ARTIFACT_DIR"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR")" = 2254:2254

CHECKPOINT_RUN_DIR=$CHECKPOINT_BASE/$RUN_ID
test "$(realpath -e "$CHECKPOINT_RUN_DIR")" = "$CHECKPOINT_RUN_DIR"
test "$(stat -c '%u:%g' "$CHECKPOINT_RUN_DIR")" = 2254:2254
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$PROJECT_DIR" rev-parse 'HEAD^{tree}')" = "$EXPECTED_SOURCE_TREE"
test -z "$(git -C "$PROJECT_DIR" status --porcelain)"
test "$(sha256sum "$PROJECT_DIR/configs/r16_p18_libero_stage1.yaml" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256"
test "$(sha256sum "$PROJECT_DIR/scripts/pai_r16_p18_leon.sh" | awk '{print $1}')" = "$EXPECTED_PROJECT_LAUNCHER_SHA256"
test "$(sha256sum "$PROJECT_DIR/configs/dev14_smoke_evidence.yaml" | awk '{print $1}')" = "$EXPECTED_SMOKE_EVIDENCE_SHA256"
test "$(sha256sum "$SMOKE_RESULT" | awk '{print $1}')" = "$EXPECTED_SMOKE_RESULT_SHA256"
test "$(stat -c '%u:%g' "$SMOKE_RESULT")" = 2254:2254
test -x "$PYTHON"
"$PYTHON" - "$SMOKE_RESULT" "$EXPECTED_CONFIG_SHA256" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["event"] == "DEV14_ONE_GPU_SMOKE_COMPLETE"
assert value["config_sha256"] == sys.argv[2]
assert value["adaptive_components_implemented"] is False
PY
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 2
test -d "$DATASET_ROOT"

export R16_EXPECTED_GIT_COMMIT=$EXPECTED_SOURCE_COMMIT
export R16_EXPECTED_CONFIG_SHA256=$EXPECTED_CONFIG_SHA256
export R16_CHECKPOINT_RUN_DIR=$CHECKPOINT_RUN_DIR
export R16_LOG_RUN_DIR=$ARTIFACT_DIR
export R16_CACHE_RUN_DIR=$ARTIFACT_DIR/cache
export XDG_CACHE_HOME=$ARTIFACT_DIR/cache/xdg
export TORCH_HOME=$ARTIFACT_DIR/cache/torch
export WANDB_DIR=$ARTIFACT_DIR/wandb
export WANDB_CACHE_DIR=$ARTIFACT_DIR/cache/wandb

exec "$PROJECT_DIR/scripts/pai_r16_p18_leon.sh" "$RUN_ID"
