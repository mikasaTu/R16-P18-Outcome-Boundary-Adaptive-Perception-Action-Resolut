#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

PROJECT_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut
EXPERIMENT_ROOT="$PROJECT_ROOT/experiments/maniskill_act_boundary_screen_v1"
UPSTREAM_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/ManiSkill-r16p18-v3.0.1
ACT_ROOT="$UPSTREAM_ROOT/examples/baselines/act"
PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python
OVERLAY=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r16p18-maniskill-act-v301-overlay/site-packages
DATA_ROOT=/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-act-boundary-screen-v1
SELECTED_RAW_ROOT="$DATA_ROOT/selected_raw"
CHECKPOINT_ROOT=/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-act-boundary-screen-v1
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
NONCE=${PAI_CANARY_NONCE:?PAI_CANARY_NONCE is required}
CACHE_DIR="/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16-p18-maniskill-act-boundary-screen-v1/pai/$RUN_ID"
EXPECTED_PROJECT_COMMIT=${R16P18_EXPECTED_PROJECT_COMMIT:?R16P18_EXPECTED_PROJECT_COMMIT is required}

EXPECTED_UPSTREAM_COMMIT=a4a4f9272ad64b1564035874b605ceb687b63ed8
EXPECTED_UPSTREAM_TREE=dc931fb9ea2f7c039623d1d419767fae56f217ae
EXPECTED_PYTHON_SHA256=89b2f5166fb529c259aedd43e5f718c60e35d58e630cb40ae6accb48fc4f961a
EXPECTED_PACKAGE_LOCK_SHA256=29965e175cc262e8a4250d1f137d3c85c5c59e8612f530b99703bceef30cef59
EXPECTED_DEMO_ZIP_LOCK_SHA256=003bce4d00456116ce5eeb5220354bae6cb3d85d9aed04dab2af8488058260b3
EXPECTED_RESNET18_SHA256=f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec

on_error() {
  local status=$?
  printf 'R16P18_FORMAL_FAILED line=%s exit=%s command=%q\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" "$BASH_COMMAND" >&2
  return "$status"
}
trap on_error ERR

for required in git sha256sum nvidia-smi stat realpath awk grep find sort \
    tee sync; do
  command -v "$required" >/dev/null
done
test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]]
[[ "$NONCE" =~ ^[a-f0-9]{32}$ ]]
[[ "$EXPECTED_PROJECT_COMMIT" =~ ^[a-f0-9]{40}$ ]]

case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16-p18-maniskill-act-boundary-screen-v1/pai-train/*) ;;
  *) printf 'artifact directory escaped fixed log root\n' >&2; exit 71 ;;
esac
case "$CHECKPOINT_ROOT" in
  /mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-act-boundary-screen-v1) ;;
  *) exit 72 ;;
esac
case "$SELECTED_RAW_ROOT" in
  /mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-act-boundary-screen-v1/selected_raw) ;;
  *) exit 73 ;;
esac
case "$CACHE_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16-p18-maniskill-act-boundary-screen-v1/pai/*) ;;
  *) exit 74 ;;
esac
for path in "$ARTIFACT_DIR" "$CHECKPOINT_ROOT" "$SELECTED_RAW_ROOT" "$CACHE_DIR"; do
  test -d "$path"
  test -w "$path"
  test "$(realpath -e "$path")" = "$path"
  probe="$path/.r16p18-owner-probe.$$"
  : >"$probe"
  test "$(stat -c '%u:%g' "$probe")" = 2254:2254
  rm -f "$probe"
done

test "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" = "$EXPECTED_PROJECT_COMMIT"
test -z "$(git -C "$PROJECT_ROOT" status --porcelain)"
test "$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)" = "$EXPECTED_UPSTREAM_COMMIT"
test "$(git -C "$UPSTREAM_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_UPSTREAM_TREE"
test -z "$(git -C "$UPSTREAM_ROOT" status --porcelain)"
test "$(sha256sum "$(realpath -e "$PYTHON")" | awk '{print $1}')" = "$EXPECTED_PYTHON_SHA256"
test "$(sha256sum "$EXPERIMENT_ROOT/locks/python-overlay-packages.txt" | awk '{print $1}')" = "$EXPECTED_PACKAGE_LOCK_SHA256"
test "$(sha256sum "$EXPERIMENT_ROOT/locks/official_demo_zip_sha256.json" | awk '{print $1}')" = "$EXPECTED_DEMO_ZIP_LOCK_SHA256"
test "$(sha256sum /mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16-p18-maniskill-act-boundary-screen-v1/torch/hub/checkpoints/resnet18-f37072fd.pth | awk '{print $1}')" = "$EXPECTED_RESNET18_SHA256"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 2

export PYTHONPATH="$OVERLAY:$UPSTREAM_ROOT:$ACT_ROOT"
export TORCH_HOME=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16-p18-maniskill-act-boundary-screen-v1/torch
export MS_ASSET_DIR="$CACHE_DIR/maniskill"
export HF_HOME="$CACHE_DIR/huggingface"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export PYTHONPYCACHEPREFIX="$CACHE_DIR/pycache"
export TMPDIR="$CACHE_DIR/tmp"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
mkdir -p "$MS_ASSET_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"

"$PYTHON" - <<'PY'
import os, torch
assert os.getuid() == 2254 and os.getgid() == 2254
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()}, flush=True)
PY

"$PYTHON" "$EXPERIMENT_ROOT/scripts/run_formal_matrix.py" \
  --phase replay-and-train \
  --python "$PYTHON" \
  --selected-raw-root "$SELECTED_RAW_ROOT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --artifact-dir "$ARTIFACT_DIR" \
  --gpu-count 2

test -s "$ARTIFACT_DIR/FIRST_REAL_WORK.json"
test -s "$ARTIFACT_DIR/TRAINING_MATRIX_COMPLETE.json"
test -s "$ARTIFACT_DIR/FORMAL_MATRIX_RESULT.json"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR/FIRST_REAL_WORK.json")" = 2254:2254
sync -f "$ARTIFACT_DIR/FORMAL_MATRIX_RESULT.json"
printf 'R16P18_FORMAL_REPLAY_TRAIN_COMPLETE run_id=%s\n' "$RUN_ID"
