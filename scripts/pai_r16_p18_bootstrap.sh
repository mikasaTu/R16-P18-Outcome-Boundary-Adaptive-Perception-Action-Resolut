#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?run id is required}
expected_commit=${2:?expected git commit is required}
expected_config_sha256=${3:?expected config sha256 is required}

case "$run_id" in
  *[!A-Za-z0-9._-]*|'')
    echo "[owner-bootstrap] invalid run id: $run_id" >&2
    exit 50
    ;;
esac

leon_uid=2254
leon_gid=2254
new_root=/mnt/cpfs/zbl-cpfs-new
user_root=$new_root/USERS/leon
data_root=$new_root/dataset/leon
checkpoint_root=$new_root/CKPT/leon
public_root=$new_root/SHARE/leon
project_dir=$user_root/code/r16-p18-libero-stage1-20260812
checkpoint_run_dir=$checkpoint_root/torch/r16-p18-libero-stage1/$run_id
log_run_dir=$user_root/logs/r16-p18-libero-stage1/$run_id
cache_run_dir=$user_root/cache/r16-p18-libero-stage1/$run_id
leon_launcher=$project_dir/scripts/pai_r16_p18_leon.sh
sentinel=$user_root/.pai_mount_identity
expected_sentinel_sha256=b3cc124ffbbebf8a12d52f70555b834f9c62a34ae09ff6ee397d0abbba600ef9
write_paths=("$checkpoint_run_dir" "$log_run_dir" "$cache_run_dir")

current_uid=$(id -u)
current_gid=$(id -g)
if [[ "$current_uid" -ne 0 && "$current_uid:$current_gid" != "$leon_uid:$leon_gid" ]]; then
  echo "[owner-bootstrap] unexpected identity $current_uid:$current_gid" >&2
  exit 51
fi
for executable in realpath findmnt sha256sum git rg; do
  command -v "$executable" >/dev/null || {
    echo "[owner-bootstrap] missing required executable: $executable" >&2
    exit 52
  }
done
if [[ "$current_uid" -eq 0 ]]; then
  command -v setpriv >/dev/null || {
    echo "[owner-bootstrap] setpriv is required" >&2
    exit 53
  }
fi

test -f "$sentinel" && test ! -L "$sentinel" || {
  echo "[owner-bootstrap] mount identity sentinel is missing or symlinked" >&2
  exit 54
}
actual_sentinel_sha256=$(sha256sum "$sentinel" | awk '{print $1}')
[[ "$actual_sentinel_sha256" == "$expected_sentinel_sha256" ]] || {
  echo "[owner-bootstrap] mount identity sentinel hash mismatch" >&2
  exit 55
}

for base_root in "$new_root" "$user_root" "$data_root" "$checkpoint_root" "$public_root"; do
  test -d "$base_root" || {
    echo "[owner-bootstrap] missing mounted base root: $base_root" >&2
    exit 56
  }
  [[ "$(realpath -e -- "$base_root")" == "$base_root" ]] || {
    echo "[owner-bootstrap] non-canonical base root: $base_root" >&2
    exit 57
  }
  mount_target=$(findmnt -n -o TARGET -T "$base_root")
  case "$mount_target" in
    /mnt/cpfs|/mnt/cpfs/*) ;;
    *)
      echo "[owner-bootstrap] unexpected mount target for $base_root: $mount_target" >&2
      exit 58
      ;;
  esac
done

test -x "$leon_launcher" || {
  echo "[owner-bootstrap] Leon launcher is not executable: $leon_launcher" >&2
  exit 59
}
test "$(git -C "$project_dir" rev-parse HEAD)" = "$expected_commit" || {
  echo "[owner-bootstrap] source commit mismatch" >&2
  exit 60
}
test -z "$(git -C "$project_dir" status --porcelain)" || {
  echo "[owner-bootstrap] source worktree is dirty" >&2
  exit 61
}
actual_config_sha256=$(sha256sum "$project_dir/configs/r16_p18_libero_stage1.yaml" | awk '{print $1}')
[[ "$actual_config_sha256" == "$expected_config_sha256" ]] || {
  echo "[owner-bootstrap] locked config hash mismatch" >&2
  exit 62
}

assert_scoped_write_path() {
  local path=$1
  local resolved
  [[ -n "$path" && "$path" == /* ]] || exit 63
  case "$path" in *"/../"*|*/..) exit 64 ;; esac
  resolved=$(realpath -m -- "$path")
  [[ "$resolved" == "$path" ]] || exit 65
  case "$path" in
    "$checkpoint_root"/torch/r16-p18-libero-stage1/"$run_id"|\
    "$user_root"/logs/r16-p18-libero-stage1/"$run_id"|\
    "$user_root"/cache/r16-p18-libero-stage1/"$run_id") ;;
    *)
      echo "[owner-bootstrap] write path is outside the exact run scope: $path" >&2
      exit 66
      ;;
  esac
}

for path in "${write_paths[@]}"; do
  assert_scoped_write_path "$path"
  mkdir -p -- "$path"
  assert_scoped_write_path "$path"
done

if [[ "$current_uid" -eq 0 ]]; then
  chown -R --no-dereference "$leon_uid:$leon_gid" -- "${write_paths[@]}"
fi

runtime_env=(
  /usr/bin/env -i
  "PATH=$user_root/envs/libero_sim/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  "HOME=$user_root"
  "USER=leon"
  "LOGNAME=leon"
  "PYTHONPATH=$project_dir"
  "XDG_CACHE_HOME=$cache_run_dir/xdg"
  "TORCH_HOME=$cache_run_dir/torch"
  "WANDB_DIR=$log_run_dir/wandb"
  "WANDB_CACHE_DIR=$cache_run_dir/wandb"
  "WANDB_API_KEY=${WANDB_API_KEY:?PAI controller must inject WANDB_API_KEY}"
  "WANDB_ENTITY=${WANDB_ENTITY:?PAI controller must inject WANDB_ENTITY}"
  "R16_EXPECTED_GIT_COMMIT=$expected_commit"
  "R16_EXPECTED_CONFIG_SHA256=$expected_config_sha256"
  "R16_CHECKPOINT_RUN_DIR=$checkpoint_run_dir"
  "R16_LOG_RUN_DIR=$log_run_dir"
  "R16_CACHE_RUN_DIR=$cache_run_dir"
  "OMP_NUM_THREADS=8"
)
[[ "$WANDB_ENTITY" == "chen_jian-cj-workspace" ]] || {
  echo "[owner-bootstrap] W&B entity contract mismatch" >&2
  exit 67
}

if [[ "$current_uid" -eq 0 ]]; then
  exec setpriv --reuid="$leon_uid" --regid="$leon_gid" --clear-groups -- \
    "${runtime_env[@]}" "$leon_launcher" "$run_id"
fi
exec "${runtime_env[@]}" "$leon_launcher" "$run_id"
