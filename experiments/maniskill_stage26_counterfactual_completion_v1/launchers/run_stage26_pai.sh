#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ "$(id -u):$(id -g)" == "2254:2254" ]] || { echo "expected Leon 2254:2254" >&2; exit 73; }
readonly source_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage26-formal-source"
readonly run_id="${PAI_CANARY_RUN_ID:?run id required}"
readonly gpu_count="${PAI_CANARY_EXPECTED_GPUS:?gpu count required}"
readonly artifact_dir="${PAI_CANARY_RUN_DIR:?artifact dir required}"
readonly result_root="/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage26-counterfactual-completion-v1/${run_id}"
readonly runtime_python="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python"
readonly overlay="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r16p18-maniskill-act-v301-overlay/site-packages"
readonly maniskill_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/ManiSkill-r16p18-v3.0.1"

for required in git sha256sum nvidia-smi realpath stat tee sync; do command -v "${required}" >/dev/null; done
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]]
[[ "${gpu_count}" -ge 2 && "${gpu_count}" -le 8 ]]
test "$(realpath -e "${source_root}")" = "${source_root}"
test "$(git -C "${source_root}" rev-parse HEAD)" = "${R16P18_EXPECTED_PROJECT_COMMIT:?}"
test "$(git -C "${source_root}" rev-parse 'HEAD^{tree}')" = "${R16P18_EXPECTED_PROJECT_TREE:?}"
test -z "$(git -C "${source_root}" status --porcelain)"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = "${gpu_count}"
test "$(sha256sum "${source_root}/experiments/maniskill_stage26_counterfactual_completion_v1/launchers/run_stage26_pai.sh" | awk '{print $1}')" = "${R16P18_EXPECTED_LAUNCHER_SHA256:?}"
test "$(sha256sum "${source_root}/experiments/maniskill_stage26_counterfactual_completion_v1/PROTOCOL_FREEZE.json" | awk '{print $1}')" = "${R16P18_EXPECTED_PROTOCOL_FREEZE_SHA256:?}"
(cd "${source_root}/experiments/maniskill_stage26_counterfactual_completion_v1" && sha256sum -c manifests/SCIENTIFIC_SHA256SUMS)
case "${result_root}" in /mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage26-counterfactual-completion-v1/*) ;; *) exit 74;; esac
case "${artifact_dir}" in /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16-p18-maniskill-stage26-counterfactual-completion-v1/pai/*) ;; *) exit 75;; esac
mkdir -p "${result_root}"
for path in "${result_root}" "${artifact_dir}"; do
  probe="${path}/.owner-probe.$$"; : >"${probe}"; [[ "$(stat -c '%u:%g' "${probe}")" == "2254:2254" ]]; rm -f "${probe}"
done

export PYTHONPATH="${source_root}:${maniskill_root}/examples/baselines/act:${maniskill_root}:${overlay}"
export PYTHONUNBUFFERED=1
exec > >(tee -a "${artifact_dir}/runtime.log") 2>&1
child=""
handle_signal(){ printf '{"protocol_id":"R16-P18-MS5-STAGE26-COUNTERFACTUAL-COMPLETION-V1","status":"PREEMPTION_SIGNAL_RECEIVED"}\n' >"${result_root}/PREEMPTION_SIGNAL.json"; sync "${result_root}"; [[ -z "${child}" ]] || kill -TERM "${child}" 2>/dev/null || true; wait "${child}" 2>/dev/null || true; exit 99; }
trap handle_signal TERM INT
"${runtime_python}" "${source_root}/experiments/maniskill_stage26_counterfactual_completion_v1/scripts/run_stage26_formal.py" --run-id "${run_id}" --result-root "${result_root}" --gpu-count "${gpu_count}" &
child="$!"; wait "${child}"
