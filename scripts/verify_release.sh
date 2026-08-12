#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-python}

"$python_bin" scripts/verify_archived_results.py
"$python_bin" -m pytest -q tests/test_stage1_contract.py
sha256sum -c provenance/SHA256SUMS
