#!/usr/bin/env python3
"""Stage-2.7R adapter over the audited complete-state ACT trainer."""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = REPO / "experiments/maniskill_act_boundary_screen_v1/scripts/train_official_act_protocol.py"
OLD_SCRIPTS = OLD.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OLD_SCRIPTS))

from multires_policy import install_into_official_trainer  # noqa: E402

spec = importlib.util.spec_from_file_location("stage27r_base_trainer", OLD)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.PROTOCOL_ID = "R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1"
base.CANDIDATE_INTERVAL = 5_000
_make_args = base.make_official_args
_make_config = base.make_train_config


def make_args(cli):
    args = _make_args(cli)
    args.num_queries = 8
    args.sim_backend = "physx_cpu"
    return args


def make_config(cli):
    config = _make_config(cli)
    return dataclasses.replace(config, protocol_id=base.PROTOCOL_ID, num_queries=8)


base.make_official_args = make_args
base.make_train_config = make_config
install_into_official_trainer(8)

if __name__ == "__main__":
    base.main()
