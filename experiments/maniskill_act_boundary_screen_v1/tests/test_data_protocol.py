from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import (  # noqa: E402
    closed_loop_seeds,
    replay_count_bounds,
    selection_key,
    sha256_hdf5_group,
    sha256_initial_state,
    validate_replayed_split_count,
)


def test_initial_hash_uses_only_first_state_and_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as handle:
        states = handle.create_group("states")
        states.create_dataset("b", data=np.array([[1, 2], [9, 9]], dtype=np.float32))
        nested = states.create_group("a")
        nested.create_dataset("x", data=np.array([[3], [8]], dtype=np.int32))
    with h5py.File(path, "r+") as handle:
        first = sha256_initial_state(handle["states"])
        handle["states/b"][1] = np.array([4, 4], dtype=np.float32)
        assert sha256_initial_state(handle["states"]) == first
        handle["states/b"][0] = np.array([4, 4], dtype=np.float32)
        assert sha256_initial_state(handle["states"]) != first


def test_full_trajectory_hash_changes_with_any_timestep(tmp_path: Path) -> None:
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        trajectory.create_dataset("actions", data=np.zeros((2, 3), dtype=np.float32))
    with h5py.File(path, "r+") as handle:
        first = sha256_hdf5_group(handle["traj_0"])
        handle["traj_0/actions"][1, 2] = 1
        assert sha256_hdf5_group(handle["traj_0"]) != first


def test_selection_key_is_deterministic_and_identity_sensitive() -> None:
    first = selection_key("PushT-v1", 7, 11, "a" * 64)
    assert first == selection_key("PushT-v1", 7, 11, "a" * 64)
    assert first != selection_key("PushT-v1", 8, 11, "a" * 64)


def test_closed_loop_seeds_are_unique_and_disjoint() -> None:
    forbidden = set(range(500))
    seeds = closed_loop_seeds("StackCube-v1", forbidden)
    assert len(seeds) == 100
    assert len(set(seeds)) == 100
    assert not forbidden.intersection(seeds)


def test_replayed_split_count_uses_frozen_95_percent_gate() -> None:
    assert replay_count_bounds("train") == (190, 200)
    assert replay_count_bounds("validation") == (48, 50)
    for split, accepted in (("train", (190, 200)), ("validation", (48, 50))):
        for observed in accepted:
            validate_replayed_split_count(split, observed)


def test_replayed_split_count_rejects_below_gate_and_overflow() -> None:
    with pytest.raises(RuntimeError, match="outside frozen gate"):
        validate_replayed_split_count("train", 189)
    with pytest.raises(RuntimeError, match="outside frozen gate"):
        validate_replayed_split_count("validation", 51)
