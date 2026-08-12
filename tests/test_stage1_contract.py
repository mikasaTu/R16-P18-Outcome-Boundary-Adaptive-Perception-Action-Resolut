from __future__ import annotations

import random

import numpy as np
import torch

from boundarybc.checkpoint import (
    capture_rng_state,
    checkpoint_path,
    discover_complete_checkpoints,
    latest_complete_checkpoint,
    load_complete_checkpoint,
    marker_path,
    restore_rng_state,
    save_complete_checkpoint,
)
from boundarybc.config import load_config
from boundarybc.model import BoundaryBCS
from boundarybc.pipeline import _assign_units


def test_locked_config_and_visual_action_shapes() -> None:
    config = load_config("configs/r16_p18_libero_stage1.yaml")
    assert [task.key for task in config.tasks] == ["push_plate", "bottle_rack", "bowl_plate"]
    model = BoundaryBCS()
    model.eval()
    with torch.inference_mode():
        image = torch.zeros(1, 3, 128, 128)
        microtokens = model.encode_microtokens(image)
        prediction = model(image, torch.zeros(1, 15))
    assert microtokens.shape == (1, 64, 128)
    assert model.uniform_tokens(microtokens).shape == (1, 16, 128)
    assert prediction.shape == (1, 8, 7)


def test_complete_checkpoint_discovery_retention_and_partial_ignore(tmp_path) -> None:
    partial = checkpoint_path(tmp_path, 9)
    torch.save({"partial": True}, partial)
    assert latest_complete_checkpoint(tmp_path) is None

    for step in (10, 20, 30, 40):
        save_complete_checkpoint(
            tmp_path,
            step=step,
            payload=_payload(step),
            keep_last=3,
        )
    complete = discover_complete_checkpoints(tmp_path)
    assert [step for step, _ in complete] == [20, 30, 40]
    assert partial.is_file()
    assert not marker_path(partial).exists()
    assert latest_complete_checkpoint(tmp_path) == checkpoint_path(tmp_path, 40)
    loaded = load_complete_checkpoint(checkpoint_path(tmp_path, 40), map_location="cpu")
    assert loaded["global_step"] == 40


def test_checkpoint_rejects_missing_state_and_step_mismatch(tmp_path) -> None:
    incomplete = _payload(10)
    del incomplete["optimizer"]
    try:
        save_complete_checkpoint(tmp_path, step=10, payload=incomplete)
    except ValueError as error:
        assert "optimizer" in str(error)
    else:
        raise AssertionError("incomplete payload was accepted")
    try:
        save_complete_checkpoint(tmp_path, step=11, payload=_payload(10))
    except ValueError as error:
        assert "step mismatch" in str(error)
    else:
        raise AssertionError("mismatched global step was accepted")


def test_rng_state_round_trip() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected


def test_two_gpu_assignments_never_share_a_device_queue() -> None:
    units = [(f"task-{index}", index) for index in range(9)]
    assignments = _assign_units(units, ("cuda:0", "cuda:1"))
    assert assignments == (
        ("cuda:0", tuple(units[0::2])),
        ("cuda:1", tuple(units[1::2])),
    )
    flattened = [unit for _, assigned in assignments for unit in assigned]
    assert sorted(flattened) == sorted(units)


def _payload(step: int) -> dict[str, object]:
    return {
        "run_id": "unit-test",
        "task_key": "push_plate",
        "model_seed": 16018,
        "model": {"weight": torch.tensor([float(step)])},
        "optimizer": {"state": {}},
        "scheduler": {"last_epoch": step},
        "rng": capture_rng_state(),
        "batch_generator_state": torch.Generator().get_state(),
        "global_step": step,
    }
