from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from ema_compat import NonDeepSpeedEMAModel  # noqa: E402


def test_non_deepspeed_ema_matches_diffusers_update_formula() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    ema = NonDeepSpeedEMAModel(parameters=[parameter], power=0.75)

    for value in (torch.tensor([3.0, 4.0]), torch.tensor([5.0, -1.0])):
        previous = ema.shadow_params[0].clone()
        with torch.no_grad():
            parameter.copy_(value)
        next_step = ema.optimization_step + 1
        expected_decay = ema.get_decay(next_step)
        expected = previous - (1 - expected_decay) * (previous - parameter)
        ema.step([parameter])
        torch.testing.assert_close(ema.shadow_params[0], expected)
        assert ema.cur_decay_value == expected_decay


def test_non_deepspeed_ema_state_roundtrip_and_copy() -> None:
    first_parameter = torch.nn.Parameter(torch.tensor([2.0]))
    first = NonDeepSpeedEMAModel(parameters=[first_parameter], power=0.75)
    with torch.no_grad():
        first_parameter.fill_(7.0)
    first.step([first_parameter])

    serialized = io.BytesIO()
    torch.save(first.state_dict(), serialized)
    serialized.seek(0)

    resumed_parameter = torch.nn.Parameter(torch.tensor([-5.0]))
    resumed = NonDeepSpeedEMAModel(parameters=[resumed_parameter], power=0.75)
    resumed.load_state_dict(torch.load(serialized, weights_only=True))
    assert resumed.optimization_step == first.optimization_step
    torch.testing.assert_close(resumed.shadow_params[0], first.shadow_params[0])

    copied = torch.nn.Parameter(torch.tensor([0.0]))
    resumed.copy_to([copied])
    torch.testing.assert_close(copied, first.shadow_params[0])


def test_non_deepspeed_ema_rejects_parameter_inventory_drift() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    ema = NonDeepSpeedEMAModel(parameters=[parameter], power=0.75)
    try:
        ema.step([])
    except RuntimeError as exc:
        assert "parameter inventory changed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing EMA parameter inventory failure")
