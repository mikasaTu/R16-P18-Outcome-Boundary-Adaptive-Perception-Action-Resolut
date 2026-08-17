from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import atomic_json  # noqa: E402
from analyze_stage27r import holm, paired_summary  # noqa: E402
import multires_policy  # noqa: E402
from multires_policy import MultiResolutionAgent, Native128Dataset, crop_tile  # noqa: E402
from prepare_exact_replay_data import replay_state_flags  # noqa: E402


def test_crop_tiles_partition_exactly() -> None:
    image = torch.arange(128 * 128).reshape(1, 1, 128, 128)
    tiles = [crop_tile(image, tile, 2) for tile in range(4)]
    restored = torch.cat([torch.cat(tiles[:2], -1), torch.cat(tiles[2:], -1)], -2)
    assert torch.equal(restored, image)


def test_crop_validation() -> None:
    with pytest.raises(ValueError):
        crop_tile(torch.zeros(1, 1, 128, 128), 4, 2)
    with pytest.raises(ValueError):
        crop_tile(torch.zeros(1, 1, 128, 128), 0, 3)


def test_dataset_marks_only_first_temporal_quintile_as_free_space(monkeypatch) -> None:
    monkeypatch.setattr(
        multires_policy.official_act.SmallDemoDataset_ACTPolicy,
        "__getitem__",
        lambda self, index: {"observations": {"state": torch.zeros(1)}, "actions": torch.zeros(1)},
    )
    dataset = Native128Dataset.__new__(Native128Dataset)
    dataset.slices = [(0, 1), (0, 2)]
    dataset.trajectories = {"actions": [torch.zeros(10, 7)]}
    assert bool(dataset[0]["observations"]["_free_space_mask"])
    assert not bool(dataset[1]["observations"]["_free_space_mask"])


def test_consistency_uses_same_posterior_sample_and_free_space_mask(monkeypatch) -> None:
    class FakeModel(torch.nn.Module):
        latent_dim = 2

        def __init__(self):
            super().__init__()
            self.noises = []

        def forward(self, obs, actions=None):
            self.noises.append(obs["_latent_noise"].detach().clone())
            batch, horizon, action_dim = actions.shape
            base = obs["_latent_noise"][:, :1, None].expand(batch, horizon, action_dim)
            if obs["_visual_mode"] == "fine":
                base = base + torch.tensor([1.0, 50.0, 50.0])[:, None, None]
            zeros = torch.zeros(batch, self.latent_dim)
            return base, [zeros, zeros]

    agent = MultiResolutionAgent.__new__(MultiResolutionAgent)
    torch.nn.Module.__init__(agent)
    agent.model = FakeModel()
    agent.normalize = torch.nn.Identity()
    agent.kl_weight = 0.0
    agent.consistency_weight = 0.1
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(1))
    observations = {
        "state": torch.zeros(3, 2),
        "rgb": torch.zeros(3, 1, 3, 8, 8, dtype=torch.uint8),
        "_free_space_mask": torch.tensor([True, False, False]),
    }
    result = agent.compute_loss(observations, torch.zeros(3, 2, 1))
    assert len(agent.model.noises) == 2
    assert torch.equal(agent.model.noises[0], agent.model.noises[1])
    assert result["consistency"].item() == pytest.approx(0.5)


def test_pusht_rgb_conversion_uses_recorded_successful_env_states() -> None:
    assert replay_state_flags("PushT-v1") == ["--use-env-states"]
    assert replay_state_flags("StackCube-v1") == ["--use-first-env-state"]


def test_training_launcher_isolates_concurrent_wandb_services() -> None:
    text = (ROOT / "launchers/run_data_and_training_pai.sh").read_text()
    assert 'TMPDIR="${worker_tmp}"' in text
    assert 'WANDB_DIR="${wandb_dir}"' in text
    assert "export WANDB__SERVICE_WAIT=300" in text


def test_fail_on_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    atomic_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        atomic_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_protocol_freeze_has_required_status_order() -> None:
    import yaml
    value = yaml.safe_load((ROOT / "preregistration.yaml").read_text())
    assert value["final_status"]["precedence"] == [
        "NO_GO_CAUSAL_BACKEND", "NO_GO_CORE_MECHANISM", "REVISE_VISUAL_ONLY",
        "REVISE_SHARED_AXIS_ROUTER", "GO_FULL_JOINT",
    ]
    assert value["execution_contract"]["run_all_oracle_arms_after_any_gate_failure"] is True


def test_paired_summary_clusters_model_seeds_by_source_episode() -> None:
    result = paired_summary([1.0, 3.0, -1.0, 1.0], [("task", "ep0"), ("task", "ep0"), ("task", "ep1"), ("task", "ep1")], n=1000)
    assert result["cluster_count"] == 2
    assert result["mean"] == pytest.approx(1.0)


def test_holm_is_monotone_in_sorted_pvalues() -> None:
    result = holm({"a": .01, "b": .03, "c": .2})
    assert result["a"] <= result["b"] <= result["c"]
