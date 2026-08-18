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
from posthoc_independent_audit import expected_schedule, recompute_outcome  # noqa: E402
from posthoc_independent_audit import lower_tile_tiebreak  # noqa: E402
from audit_formal_results import frozen_preregistration_digest, official_scientific_manifest  # noqa: E402
from resume_derived_output import run_or_validate  # noqa: E402
from validate_oracle_shard import expected_conditions  # noqa: E402
from install_formal_complete import MARKER, install_or_validate, validate_prerequisites  # noqa: E402


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
    assert "--num-workers 8" in text


def test_formal_precheck_runs_real_deterministic_smoke() -> None:
    text = (ROOT / "launchers/run_stage27r_formal_pai.sh").read_text()
    assert "deterministic_lockstep_smoke.py" in text
    assert "DETERMINISTIC_LOCKSTEP_SMOKE.json" in text
    assert "--dataset-root \"${data_root}\"" in text
    assert "--maniskill-root \"${ms}\"" in text
    assert "install_formal_complete.py" in text
    assert text.index("ORACLE_VALIDATION.json") < text.index('derived_output statistics')
    assert "producer_registry_evidence" in text
    assert "ORACLE_INPUT_SNAPSHOT.json" in text


def test_official_manifest_excludes_resume_candidates_and_derived_metadata() -> None:
    text = (ROOT / "scripts/audit_formal_results.py").read_text()
    for name in ("INDEPENDENT_AUDIT.json", "POSTHOC_INDEPENDENT_AUDIT.json", "FORMAL_COMPLETE.json"):
        assert f'"{name}"' in text
    for name in ("statistics.json", "MECHANISM_AUDIT.json", "RESULT_VECTOR.json"):
        assert f'"{name}"' not in text.split("derived_names =", 1)[1].split("}", 1)[0]
    assert '".resume-"' in text
    assert '".tmp-"' in text


def test_official_manifest_is_stable_and_covers_scientific_outputs(tmp_path: Path) -> None:
    included = ("statistics.json", "MECHANISM_AUDIT.json", "RESULT_VECTOR.json")
    excluded = (
        "INDEPENDENT_AUDIT.json",
        "POSTHOC_INDEPENDENT_AUDIT.json",
        "FORMAL_COMPLETE.json",
        ".statistics.json.resume-7-" + "a" * 32 + ".json",
        "statistics.json.tmp-7-" + "b" * 32 + ".json",
    )
    for name in included + excluded:
        path = tmp_path / name
        path.write_text(name + "\n", encoding="utf-8")
    output = tmp_path / "OFFICIAL_AUDIT.json"
    first = official_scientific_manifest(tmp_path, output)
    output.write_text(json.dumps({"manifest": first}), encoding="utf-8")
    second = official_scientific_manifest(tmp_path, output)
    assert first == second
    assert {entry["path"] for entry in first} == set(included)


def test_state_bank_fidelity_uses_independent_cpu_environments() -> None:
    text = (ROOT / "scripts/build_lockstep_state_bank.py").read_text()
    assert "make_env(task,2)" not in text
    assert "left=make_env(task,1); right=make_env(task,1)" in text


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


def test_independent_audit_recomputes_trace_outcomes_and_schedule() -> None:
    row = {
        "condition": "FC_tile0",
        "success_trace": [False, True, True, True, True, True, False, False, True],
        "reward_trace": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.55, 0.5, 0.7],
        "intended_contact_trace": [False, True, True, True, True, True, True, True, True],
        "grasp_trace": [False, True, True, True, True, True, False, False, False],
        "catastrophic_trace": [False] * 9,
    }
    outcome = recompute_outcome(row)
    assert outcome["success_once"] is True
    assert outcome["success_hold5"] is True
    assert outcome["longest_success_streak"] == 5
    assert outcome["normalized_progress"] == pytest.approx(0.6)
    assert outcome["dropped_or_slipped"] is False  # a stable hold5 is not a drop
    schedule = expected_schedule(row)
    assert schedule["executed_steps"] == 9
    assert schedule["policy_forward_calls"] == 3  # two coarse treatment + one fine continuation
    assert schedule["fine_encoder_calls"] == 3


def test_independent_tile_tie_break_matches_official_lower_tile_rule() -> None:
    assert lower_tile_tiebreak({"utility_value": 1.0, "condition": "FC_tile0"}) > lower_tile_tiebreak({"utility_value": 1.0, "condition": "FC_tile3"})


def test_frozen_preregistration_is_derived_from_immutable_git_snapshot() -> None:
    result = frozen_preregistration_digest(ROOT.parent.parent, ROOT)
    assert result["pass"] is True
    assert result["digest_field_present"] is False
    assert result["derived_from_frozen_git_commit"]


def _producer(value: str) -> list[str]:
    code = (
        "from pathlib import Path; "
        "Path('__OUTPUT__').write_text(" + repr(value) + ", encoding='utf-8')"
    )
    return [sys.executable, "-c", code, "--output", "__OUTPUT__"]


def test_resume_derived_output_preserves_existing_target_and_validates_equivalence(tmp_path: Path) -> None:
    target = tmp_path / "statistics.json"
    assert run_or_validate(target, _producer("same")) == "installed_missing"
    assert target.read_text() == "same"
    before = target.stat()
    assert run_or_validate(target, _producer("same")) == "validated_existing"
    assert target.read_text() == "same"
    after = target.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        run_or_validate(target, _producer("different"))
    assert target.read_text() == "same"
    assert target.stat().st_mtime_ns == before.st_mtime_ns


def test_resume_rejects_stale_existing_target_before_producer(tmp_path: Path) -> None:
    target = tmp_path / "marker.json"
    target.write_text('{"stale":true}\n', encoding="utf-8")
    before = target.read_bytes(), target.stat().st_mtime_ns
    validator = [
        sys.executable,
        str(ROOT / "scripts/validate_derived_output.py"),
        "--path",
        "__TARGET__",
        "--kind",
        "marker",
    ]
    with pytest.raises(Exception):
        run_or_validate(target, _producer(json.dumps(MARKER, sort_keys=True)), validator)
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert not list(tmp_path.glob(".marker.json.resume-*"))


def test_resume_interrupted_candidate_keeps_target_untouched(tmp_path: Path) -> None:
    target = tmp_path / "statistics.json"
    interrupted = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('__OUTPUT__').write_text('candidate'); raise SystemExit(17)",
        "--output",
        "__OUTPUT__",
    ]
    with pytest.raises(Exception):
        run_or_validate(target, interrupted)
    assert not target.exists()
    candidates = list(tmp_path.glob(".statistics.json.resume-*"))
    assert len(candidates) == 1
    assert candidates[0].read_text() == "candidate"
    assert run_or_validate(target, _producer("statistics")) == "installed_missing"


def test_resume_exclusive_install_handles_concurrent_equal_and_conflicting_target(tmp_path: Path) -> None:
    target = tmp_path / "statistics.json"
    equal_race = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path(r'__TARGET_PATH__').write_text('same'); Path('__OUTPUT__').write_text('same')",
        "--output",
        "__OUTPUT__",
    ]
    equal_race[2] = equal_race[2].replace("__TARGET_PATH__", str(target))
    assert run_or_validate(target, equal_race) == "validated_existing"
    target.unlink()
    conflict_race = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path(r'__TARGET_PATH__').write_text('winner'); Path('__OUTPUT__').write_text('loser')",
        "--output",
        "__OUTPUT__",
    ]
    conflict_race[2] = conflict_race[2].replace("__TARGET_PATH__", str(target))
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        run_or_validate(target, conflict_race)
    assert target.read_text() == "winner"


def test_formal_complete_marker_is_exclusive_and_stable(tmp_path: Path) -> None:
    marker = tmp_path / "FORMAL_COMPLETE.json"
    assert install_or_validate(marker, {}) == "installed_missing"
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    assert install_or_validate(marker, {}) == "validated_existing"
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before
    marker.write_text('{"protocol_id":"wrong","status":"FORMAL_COMPLETE"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="marker mismatch"):
        install_or_validate(marker, {})


def test_completion_gate_rejects_missing_or_invalid_prerequisites(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        validate_prerequisites(
            official_audit=tmp_path / "missing-official.json",
            posthoc_audit=tmp_path / "missing-posthoc.json",
            result_vector=tmp_path / "missing-result.json",
            oracle_validation=tmp_path / "missing-oracle.json",
        )
    marker = tmp_path / "FORMAL_COMPLETE.json"
    assert not marker.exists()


def test_confirmatory_oracle_contract_has_exact_34_conditions() -> None:
    conditions = expected_conditions(4)
    assert len(conditions) == 34
    assert conditions[:2] == ("CC", "CF")
    assert conditions[-1] == "FF_tile15"


def test_resume_interruption_boundaries_are_all_idempotent(tmp_path: Path) -> None:
    # These are the three boundaries an idle restart can observe: only oracle
    # shards, oracle plus statistics, or all official derived outputs before
    # the independent audit/terminal marker.
    for existing in ((), ("statistics.json",), ("statistics.json", "MECHANISM_AUDIT.json", "RESULT_VECTOR.json")):
        root = tmp_path / "-".join(existing or ("oracle",))
        root.mkdir()
        for name in existing:
            assert run_or_validate(root / name, _producer(name)) == "installed_missing"
        for name in ("statistics.json", "MECHANISM_AUDIT.json", "RESULT_VECTOR.json", "INDEPENDENT_AUDIT.json", "POSTHOC_INDEPENDENT_AUDIT.json"):
            expected = "" if name in existing else name
            if name in existing:
                assert run_or_validate(root / name, _producer(name)) == "validated_existing"
            else:
                assert run_or_validate(root / name, _producer(expected)) == "installed_missing"
        for name in ("statistics.json", "MECHANISM_AUDIT.json", "RESULT_VECTOR.json", "INDEPENDENT_AUDIT.json", "POSTHOC_INDEPENDENT_AUDIT.json"):
            assert run_or_validate(root / name, _producer(name)) == "validated_existing"
