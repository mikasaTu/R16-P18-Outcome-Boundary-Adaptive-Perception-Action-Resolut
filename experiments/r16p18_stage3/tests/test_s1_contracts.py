from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_s1_static  # noqa: E402
import compute_feasibility  # noqa: E402
import profile_s1_costs  # noqa: E402
import update_sha256s  # noqa: E402


def _profile(
    *,
    visual_wall=(1.0, 2.0),
    visual_flops=(1.0, 2.0),
    action_wall=(1.0, 2.0),
    action_flops=(1.0, 2.0),
    native=True,
):
    rows = []
    for axis, values in (
        ("visual", {"coarse": (visual_wall[0], visual_flops[0]), "fine": (visual_wall[1], visual_flops[1])}),
        ("action", {"coarse": (action_wall[0], action_flops[0]), "fine": (action_wall[1], action_flops[1])}),
    ):
        for resolution, (wall, flops) in values.items():
            rows.append(
                {
                    "axis": axis,
                    "resolution": resolution,
                    "wall_clock_ms_median": wall,
                    "flops_per_state": flops,
                    "native_supported": native,
                }
            )
    return {
        "protocol_id": compute_feasibility.PROTOCOL_ID,
        "evidence_mode": "fresh_profile",
        "native_support": {
            "visual": {"coarse": native, "fine": native},
            "action": {"coarse": native, "fine": native},
        },
        "samples": rows,
    }


def test_formula_boundaries():
    assert compute_feasibility.feasibility_fraction(0.75, 0.5, False) == pytest.approx(0.25)
    assert compute_feasibility.feasibility_fraction(0.75, 0.5, True) == pytest.approx(0.5)
    assert compute_feasibility.feasibility_fraction(0.25, 0.5, False) == 0.0
    assert compute_feasibility.feasibility_fraction(0.75, 1.0, False) == 0.0
    assert compute_feasibility.feasibility_fraction(0.75, 1.2, True) == 0.0
    with pytest.raises(ValueError):
        compute_feasibility.feasibility_fraction(0.5, 0.0, False)


def test_schedule_reproduction_uses_archived_query_counts(tmp_path):
    statistics_path = tmp_path / "statistics.json"
    records = []
    for condition, global_calls, fine_calls, utility in (
        ("CC", 44, 40, 0.0),
        ("FF_tile0", 56, 56, 1.0),
        ("FF_tile1", 54, 54, 0.5),
    ):
        records.append(
            {
                "task": "StackCube-v1",
                "model_seed": 16018,
                "bank": "confirmatory",
                "bank_id": "state-0",
                "condition": condition,
                "utility": {"balanced": utility},
                "accounting": {
                    "global_encoder_calls": global_calls,
                    "fine_encoder_calls": fine_calls,
                },
            }
        )
    statistics_path.write_text(json.dumps({"aggregated_state_treatments": records}))
    rows = [
        {
            "axis": "visual",
            "resolution": "coarse",
            "query_count": 2,
            "wall_clock_ms_median": 2.0,
            "wall_clock_ms_samples": [2.0, 2.0],
            "flops_per_state": 20.0,
        },
        {
            "axis": "visual",
            "resolution": "fine",
            "query_count": 2,
            "wall_clock_ms_median": 4.0,
            "wall_clock_ms_samples": [4.0, 4.0],
            "flops_per_state": 40.0,
        },
    ]
    result = profile_s1_costs._schedule_reproduction(statistics_path, rows)
    assert result["coarse_query_count"] == pytest.approx(2.0)
    assert result["fine_continuation_query_count"] == pytest.approx(20.0)
    assert result["full_query_count"] == pytest.approx(28.0)
    assert result["measured_wall_clock_ratio"] == pytest.approx(0.75)
    assert result["measured_flops_ratio"] == pytest.approx(0.75)


def test_wall_flop_disagreement_is_reported():
    profile = _profile(
        visual_wall=(1.0, 2.0),  # rho_wall=.5 -> .25 at alpha=.75, no reuse
        visual_flops=(8.0, 10.0),  # rho_flops=.8 -> 0 at alpha=.75, no reuse
        action_wall=(1.0, 2.0),
        action_flops=(1.0, 2.0),
    )
    visual = compute_feasibility._axis_feasibility(profile, "visual")
    assert visual["disagreements"]
    assert any(
        row["alpha"] == pytest.approx(0.75)
        and row["sharing"] == "without_reuse"
        and row["threshold"] == pytest.approx(0.2)
        for row in visual["disagreements"]
    )
    assert visual["candidates_by_metric"]["wall_clock_ms"]["0.20"]
    assert not visual["candidates_by_metric"]["flops"]["0.20"]


def test_native_grid_gate_and_decision_precedence(tmp_path):
    profile = _profile(
        visual_wall=(1.0, 4.0),
        visual_flops=(1.0, 4.0),
        action_wall=(1.0, 4.0),
        action_flops=(1.0, 4.0),
        native=True,
    )
    profile["reproduction"] = {
        "measured_flops_ratio": compute_feasibility.REFERENCE["flops_ratio"],
        "measured_wall_clock_ratio": compute_feasibility.REFERENCE["wall_clock_ratio"],
        "source": "unit-test-reference",
    }
    output = tmp_path / "results"
    result = compute_feasibility.compute_ephemeral(
        profile, tmp_path / "archived.json", "0" * 64, output
    )
    assert result["decision"]["unique_label"] == "PROCEED_JOINT"
    assert result["decision"]["g1"]["4_native_resolution_support"]["status"] == "PASS"

    # A missing native module support must take substrate precedence even when
    # the arithmetic itself would leave a non-empty feasible region.
    profile_no_native = _profile(
        visual_wall=(1.0, 4.0),
        visual_flops=(1.0, 4.0),
        action_wall=(1.0, 4.0),
        action_flops=(1.0, 4.0),
        native=False,
    )
    result_no_native = compute_feasibility.compute_ephemeral(
        profile_no_native, tmp_path / "archived2.json", "1" * 64, tmp_path / "results2"
    )
    assert result_no_native["decision"]["unique_label"] == "BLOCKED_BY_SUBSTRATE"


def test_decision_budget_and_substrate_precedence():
    # Construct a valid feasibility object through the production path, then
    # exercise the two upstream precedence branches independently.
    profile = _profile(
        visual_wall=(1.0, 10.0),
        visual_flops=(1.0, 10.0),
        action_wall=(1.0, 10.0),
        action_flops=(1.0, 10.0),
    )
    axes = {axis: compute_feasibility._axis_feasibility(profile, axis) for axis in ("visual", "action")}
    feasibility = {"visual": axes["visual"], "action": axes["action"]}
    repro_ok = {"fresh_profile_available": True, "within_5_percent": True, "explanation_accepts_gate": False}
    decision = compute_feasibility._decision(repro_ok, feasibility)
    assert decision["unique_label"] == "PROCEED_JOINT"
    # Remove visual candidates while retaining an otherwise valid substrate.
    for axis in ("visual",):
        axes[axis]["candidates_both_metrics"]["0.20"] = []
    decision_budget = compute_feasibility._decision(repro_ok, feasibility)
    assert decision_budget["unique_label"] == "BLOCKED_BY_BUDGET"
    assert compute_feasibility._decision(
        {"fresh_profile_available": False, "within_5_percent": True, "explanation_accepts_gate": False},
        feasibility,
    )["unique_label"] == "BLOCKED_BY_SUBSTRATE"


def test_static_audit_and_manifest(tmp_path):
    audit_output = tmp_path / "static.json"
    result = audit_s1_static.audit(ROOT, audit_output)
    assert result["status"] == "PASS"
    manifest_root = tmp_path / "manifest_root"
    manifest_root.mkdir()
    (manifest_root / "a.txt").write_text("alpha\n", encoding="utf-8")
    (manifest_root / "nested").mkdir()
    (manifest_root / "nested" / "b.txt").write_text("beta\n", encoding="utf-8")
    manifest = manifest_root / "SHA256SUMS"
    generated = update_sha256s.generate(manifest_root, manifest)
    assert generated["status"] == "PASS"
    assert update_sha256s.verify(manifest_root, manifest)["entries"] == 2
    assert "SHA256SUMS" not in manifest.read_text(encoding="utf-8")


def test_archived_fallback_keeps_fresh_gate_failed(tmp_path):
    archive = tmp_path / "archived.json"
    archive.write_text(
        json.dumps(
            {
                "compute_budget": {
                    "all_coarse_cost": 92438200000000.0,
                    "all_fine_cost": 125120200000000.0,
                }
            }
        ),
        encoding="utf-8",
    )
    result = compute_feasibility.compute_from_sources(None, archive, tmp_path / "fallback")
    assert result["reproduction"]["status"] == "ARCHIVED_REFERENCE_ONLY"
    assert result["reproduction"]["fresh_profile_available"] is False
    assert result["decision"]["unique_label"] == "BLOCKED_BY_SUBSTRATE"
    assert result["decision"]["g1"]["4_native_resolution_support"]["status"] == "PASS"
    assert (tmp_path / "fallback" / "S1_DECISION.md").read_text().strip().endswith(
        "BLOCKED_BY_SUBSTRATE"
    )
    # FLOP-only feasibility remains visible for diagnostics, while wall-clock
    # values are correctly unavailable instead of being fabricated.
    feasibility = json.loads((tmp_path / "fallback" / "S1_FEASIBILITY.json").read_text())
    assert feasibility["axes"]["visual"]["costs"]["coarse"]["wall_clock_ms_median"] is None
    assert feasibility["axes"]["visual"]["candidates_by_metric"]["flops"]["0.20"]
    assert result["cost_curve"]["axes"]["visual"]["rows"][0]["flops_error_bar_stdev"] is None
    assert feasibility["axes"]["action"]["costs"]["fine"]["flops_per_state_median"] == pytest.approx(34.4e9)
    assert feasibility["axes"]["visual"]["costs"]["fine_grid2"]["flops_per_state_median"] == pytest.approx(15.8e9)
    assert feasibility["axes"]["visual"]["costs"]["fine_grid4"]["flops_per_state_median"] == pytest.approx(15.8e9)
