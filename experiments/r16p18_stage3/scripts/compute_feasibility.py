#!/usr/bin/env python3
"""Compute S1 cost curves, budget bounds, and the frozen G1 decision.

The input must be a profiler output made from a cached observation tensor and
an existing module.  This script performs no model calls.  It deliberately
keeps wall-clock and FLOP accounts separate and refuses to turn missing FLOP
metadata into a positive feasibility claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROTOCOL_ID = "R16-P18-STAGE3-S1-BUDGET-FEASIBILITY-V1"
ALPHAS = (0.25, 0.50, 0.75)
THRESHOLDS = (0.10, 0.20, 0.30)
METRICS = ("wall_clock_ms", "flops")
SHARING = ("without_reuse", "with_coarse_reuse")
REFERENCE = {
    "flops_numerator": 92438200000000.0,
    "flops_denominator": 125120200000000.0,
    "flops_ratio": 0.738795174560143,
    "wall_clock_ratio": 0.7502597918865904,
}
ARCHIVED_PURE_AXIS_FLOPS = {
    # These are code-derived fixed-window proxies, not fresh measurements.
    # Visual keeps the 4-step action schedule; action changes only the query
    # interval for the fixed eight-output chunk.
    "visual": {"coarse": 8.6e9, "fine": 15.8e9},
    "action": {"coarse": 8.6e9, "fine": 34.4e9},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"fail-on-overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty cost sample")
    return float(statistics.median(float(value) for value in values))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _cost_from_row(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    wall = row.get("wall_clock_ms_median", row.get("wall_clock_ms", row.get("wall_ms")))
    flops = row.get("flops_per_state", row.get("flops", row.get("estimated_flops")))
    if isinstance(wall, Sequence) and not isinstance(wall, (str, bytes)):
        wall = _median([float(item) for item in wall])
    else:
        wall = _number(wall)
    if isinstance(flops, Sequence) and not isinstance(flops, (str, bytes)):
        flops = _median([float(item) for item in flops])
    else:
        flops = _number(flops)
    return wall, flops


def _native_support(profile: Mapping[str, Any], axis: str, resolution: str) -> bool:
    support = profile.get("native_support", {})
    if not isinstance(support, Mapping):
        return False
    axis_support = support.get(axis, {})
    if not isinstance(axis_support, Mapping):
        return False
    return axis_support.get(resolution) is True


def _extract_costs(profile: Mapping[str, Any], axis: str) -> dict[str, dict[str, Any]]:
    """Aggregate profiler rows by native axis/resolution."""
    grouped: dict[str, list[tuple[float | None, float | None]]] = defaultdict(list)
    rows = profile.get("samples", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("profile.samples must be a list")
    for row in rows:
        if not isinstance(row, Mapping) or row.get("axis") != axis:
            continue
        resolution = row.get("resolution")
        if not isinstance(resolution, str) or not resolution:
            raise ValueError(f"missing {axis} resolution in profile row")
        wall, flops = _cost_from_row(row)
        raw_wall = row.get("wall_clock_ms_samples")
        if isinstance(raw_wall, Sequence) and not isinstance(raw_wall, (str, bytes)):
            finite_wall = [_number(item) for item in raw_wall]
            finite_wall = [item for item in finite_wall if item is not None]
            grouped[resolution].extend((item, flops) for item in finite_wall)
        else:
            grouped[resolution].append((wall, flops))
    if not grouped:
        axes = profile.get("axes", {})
        axis_payload = axes.get(axis) if isinstance(axes, Mapping) else None
        resolutions = axis_payload.get("resolutions") if isinstance(axis_payload, Mapping) else None
        if isinstance(resolutions, Mapping):
            for resolution, payload in resolutions.items():
                if not isinstance(resolution, str) or not isinstance(payload, Mapping):
                    continue
                grouped[resolution].append(_cost_from_row(payload))
    if not grouped:
        raise ValueError(f"profile contains no rows for axis {axis}")
    result: dict[str, dict[str, Any]] = {}
    for resolution, values in grouped.items():
        wall_values = [value[0] for value in values if value[0] is not None]
        flops_values = [value[1] for value in values if value[1] is not None]
        result[resolution] = {
            "wall_clock_ms_median": _median(wall_values) if wall_values else None,
            "flops_per_state_median": _median(flops_values) if flops_values else None,
            "wall_clock_ms_samples": wall_values,
            "flops_per_state_samples": flops_values,
            "native_supported": _native_support(profile, axis, resolution),
        }
    return result


def _pairs(profile: Mapping[str, Any], axis: str, costs: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs_payload = profile.get("resolution_pairs")
    if isinstance(pairs_payload, Mapping):
        axis_pairs = pairs_payload.get(axis)
        if isinstance(axis_pairs, Sequence) and not isinstance(axis_pairs, (str, bytes)):
            pairs = []
            for pair in axis_pairs:
                if isinstance(pair, Sequence) and len(pair) == 2:
                    coarse, fine = str(pair[0]), str(pair[1])
                    if coarse in costs and fine in costs:
                        pairs.append((coarse, fine))
            if pairs:
                return pairs
    if "coarse" in costs and "fine" in costs:
        return [("coarse", "fine")]
    names = list(costs)
    if len(names) < 2:
        raise ValueError(f"axis {axis} needs at least two resolution points")
    return [(names[index], names[index + 1]) for index in range(len(names) - 1)]


def feasibility_fraction(alpha: float, rho: float, reuse: bool) -> float:
    """Return a clipped k/N bound for the two preregistered cost conventions."""
    alpha = float(alpha)
    rho = float(rho)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be finite and positive")
    # A coarse cost no smaller than fine leaves no room for a coarse baseline
    # followed by a refinement under any preregistered alpha < 1.
    if rho >= 1.0 or alpha < rho:
        return 0.0
    bound = (alpha - rho) / (1.0 - rho) if reuse else alpha - rho
    return max(0.0, min(1.0, float(bound)))


def _ratio(coarse: float | None, fine: float | None) -> float | None:
    if coarse is None or fine is None or fine <= 0.0:
        return None
    return float(coarse / fine)


def _axis_feasibility(profile: Mapping[str, Any], axis: str) -> dict[str, Any]:
    costs = _extract_costs(profile, axis)
    pairs = _pairs(profile, axis, costs)
    combinations: list[dict[str, Any]] = []
    for coarse_name, fine_name in pairs:
        coarse = costs[coarse_name]
        fine = costs[fine_name]
        ratios = {
            "wall_clock_ms": _ratio(coarse["wall_clock_ms_median"], fine["wall_clock_ms_median"]),
            "flops": _ratio(coarse["flops_per_state_median"], fine["flops_per_state_median"]),
        }
        row: dict[str, Any] = {
            "coarse": coarse_name,
            "fine": fine_name,
            "coarse_native": bool(coarse["native_supported"]),
            "fine_native": bool(fine["native_supported"]),
            "native_pair": bool(coarse["native_supported"] and fine["native_supported"]),
            "rho": ratios,
            "alphas": {},
        }
        for alpha in ALPHAS:
            alpha_key = f"{alpha:.2f}"
            row["alphas"][alpha_key] = {}
            for metric in METRICS:
                rho = ratios[metric]
                row["alphas"][alpha_key][metric] = {}
                for share in SHARING:
                    if rho is None:
                        value = None
                    else:
                        value = feasibility_fraction(alpha, rho, share == "with_coarse_reuse")
                    row["alphas"][alpha_key][metric][share] = {
                        "k_over_N_upper_bound": value,
                        "meets_0.10": bool(value is not None and value >= 0.10),
                        "meets_0.20": bool(value is not None and value >= 0.20),
                        "meets_0.30": bool(value is not None and value >= 0.30),
                    }
        combinations.append(row)
    candidates_by_metric = {
        metric: {
            f"{threshold:.2f}": [
                {
                    "coarse": row["coarse"],
                    "fine": row["fine"],
                    "alpha": float(alpha_key),
                    "sharing": sharing,
                    "native_pair": row["native_pair"],
                    "k_over_N": row["alphas"][alpha_key][metric][sharing]["k_over_N_upper_bound"],
                }
                for row in combinations
                for alpha_key in row["alphas"]
                for sharing in SHARING
                if row["alphas"][alpha_key][metric][sharing]["k_over_N_upper_bound"] is not None
                and row["alphas"][alpha_key][metric][sharing]["k_over_N_upper_bound"] >= threshold
            ]
            for threshold in THRESHOLDS
        }
        for metric in METRICS
    }
    candidates = {
        f"{threshold:.2f}": [
            {
                "coarse": row["coarse"],
                "fine": row["fine"],
                "alpha": float(alpha_key),
                "sharing": sharing,
                "native_pair": row["native_pair"],
                "wall_clock_k_over_N": row["alphas"][alpha_key]["wall_clock_ms"][sharing]["k_over_N_upper_bound"],
                "flops_k_over_N": row["alphas"][alpha_key]["flops"][sharing]["k_over_N_upper_bound"],
            }
            for row in combinations
            for alpha_key in row["alphas"]
            for sharing in SHARING
            if row["alphas"][alpha_key]["wall_clock_ms"][sharing]["k_over_N_upper_bound"] is not None
            and row["alphas"][alpha_key]["flops"][sharing]["k_over_N_upper_bound"] is not None
            and min(
                row["alphas"][alpha_key]["wall_clock_ms"][sharing]["k_over_N_upper_bound"],
                row["alphas"][alpha_key]["flops"][sharing]["k_over_N_upper_bound"],
            ) >= threshold
        ]
        for threshold in THRESHOLDS
    }
    return {
        "axis": axis,
        "costs": costs,
        "resolution_pairs": pairs,
        "combinations": combinations,
        "candidates_by_metric": candidates_by_metric,
        "candidates_both_metrics": candidates,
        "disagreements": _find_disagreements(combinations),
    }


def _find_disagreements(combinations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for combination in combinations:
        for alpha_key, alpha_payload in combination["alphas"].items():
            for sharing in SHARING:
                wall = alpha_payload["wall_clock_ms"][sharing]["k_over_N_upper_bound"]
                flops = alpha_payload["flops"][sharing]["k_over_N_upper_bound"]
                if wall is None or flops is None:
                    continue
                for threshold in THRESHOLDS:
                    if (wall >= threshold) != (flops >= threshold):
                        rows.append(
                            {
                                "coarse": combination["coarse"],
                                "fine": combination["fine"],
                                "alpha": float(alpha_key),
                                "sharing": sharing,
                                "threshold": threshold,
                                "wall_clock_k_over_N": wall,
                                "flops_k_over_N": flops,
                            }
                        )
    return rows


def _reproduction(profile: Mapping[str, Any], axes: Mapping[str, Any]) -> dict[str, Any]:
    source = profile.get("reproduction")
    if not isinstance(source, Mapping):
        visual = axes["visual"]["costs"]
        if "coarse" in visual and "fine" in visual:
            source = {
                "measured_flops_ratio": _ratio(
                    visual["coarse"]["flops_per_state_median"],
                    visual["fine"]["flops_per_state_median"],
                ),
                "measured_wall_clock_ratio": _ratio(
                    visual["coarse"]["wall_clock_ms_median"],
                    visual["fine"]["wall_clock_ms_median"],
                ),
                "source": "visual_axis_coarse_fine_profile",
            }
        else:
            source = {}
    measured_flops = _number(source.get("measured_flops_ratio"))
    measured_wall = _number(source.get("measured_wall_clock_ratio"))
    flops_deviation = (
        abs(measured_flops - REFERENCE["flops_ratio"]) / REFERENCE["flops_ratio"]
        if measured_flops is not None
        else None
    )
    wall_deviation = (
        abs(measured_wall - REFERENCE["wall_clock_ratio"]) / REFERENCE["wall_clock_ratio"]
        if measured_wall is not None
        else None
    )
    explanation = source.get("verifiable_explanation")
    explanation_pass = isinstance(explanation, str) and bool(explanation.strip())
    within_five = (
        measured_flops is not None
        and measured_wall is not None
        and flops_deviation is not None
        and wall_deviation is not None
        and flops_deviation <= 0.05
        and wall_deviation <= 0.05
    )
    fresh = bool(profile.get("evidence_mode", "fresh_profile") == "fresh_profile")
    measurement = profile.get("measurement", {})
    if not isinstance(measurement, Mapping):
        measurement = {}
    if not fresh:
        status = "ARCHIVED_REFERENCE_ONLY"
    else:
        status = "PASS" if within_five or explanation_pass else "UNAVAILABLE_OR_FAIL"
    return {
        "original": REFERENCE,
        "measured": {
            "flops_ratio": measured_flops,
            "wall_clock_ratio": measured_wall,
            "source": source.get("source", "profile") if isinstance(source, Mapping) else "profile",
        },
        "relative_deviation": {
            "flops": flops_deviation,
            "wall_clock": wall_deviation,
        },
        "measurement_variance": source.get("measurement_variance", {
            "wall_clock_ratio": None,
            "flops_ratio": None,
        }),
        "measurement_protocol": {
            "batch_size": measurement.get("batch_size", 1 if fresh else None),
            "warmup": measurement.get("warmup"),
            "repeats": measurement.get("repeats"),
            "cuda_synchronize": measurement.get("cuda_synchronize"),
            "device": measurement.get("device"),
        },
        "within_5_percent": within_five,
        "fresh_profile_available": fresh,
        "verifiable_explanation": explanation,
        "explanation_accepts_gate": explanation_pass,
        "status": status,
        "new_measurement_required": not (fresh and (within_five or explanation_pass)),
    }


def _cost_curve(axes: Mapping[str, Any]) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for axis, payload in axes.items():
        costs = payload["costs"]
        full_names = [name for name in costs if name == "fine"] or list(costs)[-1:]
        full = costs[full_names[0]] if full_names else {}
        full_wall = full.get("wall_clock_ms_median")
        full_flops = full.get("flops_per_state_median")
        rows = []
        for name, value in costs.items():
            rows.append(
                {
                    "resolution": name,
                    "native_supported": value["native_supported"],
                    "wall_clock_ms_median": value["wall_clock_ms_median"],
                    "wall_clock_ms_error_bar_stdev": _stdev(value["wall_clock_ms_samples"]),
                    "wall_clock_normalized_to_fine": _safe_divide(value["wall_clock_ms_median"], full_wall),
                    "wall_clock_normalized_error_bar_stdev": _safe_divide(
                        _stdev(value["wall_clock_ms_samples"]), full_wall
                    ),
                    "flops_per_state_median": value["flops_per_state_median"],
                    "flops_error_bar_stdev": _stdev(value["flops_per_state_samples"]),
                    "flops_normalized_to_fine": _safe_divide(value["flops_per_state_median"], full_flops),
                }
            )
        curves[axis] = {
            "full_resolution": full_names[0] if full_names else None,
            "rows": rows,
            "wall_clock_primary": True,
            "flops_secondary": True,
        }
    return {"protocol_id": PROTOCOL_ID, "axes": curves}


def _archived_profile(path: Path) -> tuple[dict[str, Any], str]:
    """Adapt immutable Stage-2.7R accounting into a diagnostic-only profile.

    The archived source contains FLOP totals and the historical wall-clock
    ratio, but not fresh batch-size-one samples.  Consequently wall-clock
    samples remain null and this profile can never satisfy G1/S1.1.
    """
    if not path.is_file():
        raise FileNotFoundError(f"archived accounting not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("archived accounting root must be a JSON object")
    budget = payload.get("compute_budget")
    if not isinstance(budget, Mapping):
        raise ValueError("archived accounting has no compute_budget object")
    coarse = _number(budget.get("all_coarse_cost"))
    fine = _number(budget.get("all_fine_cost"))
    if coarse is None or fine is None or fine <= 0:
        raise ValueError("archived accounting has invalid coarse/fine FLOP totals")
    historical_wall_ratio = _number(budget.get("wall_clock_ratio"))
    if historical_wall_ratio is None:
        historical_wall_ratio = REFERENCE["wall_clock_ratio"]
    pure_axis = ARCHIVED_PURE_AXIS_FLOPS
    profile = {
        "protocol_id": PROTOCOL_ID,
        "evidence_mode": "archived_stage27r_accounting",
        "native_support": {
            "visual": {"coarse": True, "fine_grid2": True, "fine_grid4": True},
            "action": {"coarse": True, "fine": True},
        },
        "resolution_pairs": {
            "visual": [["coarse", "fine_grid2"], ["coarse", "fine_grid4"]],
            "action": [["coarse", "fine"]],
        },
        "samples": [
            {
                "axis": "visual",
                "resolution": "coarse",
                "wall_clock_ms_median": None,
                "flops_per_state": pure_axis["visual"]["coarse"],
                "fresh_measurement": False,
                "source": str(path),
            },
            {
                "axis": "visual",
                "resolution": "fine_grid2",
                "wall_clock_ms_median": None,
                "flops_per_state": pure_axis["visual"]["fine"],
                "fresh_measurement": False,
                "source": str(path),
            },
            {
                "axis": "visual",
                "resolution": "fine_grid4",
                "wall_clock_ms_median": None,
                "flops_per_state": pure_axis["visual"]["fine"],
                "fresh_measurement": False,
                "source": str(path),
            },
            {
                "axis": "action",
                "resolution": "coarse",
                "wall_clock_ms_median": None,
                "flops_per_state": pure_axis["action"]["coarse"],
                "fresh_measurement": False,
                "source": str(path),
            },
            {
                "axis": "action",
                "resolution": "fine",
                "wall_clock_ms_median": None,
                "flops_per_state": pure_axis["action"]["fine"],
                "fresh_measurement": False,
                "source": str(path),
            },
        ],
        "reproduction": {
            # Reproduction is intentionally the historical joint CC/FF ratio;
            # pure-axis proxies above must never overwrite this S1.1 field.
            "measured_flops_ratio": coarse / fine,
            "measured_wall_clock_ratio": None,
            "archived_wall_clock_ratio": historical_wall_ratio,
            "source": "immutable_stage27r_accounting",
            "verifiable_explanation": None,
        },
        "reference_stage27r": REFERENCE,
        "archived_source": {
            "path": str(path),
            "sha256": sha256_file(path),
            "new_measurement": False,
        },
        "archived_pure_axis_proxy": {
            "visual": pure_axis["visual"],
            "action": pure_axis["action"],
            "units": "estimated_flops_per_fixed_eight-step_window",
            "source": "code-derived schedule proxy; not fresh profiling",
        },
    }
    return profile, sha256_file(path)


def _stdev(values: Sequence[float]) -> float | None:
    # A single analytic proxy is not a repeated measurement. Reporting zero
    # here would falsely imply an observed zero-variance sample.
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _safe_divide(value: float | None, denominator: float | None) -> float | None:
    if value is None or denominator is None or denominator <= 0:
        return None
    return float(value / denominator)


def _svg_curve(axis: str, curve: Mapping[str, Any]) -> str:
    rows = curve["rows"]
    width, height = 760, 430
    margin = 65
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    series = [
        ("wall_clock_normalized_to_fine", "#1f77b4", "wall-clock"),
        ("flops_normalized_to_fine", "#d62728", "FLOPs"),
    ]
    all_values = []
    for row in rows:
        for key, _, _ in series:
            value = row.get(key)
            if value is None:
                continue
            upper = float(value)
            if key == "wall_clock_normalized_to_fine":
                upper += float(row.get("wall_clock_normalized_error_bar_stdev") or 0.0)
            all_values.append(upper)
    ymax = max(1.0, max(all_values, default=1.0) * 1.15)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="24" text-anchor="middle" font-size="16">{axis} cost curve</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
        f'<text x="{width/2}" y="414" text-anchor="middle" font-size="12">native resolution</text>',
        f'<text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle" font-size="12">cost normalized to fine</text>',
    ]
    for index, row in enumerate(rows):
        x = margin + (plot_w * index / max(1, len(rows) - 1))
        parts.append(f'<text x="{x:.2f}" y="{height-margin+20}" text-anchor="middle" font-size="11">{row["resolution"]}</text>')
    for key, colour, label in series:
        points = []
        for index, row in enumerate(rows):
            value = row.get(key)
            if value is None:
                continue
            x = margin + (plot_w * index / max(1, len(rows) - 1))
            y = height - margin - plot_h * float(value) / ymax
            points.append(f"{x:.2f},{y:.2f}")
            if key == "wall_clock_normalized_to_fine":
                error = row.get("wall_clock_normalized_error_bar_stdev")
                if error is not None:
                    y_hi = height - margin - plot_h * (float(value) + float(error)) / ymax
                    y_lo = height - margin - plot_h * max(0.0, float(value) - float(error)) / ymax
                    parts.append(
                        f'<line class="error-bar" x1="{x:.2f}" y1="{y_hi:.2f}" '
                        f'x2="{x:.2f}" y2="{y_lo:.2f}" stroke="{colour}" stroke-width="1"/>'
                    )
        if points:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" stroke-width="2"/>')
            for point in points:
                x, y = point.split(",")
                parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{colour}"/>')
        parts.append(f'<text x="{width-150}" y="{36 + 18*series.index((key, colour, label))}" fill="{colour}" font-size="12">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _svg_heatmap(axis: str, feasibility: Mapping[str, Any]) -> str:
    rows = feasibility["combinations"]
    width, height = 920, 520
    left, top, cell_w, cell_h = 190, 75, 110, 34
    labels = []
    cells = []
    for combo in rows:
        for alpha_key in sorted(combo["alphas"]):
            for metric in METRICS:
                for sharing in SHARING:
                    label = f'{combo["coarse"]}->{combo["fine"]} | {metric} | {sharing} | a={alpha_key}'
                    value = combo["alphas"][alpha_key][metric][sharing]["k_over_N_upper_bound"]
                    labels.append(label)
                    cells.append(value)
    height = max(height, top + cell_h * len(labels) + 30)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="26" text-anchor="middle" font-size="16">{axis} feasibility heatmap (k/N upper bound)</text>',
        f'<text x="{left+cell_w/2}" y="56" text-anchor="middle" font-size="12">bound</text>',
    ]
    for index, label in enumerate(labels):
        y = top + index * cell_h
        value = cells[index]
        colour = "#eeeeee" if value is None else _heat_colour(float(value))
        text = "NA" if value is None else f"{float(value):.3f}"
        parts.append(f'<text x="{left-8}" y="{y+22}" text-anchor="end" font-size="10">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{cell_w}" height="{cell_h-2}" fill="{colour}" stroke="white"/>')
        parts.append(f'<text x="{left+cell_w/2}" y="{y+22}" text-anchor="middle" font-size="11">{text}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _heat_colour(value: float) -> str:
    value = max(0.0, min(1.0, value))
    red = int(255 * (1.0 - value))
    green = int(220 * value)
    return f"rgb({red},{green},80)"


def _decision(repro: Mapping[str, Any], feasibility: Mapping[str, Any]) -> dict[str, Any]:
    visual_candidates = feasibility["visual"]["candidates_both_metrics"]["0.20"]
    action_candidates = feasibility["action"]["candidates_both_metrics"]["0.20"]
    visual_gate = any(row["native_pair"] for row in visual_candidates)
    action_gate = any(row["native_pair"] for row in action_candidates)
    # Native support is a substrate property, independent of whether a fresh
    # wall-clock measurement was available to certify the budget gate. Check
    # every enumerated pair instead of deriving G1.4 from G1.2/G1.3.
    enumerated_pairs = [
        {"axis": axis, **combination}
        for axis in ("visual", "action")
        for combination in feasibility[axis]["combinations"]
    ]
    native_gate = bool(enumerated_pairs) and all(
        row["native_pair"] for row in enumerated_pairs
    )
    # An archived Stage-2.7R ratio can populate diagnostic tables but cannot
    # satisfy S1.1: G1 requires a fresh batch-size-one measurement.
    numeric_repro_gate = bool(
        repro.get("fresh_profile_available")
        and (repro.get("within_5_percent") or repro.get("explanation_accepts_gate"))
    )
    repro_gate = bool(numeric_repro_gate and repro.get("protocol_device_compliant", True))
    if not repro_gate or not native_gate:
        label = "BLOCKED_BY_SUBSTRATE"
    elif not visual_gate:
        label = "BLOCKED_BY_BUDGET"
    elif not action_gate:
        label = "PROCEED_VISION_ONLY"
    else:
        label = "PROCEED_JOINT"
    if not numeric_repro_gate or not native_gate:
        diagnostic_label = "BLOCKED_BY_SUBSTRATE"
    elif not visual_gate:
        diagnostic_label = "BLOCKED_BY_BUDGET"
    elif not action_gate:
        diagnostic_label = "PROCEED_VISION_ONLY"
    else:
        diagnostic_label = "PROCEED_JOINT"
    conditional_flags = []
    if visual_gate and not any(
        row["sharing"] == "without_reuse" and row["native_pair"]
        for row in visual_candidates
    ):
        conditional_flags.append("VISUAL_GATE_REQUIRES_COARSE_REUSE")
    if action_gate and not any(
        row["sharing"] == "without_reuse" and row["native_pair"]
        for row in action_candidates
    ):
        conditional_flags.append("ACTION_GATE_REQUIRES_COARSE_REUSE")
    return {
        "protocol_id": PROTOCOL_ID,
        "flop_only_candidates": {
            axis: feasibility[axis]["candidates_by_metric"]["flops"]
            for axis in ("visual", "action")
        },
        "g1": {
            "1_cost_reproduction": {
                "status": "PASS" if repro_gate else "FAIL",
                "evidence": dict(repro),
            },
            "2_visual_k_over_N_ge_0.20": {
                "status": "PASS" if visual_gate else "FAIL",
                "candidates": visual_candidates,
            },
            "3_action_k_over_N_ge_0.20": {
                "status": "PASS" if action_gate else "FAIL",
                "candidates": action_candidates,
            },
            "4_native_resolution_support": {
                "status": "PASS" if native_gate else "FAIL",
                "evidence_tier": (
                    "source-confirmed native paths; fresh runtime forward verified"
                    if repro.get("fresh_profile_available")
                    else "source-confirmed native paths; fresh runtime forward unverified"
                ),
                "enumerated_pairs": [
                    {
                        "axis": row["axis"],
                        "coarse": row["coarse"],
                        "fine": row["fine"],
                        "native_pair": row["native_pair"],
                    }
                    for row in enumerated_pairs
                ],
            },
        },
        "metric_policy": "a candidate must meet the threshold under both wall-clock and FLOP accounts; disagreements are reported separately",
        "unique_label": label,
        "diagnostic_budget_geometry_label": diagnostic_label,
        "conditional_flags": conditional_flags,
        "termination": "G1 recorded; stop and await human confirmation; no S2 preparation",
        "notes": [
            "No budget conclusion is valid when the profile is unavailable or FLOP metadata is missing.",
            "A numerically matching profile cannot pass G1.1 when its device violates the frozen one-owner-safe requirement.",
            "Archived Stage-2.7R accounting is diagnostic fallback only; it cannot pass the fresh S1.1 reproduction gate.",
            "Action resolution means native query interval 4 versus 1 with a fixed 8-query output chunk.",
            "Visual resolution means native coarse/fine forward paths; no blur or synthetic resize substitute is accepted.",
        ],
    }


def compute(
    profile_path: Path,
    output_dir: Path,
    runtime_audit_path: Path | None = None,
) -> dict[str, Any]:
    if not profile_path.is_file():
        raise FileNotFoundError(f"profile not found: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, Mapping):
        raise ValueError("profile root must be a JSON object")
    if profile.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("profile protocol_id mismatch")
    axes = {axis: _axis_feasibility(profile, axis) for axis in ("visual", "action")}
    repro = _reproduction(profile, axes)
    if runtime_audit_path is not None:
        runtime_audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
        co_tenant = runtime_audit.get("co_tenant_processes_present") is True
        repro["runtime_audit_path"] = str(runtime_audit_path)
        repro["runtime_audit_sha256"] = sha256_file(runtime_audit_path)
        repro["protocol_device_requirement"] = "one_owner_safe_cuda_gpu"
        repro["protocol_device_compliant"] = not co_tenant
        repro["protocol_device_reason"] = (
            "foreign-owner GPU processes were present during profiling"
            if co_tenant
            else "no co-tenant GPU process was recorded"
        )
        if co_tenant:
            repro["status"] = "NUMERIC_PASS_PROTOCOL_DEVICE_FAIL"
            repro["new_measurement_required"] = True
    curve = _cost_curve(axes)
    feasibility = {
        "protocol_id": PROTOCOL_ID,
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "alphas": list(ALPHAS),
        "thresholds": list(THRESHOLDS),
        "metrics": list(METRICS),
        "sharing_conventions": {
            "without_reuse": "k/N <= alpha - rho",
            "with_coarse_reuse": "k/N <= (alpha - rho)/(1-rho)",
        },
        "implementation_accounting": {
            "visual_current_code": "without_reuse",
            "action_current_code": "without_reuse",
            "g1_enumeration": "both preregistered sharing conventions",
        },
        "axes": axes,
        "wall_flop_disagreements": {
            axis: payload["disagreements"] for axis, payload in axes.items()
        },
        "gate_candidates_threshold_0.20": {
            axis: payload["candidates_both_metrics"]["0.20"]
            for axis, payload in axes.items()
        },
    }
    decision = _decision(repro, feasibility["axes"])
    output_dir = Path(output_dir)
    atomic_json(output_dir / "S1_COST_REPRO.json", repro)
    atomic_json(output_dir / "S1_COST_CURVE.json", curve)
    atomic_json(output_dir / "S1_FEASIBILITY.json", feasibility)
    atomic_text(output_dir / "S1_DECISION.md", _decision_markdown(decision))
    for axis in ("visual", "action"):
        atomic_text(output_dir / "plots" / f"{axis}_cost_curve.svg", _svg_curve(axis, curve["axes"][axis]))
        atomic_text(output_dir / "plots" / f"{axis}_feasibility_heatmap.svg", _svg_heatmap(axis, feasibility["axes"][axis]))
    return {
        "reproduction": repro,
        "cost_curve": curve,
        "feasibility": feasibility,
        "decision": decision,
    }


def compute_from_sources(
    profile_path: Path | None,
    archived_path: Path | None,
    output_dir: Path,
    runtime_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Run from a fresh profile or immutable Stage-2.7R diagnostic fallback."""
    if (profile_path is None) == (archived_path is None):
        raise ValueError("provide exactly one of --profile or --archived-stage27r-accounting")
    if profile_path is not None:
        return compute(profile_path, output_dir, runtime_audit_path)
    assert archived_path is not None
    profile, source_sha = _archived_profile(archived_path)
    # Keep the fallback self-describing without writing a synthetic profile
    # that could be mistaken for a fresh measurement.
    result = compute_ephemeral(profile, archived_path, source_sha, output_dir)
    return result


def compute_ephemeral(
    profile: Mapping[str, Any],
    source_path: Path,
    source_sha: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Shared writer for fallback data held in memory."""
    axes = {axis: _axis_feasibility(profile, axis) for axis in ("visual", "action")}
    repro = _reproduction(profile, axes)
    curve = _cost_curve(axes)
    feasibility = {
        "protocol_id": PROTOCOL_ID,
        "evidence_mode": profile.get("evidence_mode"),
        "profile_path": None,
        "archived_source_path": str(source_path),
        "archived_source_sha256": source_sha,
        "alphas": list(ALPHAS),
        "thresholds": list(THRESHOLDS),
        "metrics": list(METRICS),
        "sharing_conventions": {
            "without_reuse": "k/N <= alpha - rho",
            "with_coarse_reuse": "k/N <= (alpha-rho)/(1-rho)",
        },
        "implementation_accounting": {
            "visual_current_code": "without_reuse",
            "action_current_code": "without_reuse",
            "g1_enumeration": "both preregistered sharing conventions",
        },
        "axes": axes,
        "wall_flop_disagreements": {axis: payload["disagreements"] for axis, payload in axes.items()},
        "gate_candidates_threshold_0.20": {
            axis: payload["candidates_both_metrics"]["0.20"] for axis, payload in axes.items()
        },
        "diagnostic_only": True,
        "fresh_profile_available": False,
    }
    decision = _decision(repro, feasibility["axes"])
    output_dir = Path(output_dir)
    atomic_json(output_dir / "S1_COST_REPRO.json", repro)
    atomic_json(output_dir / "S1_COST_CURVE.json", curve)
    atomic_json(output_dir / "S1_FEASIBILITY.json", feasibility)
    atomic_text(output_dir / "S1_DECISION.md", _decision_markdown(decision))
    for axis in ("visual", "action"):
        atomic_text(output_dir / "plots" / f"{axis}_cost_curve.svg", _svg_curve(axis, curve["axes"][axis]))
        atomic_text(output_dir / "plots" / f"{axis}_feasibility_heatmap.svg", _svg_heatmap(axis, feasibility["axes"][axis]))
    return {"reproduction": repro, "cost_curve": curve, "feasibility": feasibility, "decision": decision}


def _decision_markdown(decision: Mapping[str, Any]) -> str:
    lines = [
        "# S1 Decision",
        "",
        "This document is generated from the supplied profile. It records G1 only; it is not a rollout or training result.",
        "",
        "## G1 gates",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    for key, value in decision["g1"].items():
        lines.append(f"| {key} | **{value['status']}** |")
    lines += [
        "",
        f"**Unique label: `{decision['unique_label']}`**",
        "",
        f"Diagnostic budget-geometry label: `{decision['diagnostic_budget_geometry_label']}`",
        "",
        "Conditional flags: "
        + (", ".join(f"`{flag}`" for flag in decision["conditional_flags"]) or "none"),
        "",
        "G1.4 evidence tier: " + decision["g1"]["4_native_resolution_support"]["evidence_tier"] + ".",
        "",
        "G1.1 numeric reproduction within 5%: "
        + str(bool(decision["g1"]["1_cost_reproduction"]["evidence"].get("within_5_percent")))
        + "; protocol device compliant: "
        + str(decision["g1"]["1_cost_reproduction"]["evidence"].get("protocol_device_compliant", True))
        + ".",
        "",
        "### Candidate combinations at the preregistered 0.20 threshold",
        "",
    ]
    for gate in ("2_visual_k_over_N_ge_0.20", "3_action_k_over_N_ge_0.20"):
        lines.append(f"- `{gate}`:")
        candidates = decision["g1"][gate]["candidates"]
        if not candidates:
            lines.append("  - none under the conservative both-metrics rule")
        for candidate in candidates:
            lines.append(
                f"  - {candidate['coarse']} -> {candidate['fine']}, alpha={candidate['alpha']:.2f}, "
                f"{candidate['sharing']}, wall={candidate['wall_clock_k_over_N']:.4f}, "
                f"FLOPs={candidate['flops_k_over_N']:.4f}, native={candidate['native_pair']}"
            )
    lines += [
        "",
        "### Per-metric FLOP candidates (sensitivity table)",
        "",
        "These rows pass the FLOP account alone; G1 above uses the intersection with wall-clock.",
        "",
    ]
    for axis in ("visual", "action"):
        lines.append(f"- `{axis}` threshold counts (0.10/0.20/0.30): " + "/".join(
            str(len(decision["flop_only_candidates"][axis][threshold]))
            for threshold in ("0.10", "0.20", "0.30")
        ))
        for candidate in decision["flop_only_candidates"][axis]["0.20"]:
            lines.append(
                f"  - {candidate['coarse']} -> {candidate['fine']}, alpha={candidate['alpha']:.2f}, "
                f"{candidate['sharing']}, FLOPs={candidate['k_over_N']:.6f}, "
                f"native={candidate['native_pair']}"
            )
    lines += [
        "",
        "## Accounting policy",
        "",
        decision["metric_policy"],
        "",
        "Wall-clock/FLOP disagreement cells are listed in `S1_FEASIBILITY.json`; they are not silently resolved.",
        "",
        decision["termination"],
        "",
        decision["unique_label"],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", type=Path)
    source.add_argument("--archived-stage27r-accounting", type=Path)
    parser.add_argument("--runtime-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compute_from_sources(
        args.profile,
        args.archived_stage27r_accounting,
        args.output_dir,
        args.runtime_audit,
    )


if __name__ == "__main__":
    main()
