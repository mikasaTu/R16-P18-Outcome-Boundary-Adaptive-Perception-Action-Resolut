from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    key: str
    name: str
    role: str
    baseline_success_min: float
    baseline_success_max: float


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def protocol_id(self) -> str:
        return str(self.raw["protocol_id"])

    @property
    def tasks(self) -> tuple[TaskConfig, ...]:
        return tuple(
            TaskConfig(
                key=key,
                name=value["name"],
                role=value["role"],
                baseline_success_min=float(value["baseline_success_min"]),
                baseline_success_max=float(value["baseline_success_max"]),
            )
            for key, value in self.raw["tasks"].items()
        )

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.raw["training"]["seeds"])

    def task(self, key: str) -> TaskConfig:
        for task in self.tasks:
            if task.key == key:
                return task
        raise KeyError(f"unknown task key: {key}")

    def canonical_json(self) -> str:
        return json.dumps(self.raw, sort_keys=True, separators=(",", ":"))


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    payload = path.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    _validate_config(raw)
    return ExperimentConfig(
        path=path,
        raw=raw,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _validate_config(raw: dict[str, Any]) -> None:
    if raw.get("protocol_id") != "R16-P18-LIBERO-STAGE1-PILOT-V1":
        raise ValueError("unexpected protocol_id")
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or tuple(tasks) != (
        "push_plate",
        "bottle_rack",
        "bowl_plate",
    ):
        raise ValueError("the three preregistered tasks or their order changed")
    model = raw.get("model", {})
    exact_model_fields = {
        "micro_feature_map": [8, 8],
        "baseline_visual_tokens": 16,
        "proprio_dim": 15,
        "action_horizon": 8,
        "execute_horizon": 4,
    }
    for key, expected in exact_model_fields.items():
        if model.get(key) != expected:
            raise ValueError(f"model.{key} must remain {expected!r}")
    training = raw.get("training", {})
    if training.get("seeds") != [16018, 16019, 16020]:
        raise ValueError("training seeds changed after preregistration")
    if training.get("complete_checkpoint_interval") == 1000:
        raise ValueError("this pilot intentionally uses a 250-step cadence")
    if raw.get("stop_rules", {}).get("stop_after_stage1") is not True:
        raise ValueError("stage-1 stop rule must be enabled")

