#!/usr/bin/env python3
"""Fail-closed contract tests for the R16-P18 LIBERO baseline gate."""

from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "r16p18-libero-stage1-baseline-exp-efficiency-2gpu-formal.json"


def load_script(name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    if spec is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


PAI_JOB = load_script("pai_job_r16p18_contract_test", ROOT / "bin" / "pai-job")


class R16P18LiberoStage1ContractTest(unittest.TestCase):
    def resolved(self) -> dict:
        _, resolved = PAI_JOB.resolve_template(
            TEMPLATE,
            "r16-p18-contract-unit-test",
            with_wandb=True,
        )
        return resolved

    def test_exact_formal_contract_resolves(self) -> None:
        resolved = self.resolved()
        self.assertEqual(
            resolved["resource_alias"],
            "exp-efficiency-r16p18-libero-2gpu",
        )
        self.assertEqual(resolved["worker"]["gpu"], 2)
        self.assertEqual(resolved["fault_tolerance"]["aimaster_args"], "")
        self.assertEqual(
            resolved["fault_tolerance"]["maximum_platform_restarts"],
            0,
        )
        self.assertFalse(
            resolved["fault_tolerance"]["pai_automatic_fault_tolerance"]
        )
        self.assertEqual(
            resolved["runtime"]["secret_env_names"],
            ["WANDB_API_KEY", "WANDB_ENTITY"],
        )
        self.assertFalse(resolved["evidence"]["pai_probe_created"])
        self.assertTrue(resolved["evidence"]["data_protocol_deviation"])
        self.assertFalse(resolved["evidence"]["pilot_can_return_stage1_go"])
        self.assertFalse(
            resolved["evidence"]["adaptive_components_implemented_before_gate"]
        )
        self.assertEqual(
            resolved["evidence"]["registry_launcher_sha256"],
            resolved["runtime"]["payload_sha256"],
        )
        self.assertEqual(
            resolved["evidence"]["superseded_terminal_jobs"],
            [
                {
                    "run_id": "r16-p18-libero-stage1-bc-gate-20260812-002",
                    "job_id": "dlc1eloj62mdzw2y",
                    "expected_status": "Failed",
                    "expected_purpose": "formal-training",
                }
            ],
        )
        PAI_JOB.validate_resolved(resolved)

    def test_outer_command_and_frozen_payload_are_shell_syntax_valid(self) -> None:
        resolved = self.resolved()
        subprocess.run(
            ["/bin/sh", "-n", "-c", resolved["runtime"]["command"]],
            check=True,
        )
        payload = base64.b64decode(
            resolved["runtime"]["payload_base64"],
            validate=True,
        )
        subprocess.run(["/bin/bash", "-n"], input=payload, check=True)
        self.assertIn(
            b'cd "$PROJECT_DIR"\nexec "$PROJECT_DIR/scripts/',
            payload,
        )

    def test_worker_fault_wandb_and_evidence_drift_fail_closed(self) -> None:
        mutations = []

        worker = self.resolved()
        worker["worker"]["gpu"] = 1
        mutations.append(worker)

        fault = self.resolved()
        fault["fault_tolerance"]["maximum_platform_restarts"] = 1
        mutations.append(fault)

        wandb = self.resolved()
        wandb["runtime"]["wandb_entity_contract"]["entity"] = "other"
        mutations.append(wandb)

        evidence = self.resolved()
        evidence["evidence"]["optimizer_steps_per_model"] = 2999
        mutations.append(evidence)

        for index, resolved in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(PAI_JOB.Refused):
                PAI_JOB.validate_resolved(resolved)

    def test_template_uses_no_idle_or_4090_resource(self) -> None:
        requested = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(
            requested["resource_alias"],
            "exp-efficiency-r16p18-libero-2gpu",
        )
        self.assertNotIn("4090", json.dumps(requested).lower())
        self.assertFalse(requested["submission"]["disable_ecs_stock_check"])
        self.assertEqual(requested["fault_tolerance"]["aimaster_args"], "")

    def test_absent_persistent_evidence_is_incomplete_not_success(self) -> None:
        artifacts = PAI_JOB.verify_r16p18_artifacts(self.resolved())
        self.assertFalse(artifacts["complete"])
        self.assertFalse(artifacts["first_training_step"]["complete"])
        self.assertFalse(artifacts["first_completed_rollout"]["complete"])
        self.assertFalse(artifacts["baseline_gate"]["complete"])

    def test_replacement_cleanup_requires_exact_verified_absence(self) -> None:
        target = {
            "run_id": "failed-predecessor",
            "job_id": "dlcfailed123",
            "expected_status": "Failed",
            "expected_purpose": "formal-training",
        }
        resolved = {
            "run_id": "replacement",
            "evidence": {"superseded_terminal_jobs": [target]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            runs_root = Path(temporary)
            predecessor = runs_root / target["run_id"]
            predecessor.mkdir(mode=0o700)
            with mock.patch.object(PAI_JOB, "RUNS_ROOT", runs_root):
                incomplete = PAI_JOB.verify_r16p18_superseded_cleanup(resolved)
                self.assertFalse(incomplete["complete"])

                delete_evidence = predecessor / "pai-task-delete-evidence.json"
                delete_evidence.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "run_id": target["run_id"],
                            "job_id": target["job_id"],
                            "workspace_id": "179169",
                            "pre_delete_status": target["expected_status"],
                            "expected_purpose": target["expected_purpose"],
                            "performed_by_uid": 2254,
                            "performed_by_gid": 2254,
                            "cpfs_and_registry_evidence_preserved": True,
                            "absence": {"complete": True},
                            "complete": True,
                        }
                    ),
                    encoding="utf-8",
                )
                delete_evidence.chmod(0o600)
                complete = PAI_JOB.verify_r16p18_superseded_cleanup(resolved)
                self.assertTrue(complete["complete"])
                self.assertTrue(complete["targets"][0]["verified_absent"])


if __name__ == "__main__":
    unittest.main()
