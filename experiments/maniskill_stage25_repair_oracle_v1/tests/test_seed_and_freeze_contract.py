from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "manifests" / name).read_text(encoding="utf-8"))


def test_seed_banks_are_disjoint_except_screen_prefix() -> None:
    screen = load("checkpoint_screen_seed_bank.json")["tasks"]
    final = load("checkpoint_final_val_seed_bank.json")["tasks"]
    confirm = load("confirmatory_test_seed_bank.json")["tasks"]
    oracle = load("oracle_source_seed_bank.json")["tasks"]
    used: set[int] = set()
    for task in sorted(final):
        assert len(screen[task]) == 32
        assert len(final[task]) == 100
        assert screen[task] == final[task][:32]
        assert len(confirm[task]) == 100
        assert len(oracle[task]["simulator_seeds"]) == 512
        for values in (final[task], confirm[task], oracle[task]["simulator_seeds"]):
            assert len(values) == len(set(values))
            assert not used.intersection(values)
            used.update(values)
    expert = [
        row["episode_seed"]
        for row in oracle["StackCube-v1"]["expert_source_episodes"]
    ]
    assert len(expert) == 96
    assert not used.intersection(expert)


def test_source_inventory_is_fully_verified() -> None:
    source = load("source_bindings.json")
    assert source["status"] == "STAGE2_SOURCE_AUDIT_PASS"
    assert source["checkpoint_candidate_count"] == 156
    assert source["checkpoint_payloads_verified_now"] is True
    assert all(row["payload_digest_verified_now"] for row in source["checkpoint_candidates"])

