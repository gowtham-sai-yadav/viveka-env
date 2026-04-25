"""Smoke tests for eval/holdout_eval.py — no GPU, no API key required."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_pick_holdout_is_deterministic():
    sys.path.insert(0, str(REPO))
    try:
        from eval.holdout_eval import pick_holdout
    finally:
        sys.path.pop(0)

    h1 = pick_holdout(seed=42)
    h2 = pick_holdout(seed=42)
    assert h1 == h2, "same seed should produce identical hold-out splits"
    assert len(h1) == 15, "default split is 5+5+5 = 15 scenarios"
    tiers = [t for t, _ in h1]
    assert tiers.count(2) == 5
    assert tiers.count(3) == 5
    assert tiers.count(4) == 5


def test_pick_holdout_excludes_t1():
    sys.path.insert(0, str(REPO))
    try:
        from eval.holdout_eval import pick_holdout
    finally:
        sys.path.pop(0)
    h = pick_holdout(seed=7)
    assert all(t != 1 for t, _ in h), "T1 must not appear in held-out set"


def test_holdout_eval_subprocess_random_only(tmp_path):
    out_json = tmp_path / "h.json"
    out_md = tmp_path / "h.md"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.holdout_eval",
            "--policies",
            "random",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--seed",
            "42",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert out_json.exists()
    assert out_md.exists()
    data = json.loads(out_json.read_text())
    assert data["seed"] == 42
    assert len(data["holdout"]) == 15
    assert "summary" in data or "results" in data
    if "summary" in data:
        rand = data["summary"]["random"]
        assert 0.0 <= rand["mean_reward"] <= 1.0
        assert {2, 3, 4}.issubset({int(k) for k in rand["per_tier"]})
    md = out_md.read_text()
    assert "Sealed Evaluation Set" in md
    assert "n=15" in md
