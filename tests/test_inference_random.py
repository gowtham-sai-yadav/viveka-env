"""Smoke tests for inference.py random policy. No GPU, no API key required."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_random_policy_runs_episode_directly():
    sys.path.insert(0, str(REPO))
    try:
        from inference import RandomPolicy, run_episode
        from viveka.server.environment import VivekaEnvironment
    finally:
        sys.path.pop(0)

    env = VivekaEnvironment()
    policy = RandomPolicy(seed=42)
    result = run_episode(env, policy, tier_id=1, scenario_idx=0)
    assert "scenario_id" in result
    assert isinstance(result["reward"], float)
    assert 0.0 <= result["reward"] <= 1.0
    assert result["length"] >= 1
    assert "viveka.reversibility_correct" in result["components"]


def test_inference_random_subprocess(tmp_path):
    out = tmp_path / "random.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "inference.py"),
            "--policy",
            "random",
            "--tier-mix",
            "1",
            "--max-scenarios",
            "2",
            "--output-json",
            str(out),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    data = json.loads(out.read_text())
    assert data["policy_name"] == "random"
    assert data["n_scenarios"] == 2
    assert 0.0 <= data["mean_reward"] <= 1.0
    assert len(data["scenarios"]) == 2
    for s in data["scenarios"]:
        assert "components" in s
        assert "length" in s


def test_random_policy_smart_distribution_executes_majority():
    sys.path.insert(0, str(REPO))
    try:
        from inference import RandomPolicy
        from viveka.models import VivekaObservation
    finally:
        sys.path.pop(0)

    policy = RandomPolicy(seed=0)
    obs = VivekaObservation(
        episode_id="x",
        step=0,
        user_message="test",
        user_language="en",
        available_services=["upi", "digilocker", "irctc"],
        last_action_result=None,
        visible_state={},
        pending_confirmations=[],
        user_response=None,
        message="",
        done=False,
    )
    counts: dict[str, int] = {}
    for _ in range(500):
        a = policy(obs)
        counts[a.action_type] = counts.get(a.action_type, 0) + 1

    assert counts.get("execute", 0) >= 200, f"expected execute majority, got {counts}"
    assert counts.get("respond_to_user", 0) <= 50, f"terminator should be rare: {counts}"
