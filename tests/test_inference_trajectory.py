"""Verify inference.py exposes per-action trajectory data with correctness labels."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_run_episode_includes_trajectory():
    sys.path.insert(0, str(REPO))
    try:
        from inference import RandomPolicy, run_episode
        from viveka.server.environment import VivekaEnvironment
    finally:
        sys.path.pop(0)

    env = VivekaEnvironment()
    result = run_episode(env, RandomPolicy(seed=7), tier_id=1, scenario_idx=0)
    assert "trajectory" in result
    assert isinstance(result["trajectory"], list)
    assert len(result["trajectory"]) >= 1
    rec = result["trajectory"][0]
    expected_keys = {
        "step",
        "action_type",
        "target_service",
        "operation",
        "predicted_reversibility",
        "confidence",
        "result_error_code",
        "ground_truth_reversibility",
        "correctness",
    }
    assert expected_keys.issubset(rec.keys()), f"missing: {expected_keys - rec.keys()}"


def test_trajectory_correctness_matches_registry():
    sys.path.insert(0, str(REPO))
    try:
        from inference import _extract_trajectory
    finally:
        sys.path.pop(0)

    actions = [
        {
            "step": 1,
            "action_type": "execute",
            "target_service": "upi",
            "operation": "check_balance",
            "predicted_reversibility": "reversible",
            "confidence": 0.9,
            "result": {},
        },
        {
            "step": 2,
            "action_type": "execute",
            "target_service": "upi",
            "operation": "send_money",
            "predicted_reversibility": "reversible",
            "confidence": 0.95,
            "result": {},
        },
        {
            "step": 3,
            "action_type": "ask_user",
            "target_service": None,
            "operation": None,
            "predicted_reversibility": None,
            "confidence": 0.5,
            "result": {},
        },
    ]
    traj = _extract_trajectory(actions)
    assert traj[0]["correctness"] == 1  # check_balance is reversible
    assert traj[1]["correctness"] == 0  # send_money is irreversible, predicted reversible
    assert traj[2]["correctness"] is None  # ask_user has no prediction


def test_trajectory_unknown_op_marked_none():
    sys.path.insert(0, str(REPO))
    try:
        from inference import _extract_trajectory
    finally:
        sys.path.pop(0)

    actions = [
        {
            "step": 1,
            "action_type": "execute",
            "target_service": "upi",
            "operation": "telepathy",
            "predicted_reversibility": "reversible",
            "confidence": 0.9,
            "result": {},
        }
    ]
    traj = _extract_trajectory(actions)
    assert traj[0]["correctness"] is None
    assert traj[0]["ground_truth_reversibility"] is None
