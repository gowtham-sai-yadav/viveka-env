"""Tests for VivekaEnvironment._check_expected_state."""

from __future__ import annotations

from viveka.server.environment import VivekaEnvironment


def _fresh_env() -> VivekaEnvironment:
    env = VivekaEnvironment()
    env.reset(tier_id=1, scenario_idx=0)
    return env


def test_empty_post_state_matches():
    env = _fresh_env()
    env._scenario = {"expected": {"post_state": {}}}
    result = env._check_expected_state()
    assert result["matched"] is True
    assert result["details"] == {}


def test_matching_balance():
    env = _fresh_env()
    env._services["upi"]._balance = 12500.0
    env._scenario = {"expected": {"post_state": {"upi": {"balance": 12500.0}}}}
    result = env._check_expected_state()
    assert result["matched"] is True
    assert result["details"]["upi"]["balance"] is True


def test_mismatched_balance():
    env = _fresh_env()
    env._services["upi"]._balance = 11000.0
    env._scenario = {"expected": {"post_state": {"upi": {"balance": 12500.0}}}}
    result = env._check_expected_state()
    assert result["matched"] is False
    assert result["details"]["upi"]["balance"] is False


def test_balance_within_tolerance():
    env = _fresh_env()
    env._services["upi"]._balance = 12500.005
    env._scenario = {"expected": {"post_state": {"upi": {"balance": 12500.00}}}}
    result = env._check_expected_state()
    assert result["matched"] is True
    assert result["details"]["upi"]["balance"] is True


def test_asserted_service_with_no_state_marks_missing():
    env = _fresh_env()
    env._services.pop("digilocker")
    env._scenario = {
        "expected": {"post_state": {"digilocker": {"documents": []}}}
    }
    result = env._check_expected_state()
    assert result["matched"] is False
    assert result["details"]["digilocker"] == {"_missing": True}
