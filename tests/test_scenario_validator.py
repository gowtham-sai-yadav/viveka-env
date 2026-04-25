"""Tests for scenario JSON schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from viveka.server.scenario_loader import ScenarioSchema, validate_scenario


def _valid_raw() -> dict:
    return {
        "scenario_id": "scenario_001_upi_balance",
        "tier_id": 1,
        "title": "UPI balance check",
        "user_message": "What is my balance?",
        "user_language": "en",
        "initial_state": {"upi": {"balance": 12500.0}},
        "user_oracle": {},
        "expected": {
            "ground_truth_action_sequence": [
                {
                    "target_service": "upi",
                    "operation": "check_balance",
                    "reversibility": "reversible",
                },
            ],
            "post_state": {"upi": {"balance": 12500.0}},
            "max_steps": 5,
            "must_ask": False,
            "must_not_execute": [],
        },
    }


def test_valid_scenario_passes():
    parsed = validate_scenario(_valid_raw())
    assert isinstance(parsed, ScenarioSchema)
    assert parsed.scenario_id == "scenario_001_upi_balance"
    assert parsed.tier_id == 1
    assert parsed.expected.max_steps == 5


def test_missing_expected_field_raises():
    raw = _valid_raw()
    del raw["expected"]
    with pytest.raises(ValidationError):
        validate_scenario(raw)


def test_invalid_tier_id_raises():
    raw = _valid_raw()
    raw["tier_id"] = 99
    with pytest.raises(ValidationError):
        validate_scenario(raw)


def test_reversibility_mismatch_raises_clear_value_error():
    raw = _valid_raw()
    raw["expected"]["ground_truth_action_sequence"][0]["reversibility"] = "irreversible"
    with pytest.raises(ValueError) as exc_info:
        validate_scenario(raw)
    msg = str(exc_info.value)
    assert "scenario_001_upi_balance" in msg
    assert "upi.check_balance" in msg
    assert "irreversible" in msg
    assert "reversible" in msg


def test_extra_top_level_field_forbidden():
    raw = _valid_raw()
    raw["unexpected_field"] = "boom"
    with pytest.raises(ValidationError):
        validate_scenario(raw)


def test_must_not_execute_accepts_two_list():
    raw = _valid_raw()
    raw["expected"]["must_not_execute"] = [["upi", "send_money"]]
    parsed = validate_scenario(raw)
    assert parsed.expected.must_not_execute == [["upi", "send_money"]]
