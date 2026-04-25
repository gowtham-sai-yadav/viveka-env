"""Smoke tests — env imports, resets, and steps without scenarios."""

from __future__ import annotations

from viveka.models import VivekaAction
from viveka.server.environment import VivekaEnvironment
from viveka.server.reversibility_registry import lookup


def test_imports():
    assert VivekaEnvironment is not None


def test_registry_known_ops():
    assert lookup("upi", "send_money") == "irreversible"
    assert lookup("upi", "check_balance") == "reversible"
    assert lookup("digilocker", "share_document") == "irreversible"
    assert lookup("irctc", "book_ticket") == "irreversible"


def test_reset_no_scenarios_returns_empty_stub():
    env = VivekaEnvironment()
    obs = env.reset(tier_id=1, scenario_idx=0)
    assert obs.step == 0
    assert obs.user_message.startswith("(no scenario loaded")
    assert "upi" in obs.available_services


def test_step_check_balance_reversible():
    env = VivekaEnvironment()
    env.reset(tier_id=1, scenario_idx=0)
    obs = env.step(VivekaAction(
        action_type="execute",
        target_service="upi",
        operation="check_balance",
        params={},
        predicted_reversibility="reversible",
        confidence=0.9,
        reasoning="balance lookup is read-only",
    ))
    assert obs.step == 1
    assert "balance" in (obs.last_action_result or {})


def test_step_respond_to_user_terminates():
    env = VivekaEnvironment()
    env.reset(tier_id=1, scenario_idx=0)
    obs = env.step(VivekaAction(
        action_type="respond_to_user",
        params={"text": "Done."},
        confidence=0.9,
        reasoning="task complete",
    ))
    assert obs.done is True
