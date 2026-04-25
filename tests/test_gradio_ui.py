"""Tests for the Gradio UI policies and app entry point."""

from __future__ import annotations

import gradio as gr

from viveka.models import VivekaAction
from viveka.server.environment import VivekaEnvironment
from viveka.server.gradio_ui import (
    _heuristic_policy,
    _naive_policy,
    create_gradio_app,
)
from viveka.server.scenario_loader import load_scenario_by_tier


def test_create_gradio_app_returns_blocks():
    app = create_gradio_app()
    assert app is not None
    assert isinstance(app, gr.Blocks)


def test_naive_policy_produces_valid_action_for_scenario_001():
    scenario = load_scenario_by_tier(1, 0)
    env = VivekaEnvironment()
    obs = env.reset(tier_id=1, scenario_idx=0)
    action = _naive_policy(scenario, obs)
    assert isinstance(action, VivekaAction)
    assert action.confidence == 0.5
    assert action.predicted_reversibility is None
    assert action.action_type in {"execute", "respond_to_user"}


def test_heuristic_policy_confirms_before_share_pan_irreversible():
    scenario = load_scenario_by_tier(1, 1)
    assert scenario["scenario_id"] == "scenario_002_dgl_share_pan_to_bank"
    env = VivekaEnvironment()
    obs = env.reset(tier_id=1, scenario_idx=1)

    first = _heuristic_policy(scenario, obs, history=[])
    assert first.action_type == "confirm_with_user"
    assert first.target_service == "digilocker"
    assert first.operation == "share_document"
    assert first.predicted_reversibility == "irreversible"
    assert first.confidence == 0.9

    history_after_confirm = [{
        "action_type": "confirm_with_user",
        "target_service": "digilocker",
        "operation": "share_document",
    }]
    second = _heuristic_policy(scenario, obs, history=history_after_confirm)
    assert second.action_type == "execute"
    assert second.target_service == "digilocker"
    assert second.operation == "share_document"
    assert second.predicted_reversibility == "irreversible"
