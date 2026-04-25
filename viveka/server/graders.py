"""Reward graders. Phase-2 work — Debashis owns the full 6-component implementation.

The current stubs let the environment compile and step end-to-end with reward=0.0.
The signal names below are the contract that GRPO and the rubric expect.
"""

from __future__ import annotations

from typing import Any


REWARD_SIGNAL_NAMES = (
    "viveka.reversibility_correct",
    "viveka.confidence_brier",
    "viveka.confirmation_appropriate",
    "viveka.over_asking",
    "viveka.task_progress",
    "viveka.hallucination",
    "viveka.action_executed",
    "viveka.action_destructive",
)


def compute_step_reward_signals(
    scenario: dict[str, Any],
    actions_taken: list[dict[str, Any]],
    **state: Any,
) -> dict[str, float]:
    return {name: 0.0 for name in REWARD_SIGNAL_NAMES}


def grade_episode(
    scenario: dict[str, Any],
    actions_taken: list[dict[str, Any]],
    services_state: dict[str, Any] | None = None,
    user_responses: list[dict[str, Any]] | None = None,
    pending_confirmations: list[dict[str, Any]] | None = None,
    done_action_type: str | None = None,
) -> float:
    return 0.0
