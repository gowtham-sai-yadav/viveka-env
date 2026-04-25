"""Pydantic schemas for Viveka actions, observations, and state."""

from __future__ import annotations

from typing import Any, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


SERVICE = Literal["upi", "digilocker", "irctc"]
REVERSIBILITY = Literal["reversible", "irreversible", "irreversible_trivial"]
ACTION_TYPE = Literal[
    "execute",
    "confirm_with_user",
    "ask_user",
    "abstain",
    "respond_to_user",
]


class PendingConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    target_service: SERVICE
    operation: str
    params: dict[str, Any]
    rendered: str
    user_reply: Optional[str] = None


class VivekaAction(Action):
    """Single-class action with discriminator semantics enforced at the env layer.

    - `execute`: requires target_service, operation, predicted_reversibility.
    - `confirm_with_user`: same fields; awaits user "yes/no".
    - `ask_user`: requires params["question"].
    - `abstain`: no-op step (records caution).
    - `respond_to_user`: terminal action; requires params["text"].
    """

    model_config = ConfigDict(extra="forbid")

    action_type: ACTION_TYPE
    target_service: Optional[SERVICE] = None
    operation: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    predicted_reversibility: Optional[REVERSIBILITY] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class VivekaObservation(Observation):
    episode_id: str = Field(default="")
    step: int = Field(default=0)
    user_message: str = Field(default="")
    user_language: Literal["en", "hi-en"] = Field(default="en")
    available_services: list[SERVICE] = Field(default_factory=list)
    last_action_result: Optional[dict[str, Any]] = Field(default=None)
    visible_state: dict[str, Any] = Field(default_factory=dict)
    pending_confirmations: list[PendingConfirmation] = Field(default_factory=list)
    user_response: Optional[str] = Field(default=None)
    message: str = Field(default="")


class VivekaState(State):
    scenario_id: str = Field(default="")
    user_message: str = Field(default="")
    user_language: str = Field(default="en")
    task_complete: bool = Field(default=False)
    actions_taken: list[dict[str, Any]] = Field(default_factory=list)
