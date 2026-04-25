"""HTTP client for the Viveka environment server."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from viveka.models import VivekaAction, VivekaObservation, VivekaState


class VivekaClient(EnvClient[VivekaAction, VivekaObservation, VivekaState]):
    def _step_payload(self, action: VivekaAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[VivekaObservation]:
        obs_data = payload.get("observation", payload)
        reward = payload.get("reward")
        done = payload.get("done", False)
        obs_data_clean = {k: v for k, v in obs_data.items() if k not in ("reward", "done")}
        obs = VivekaObservation(**obs_data_clean, reward=reward, done=done)
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: dict[str, Any]) -> VivekaState:
        return VivekaState(**payload)
