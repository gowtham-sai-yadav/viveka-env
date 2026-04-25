"""VivekaEnvironment — core OpenEnv environment for reversibility + calibration."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from viveka.models import (
    PendingConfirmation,
    VivekaAction,
    VivekaObservation,
    VivekaState,
)
from viveka.server.graders import compute_step_reward_signals, grade_episode
from viveka.server.reversibility_registry import lookup
from viveka.server.rubric import VivekaRubric
from viveka.server.scenario_loader import load_scenario_by_tier
from viveka.server.services._base import MockService, ServiceError
from viveka.server.services.digilocker import DigiLockerService
from viveka.server.services.irctc import IrctcService
from viveka.server.services.upi import UpiService

ALL_SERVICES = ["upi", "digilocker", "irctc"]
MAX_STEPS = 30
MAX_STRING_LEN = 5000


class VivekaEnvironment(Environment[VivekaAction, VivekaObservation, VivekaState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any) -> None:
        self._rubric = VivekaRubric()
        self._rubric.set_env(self)
        super().__init__(rubric=self._rubric)
        self._state = VivekaState()
        self._scenario: dict[str, Any] = {}
        self._services: dict[str, MockService] = {
            "upi": UpiService(),
            "digilocker": DigiLockerService(),
            "irctc": IrctcService(),
        }
        self._actions_taken: list[dict[str, Any]] = []
        self._pending_confirmations: list[PendingConfirmation] = []
        self._user_responses: list[dict[str, Any]] = []
        self._last_action_result: dict[str, Any] | None = None
        self._done_action_type: str | None = None

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> VivekaObservation:
        self._reset_rubric()

        tier_id = int(kwargs.get("tier_id", 1))
        scenario_idx = int(kwargs.get("scenario_idx", 0))

        try:
            self._scenario = load_scenario_by_tier(tier_id, scenario_idx)
        except (FileNotFoundError, IndexError):
            self._scenario = _empty_scenario(tier_id, scenario_idx)

        for svc_name, svc in self._services.items():
            initial = self._scenario.get("initial_state", {}).get(svc_name, {})
            svc.reset(initial)

        self._actions_taken = []
        self._pending_confirmations = []
        self._user_responses = []
        self._last_action_result = None
        self._done_action_type = None

        eid = episode_id or str(uuid4())
        self._state = VivekaState(
            episode_id=eid,
            step_count=0,
            scenario_id=self._scenario.get("scenario_id", "unknown"),
            user_message=self._scenario.get("user_message", ""),
            user_language=self._scenario.get("user_language", "en"),
        )
        return self._make_observation(message=self._scenario.get("user_message", ""))

    # ── step ──────────────────────────────────────────────────────────────

    def step(
        self,
        action: VivekaAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> VivekaObservation:
        if self._is_terminal() or self._state.step_count >= MAX_STEPS:
            reward = self._compute_final_reward()
            return self._make_observation(
                message="Episode already terminated or step limit reached.",
                done=True,
                reward=reward,
            )

        self._state.step_count += 1
        params = self._sanitize_params(action.params)
        record: dict[str, Any] = {
            "step": self._state.step_count,
            "action_type": action.action_type,
            "target_service": action.target_service,
            "operation": action.operation,
            "params": params,
            "predicted_reversibility": action.predicted_reversibility,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
        }

        try:
            msg, result = self._dispatch(action, params)
        except ServiceError as e:
            msg = f"Service error: {e}"
            result = {"error_code": e.code, "error_message": e.message}
        except Exception as e:  # noqa: BLE001 — surface dispatch failures into observation
            msg = f"Internal error: {e}"
            result = {"error": str(e)}

        record["result"] = result
        self._actions_taken.append(record)
        self._state.actions_taken = self._actions_taken
        self._last_action_result = result

        done = self._is_terminal() or self._state.step_count >= MAX_STEPS
        reward = self._compute_final_reward() if done else self._compute_intermediate_reward()
        obs = self._make_observation(message=msg, done=done, reward=reward)

        self._apply_rubric(action, obs)
        return obs

    @property
    def state(self) -> VivekaState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="viveka_env",
            description=(
                "Viveka — predict reversibility, emit calibrated confidence, ask the user when "
                "uncertain. Mocked Indian DPI services (UPI + DigiLocker + IRCTC)."
            ),
            version="0.1.0",
        )

    # ── dispatch ──────────────────────────────────────────────────────────

    def _dispatch(self, action: VivekaAction, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if action.action_type == "execute":
            return self._dispatch_execute(action, params)
        if action.action_type == "confirm_with_user":
            return self._dispatch_confirm(action, params)
        if action.action_type == "ask_user":
            return self._dispatch_ask(action, params)
        if action.action_type == "abstain":
            return "Abstained.", {"abstained": True}
        if action.action_type == "respond_to_user":
            return self._dispatch_respond(action, params)
        return f"Unknown action_type: {action.action_type}", {"error": "unknown action_type"}

    def _dispatch_execute(
        self, action: VivekaAction, params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        if not action.target_service or not action.operation:
            return "execute requires target_service and operation.", {"error": "missing fields"}
        if action.predicted_reversibility is None:
            return "execute requires predicted_reversibility.", {"error": "missing reversibility"}
        try:
            ground_truth = lookup(action.target_service, action.operation)
        except KeyError:
            return (
                f"Unknown operation: {action.target_service}.{action.operation}",
                {"error": "unknown operation"},
            )
        svc = self._services[action.target_service]
        result = svc.execute(action.operation, params)
        result["ground_truth_reversibility"] = ground_truth
        return f"Executed {action.target_service}.{action.operation}.", result

    def _dispatch_confirm(
        self, action: VivekaAction, params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        if not action.target_service or not action.operation:
            return "confirm_with_user requires target_service and operation.", {
                "error": "missing fields"
            }
        rendered = self._render_action(action.target_service, action.operation, params)
        user_reply = self._simulate_user_reply(action.target_service, action.operation, params)
        pc = PendingConfirmation(
            action_id=str(uuid4()),
            target_service=action.target_service,
            operation=action.operation,
            params=params,
            rendered=rendered,
            user_reply=user_reply,
        )
        self._pending_confirmations.append(pc)
        self._user_responses.append({
            "step": self._state.step_count,
            "kind": "confirm",
            "rendered": rendered,
            "reply": user_reply,
        })
        return f"Asked user to confirm: {rendered}", {
            "pending": pc.model_dump(),
            "user_reply": user_reply,
        }

    def _dispatch_ask(
        self, action: VivekaAction, params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        question = str(params.get("question", "")).strip()
        user_reply = self._simulate_user_reply_to_question(question)
        self._user_responses.append({
            "step": self._state.step_count,
            "kind": "ask",
            "question": question,
            "reply": user_reply,
        })
        return f"Asked user: {question}", {"user_reply": user_reply}

    def _dispatch_respond(
        self, action: VivekaAction, params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        text = str(params.get("text", "")).strip()
        self._done_action_type = "respond_to_user"
        self._state.task_complete = True
        return "Final response sent to user.", {"response": text}

    # ── helpers ───────────────────────────────────────────────────────────

    def _is_terminal(self) -> bool:
        return self._done_action_type == "respond_to_user" or self._state.task_complete

    def _render_action(self, service: str, operation: str, params: dict[str, Any]) -> str:
        return f"{service}.{operation}({params})"

    def _simulate_user_reply(
        self, service: str, operation: str, params: dict[str, Any]
    ) -> str:
        oracle = self._scenario.get("user_oracle", {})
        return oracle.get(f"confirm:{service}.{operation}", "yes")

    def _simulate_user_reply_to_question(self, question: str) -> str:
        oracle = self._scenario.get("user_oracle", {})
        return oracle.get(f"ask:{question}", oracle.get("ask:default", ""))

    def _snapshot_services(self) -> dict[str, dict[str, Any]]:
        return {name: svc.state() for name, svc in self._services.items()}

    def _compute_intermediate_reward(self) -> float:
        signals = compute_step_reward_signals(
            scenario=self._scenario,
            actions_taken=self._actions_taken,
        )
        if not signals:
            return 0.0
        avg = sum(signals.values()) / len(signals)
        return round(min(max(avg, 0.0), 1.0), 4)

    def _compute_final_reward(self) -> float:
        return grade_episode(
            scenario=self._scenario,
            actions_taken=self._actions_taken,
            services_state=self._snapshot_services(),
            user_responses=self._user_responses,
            pending_confirmations=[pc.model_dump() for pc in self._pending_confirmations],
            done_action_type=self._done_action_type,
        )

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, str) and len(v) > MAX_STRING_LEN:
                sanitized[k] = v[:MAX_STRING_LEN]
            else:
                sanitized[k] = v
        return sanitized

    def _make_observation(
        self,
        message: str = "",
        done: bool = False,
        reward: float | None = None,
    ) -> VivekaObservation:
        signals = compute_step_reward_signals(
            scenario=self._scenario,
            actions_taken=self._actions_taken,
        )
        last_reply = self._user_responses[-1]["reply"] if self._user_responses else None
        return VivekaObservation(
            episode_id=self._state.episode_id or "",
            step=self._state.step_count,
            user_message=self._state.user_message,
            user_language=self._state.user_language,  # type: ignore[arg-type]
            available_services=ALL_SERVICES,  # type: ignore[arg-type]
            last_action_result=self._last_action_result,
            visible_state=self._snapshot_services(),
            pending_confirmations=list(self._pending_confirmations),
            user_response=last_reply,
            message=message,
            done=done,
            reward=reward,
            metadata={
                "step_count": self._state.step_count,
                "scenario_id": self._state.scenario_id,
                "reward_signals": signals,
            },
        )


def _empty_scenario(tier_id: int, scenario_idx: int) -> dict[str, Any]:
    return {
        "scenario_id": f"empty_t{tier_id}_{scenario_idx}",
        "tier_id": tier_id,
        "user_message": "(no scenario loaded — empty stub)",
        "user_language": "en",
        "initial_state": {},
        "user_oracle": {},
        "expected": {},
    }
