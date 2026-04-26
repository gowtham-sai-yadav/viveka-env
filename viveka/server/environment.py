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
# ── Modular orchestration layers (refactored 2026-04-26) ────────────────────
# Memory orchestration (Theme 2: Long-Horizon Planning + Instruction Following)
# lives in long_horizon_memory.py. Reward boundary stabilization lives in
# reward_stabilization.py. The env composes them; their internals are
# auditable in isolation.
from viveka.server.long_horizon_memory import (
    LAST_REASONING_MAX,
    LOOP_DETECT_K,
    RECENT_ACTIONS_K,
    compute_state_diff,
    detect_loop,
    extract_goal_entities,
    extract_last_reasoning,
    format_recent_actions_lines,
)
from viveka.server.reversibility_registry import lookup
from viveka.server.reward_stabilization import logit_clip_reward
from viveka.server.rubric import VivekaRubric
from viveka.server.safety_signals import extract_safety_concerns
from viveka.server.scenario_loader import load_scenario_by_tier
from viveka.server.services._base import MockService, ServiceError
from viveka.server.services.banking import BankingService
from viveka.server.services.digilocker import DigiLockerService
from viveka.server.services.irctc import IrctcService
from viveka.server.services.telecom import TelecomService
from viveka.server.services.upi import UpiService

ALL_SERVICES = ["upi", "digilocker", "irctc", "banking", "telecom"]
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
            "banking": BankingService(),
            "telecom": TelecomService(),
        }
        self._actions_taken: list[dict[str, Any]] = []
        self._pending_confirmations: list[PendingConfirmation] = []
        self._user_responses: list[dict[str, Any]] = []
        self._last_action_result: dict[str, Any] | None = None
        self._done_action_type: str | None = None
        # Memory-orchestration state (populated in reset / consumed in
        # _make_observation). _prev_visible_state=None is the sentinel for
        # "first observation, no diff baseline yet".
        self._prev_visible_state: dict[str, Any] | None = None
        self._goal_entities: list[str] = []

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
        # Reset memory-orchestration state. Goal entities are computed once
        # from initial_state — they're sticky across the whole episode.
        self._prev_visible_state = None
        try:
            self._goal_entities = extract_goal_entities(
                self._scenario.get("initial_state", {})
            )
        except Exception:
            self._goal_entities = []

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
            # Made-up op name. Emit a structured error_code so the grader's
            # hallucination component can detect it. Without this, the agent
            # could invent op names (e.g. `get_document` instead of
            # `view_document`) and slip past the hallucination penalty —
            # observed in gpt-5.2 baseline 2026-04-26.
            svc_upper = (action.target_service or "ENV").upper()
            return (
                f"Unknown operation: {action.target_service}.{action.operation}",
                {
                    "error_code": f"{svc_upper}:UNKNOWN_OP",
                    "error_message": f"Unknown operation '{action.operation}' on '{action.target_service}'",
                },
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
        # Validate the op exists in the registry. Mirrors _dispatch_execute's
        # check (added Fix 1a, 2026-04-26). Without this, an agent could
        # spam confirm_with_user(fake_op) — silently accepted, no rev/conf
        # credit (lookup-skip in _brier_means) but ALSO no hallucination
        # penalty (only executes were checked) — net positive score.
        try:
            lookup(action.target_service, action.operation)
        except KeyError:
            svc_upper = action.target_service.upper()
            return (
                f"Unknown operation: {action.target_service}.{action.operation}",
                {
                    "error_code": f"{svc_upper}:UNKNOWN_OP",
                    "error_message": f"Unknown operation '{action.operation}' on '{action.target_service}'",
                },
            )
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

    def _check_expected_state(self) -> dict[str, Any]:
        """Compare scenario's expected.post_state against current service snapshots."""
        expected = self._scenario.get("expected", {}).get("post_state", {}) or {}
        current = self._snapshot_services()
        details: dict[str, Any] = {}
        matched = True
        for svc_name, svc_expected in expected.items():
            svc_current = current.get(svc_name)
            if svc_current is None:
                details[svc_name] = {"_missing": True}
                matched = False
                continue
            field_results: dict[str, bool] = {}
            for field, exp_value in svc_expected.items():
                cur_value = svc_current.get(field)
                ok = _values_match(exp_value, cur_value)
                field_results[field] = ok
                if not ok:
                    matched = False
            details[svc_name] = field_results
        return {"matched": matched, "details": details}

    def _compute_intermediate_reward(self) -> float:
        # Pass services_state so respond_to_user Brier is tied to actual
        # task_completion (per graders.py:318-321). Without it, intermediate
        # signals diverge from the final episode reward, causing GRPO to learn
        # from misaligned per-step credit. Audit 2026-04-26 (env + graders).
        signals = compute_step_reward_signals(
            scenario=self._scenario,
            actions_taken=self._actions_taken,
            services_state=self._snapshot_services(),
        )
        if not signals:
            return 0.0
        avg = sum(signals.values()) / len(signals)
        return round(min(max(avg, 0.0), 1.0), 4)

    def _compute_final_reward(self) -> float:
        # Raw grader output ∈ [0, 1]. We pass it through logit_clip_reward
        # (viveka.server.reward_stabilization) to map the closed interval to
        # the open (REWARD_OPEN_LO, REWARD_OPEN_HI) interval — required for
        # proper-scoring-rule-driven GRPO training to avoid log(0) gradient
        # explosion. The grader itself is unchanged; the wrapper is a thin
        # post-hoc numerical-stability layer.
        raw = grade_episode(
            scenario=self._scenario,
            actions_taken=self._actions_taken,
            services_state=self._snapshot_services(),
            user_responses=self._user_responses,
            pending_confirmations=[pc.model_dump() for pc in self._pending_confirmations],
            done_action_type=self._done_action_type,
        )
        return logit_clip_reward(raw)

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, str) and len(v) > MAX_STRING_LEN:
                sanitized[k] = v[:MAX_STRING_LEN]
            else:
                sanitized[k] = v
        return sanitized

    # ─────────────────────────────────────────────────────────────────────
    # τ-bench-style information gating (Yao et al. 2024, arXiv:2406.12045)
    # ─────────────────────────────────────────────────────────────────────
    # The agent's `visible_state` is a REDACTED metadata-only view of the
    # services. Sensitive payloads are stripped so the agent must use the
    # service operations (view_document, check_balance, list_transactions)
    # to retrieve canonical data — exactly mirroring τ-bench's contract:
    #
    #   "The contents of the database form the state s_db, which is hidden
    #    from the agent and the user, and can only be read from or written
    #    to using API actions a_db."  (τ-bench Sec 3)
    #
    # WHY this is in the env (not just the prompt):
    # 1. Forces tool use — agent cannot synthesize "Your Aadhaar number
    #    is X" by reading visible_state.documents[0].data; that field is
    #    stripped. The only path is execute(view_document, doc_id).
    # 2. Closes the calibration loophole — if the agent could see all the
    #    data it might claim to know with full confidence, but Brier on
    #    `predicted_reversibility` requires actually committing to a
    #    correct label tied to a real registry op.
    # 3. Compatible with the grader — `_snapshot_services()` (un-redacted)
    #    is still used for `task_completion` checks. Only the AGENT's view
    #    is redacted; the grader sees ground truth.
    #
    # WHAT'S HIDDEN (must use a tool to access):
    # - DigiLocker: documents[*].data — the actual Aadhaar number, name,
    #   PAN string, DL number. Agent sees doc_id/doc_type/issuer only.
    # - UPI:        transactions[*]   — amounts, payee_vpa, timestamps.
    #   Agent sees just `transactions_count`. To inspect, list_transactions.
    # - IRCTC:      bookings[*].passengers — passenger names/ages/phones.
    #   Agent sees pnr/train_no/status only.
    #
    # WHAT'S VISIBLE (no tool call needed):
    # - upi.balance, upi.payer_vpa  — these are UI-level info, not PII.
    # - dgl.consents               — token metadata (audience, status, ttl)
    #                                is policy-relevant, not sensitive.
    # - dgl.shared                 — historical share log, low-PII.
    # - irctc.catalogue, availability, now_iso — public-info equivalents.
    # - upi.mandates, upi.cards    — metadata only (status/merchant).
    # - upi.disputes               — metadata.
    #
    # τ-bench reference: scenario inputs reveal *intent* not *IDs*. Agent
    # must perform a chain of authenticate → lookup → mutate. Our env
    # mirrors this by hiding payload fields behind tool calls.

    def _redacted_visible_state(self) -> dict[str, Any]:
        """Return a metadata-only snapshot for the agent's observation.

        Sensitive payloads (document contents, transaction details,
        passenger info) are stripped. The agent must call the corresponding
        read tool to retrieve them. Mirrors τ-bench (Yao 2024) Sec 3.
        """
        snap = self._snapshot_services()

        # ── DigiLocker: strip documents[*].data ─────────────────────
        # Agent sees the LIST of documents (so it knows what exists) but
        # not the contents. To read the Aadhaar number / PAN string,
        # agent must execute(view_document, doc_id=<id from list>).
        dgl = snap.get("digilocker", {})
        if "documents" in dgl:
            dgl["documents"] = [
                {k: v for k, v in doc.items() if k != "data"}
                for doc in dgl["documents"]
            ]

        # ── UPI: strip transactions ────────────────────────────────
        # Agent sees how MANY transactions exist, not the contents.
        # To list, agent must execute(list_transactions). This forces
        # the canonical "list-then-decide" pattern from τ-bench.
        upi = snap.get("upi", {})
        if "transactions" in upi:
            upi["transactions_count"] = len(upi.get("transactions", []))
            upi["transactions"] = []  # tool-only

        # ── IRCTC: strip booking passenger details ─────────────────
        # Agent sees the booking shell (PNR, train, status) but not
        # passenger personal info. To get details, execute(check_pnr).
        irctc = snap.get("irctc", {})
        if "bookings" in irctc:
            irctc["bookings"] = [
                {k: v for k, v in b.items() if k not in ("passengers", "passenger_details")}
                for b in irctc.get("bookings", [])
            ]

        return snap

    def _make_observation(
        self,
        message: str = "",
        done: bool = False,
        reward: float | None = None,
    ) -> VivekaObservation:
        # Pass services_state so observation.metadata["reward_signals"] reflects
        # the same state-aware view the reward grader uses. Without it,
        # viveka.task_progress (and any other state-diff-dependent signal)
        # silently reads as 0.0 in the obs metadata, breaking per-step
        # dashboards / plots / debugging — even though obs.reward (scalar)
        # is correct because env.step() passes _compute_intermediate_reward()
        # separately. Mirrors the d597b32 fix on _compute_intermediate_reward.
        signals = compute_step_reward_signals(
            scenario=self._scenario,
            actions_taken=self._actions_taken,
            services_state=self._snapshot_services(),
        )
        last_reply = self._user_responses[-1]["reply"] if self._user_responses else None

        # ── Memory orchestration (added 2026-04-26) ────────────────────────
        # Build the agent-visible memory channel from data the env already
        # tracks: action history (within-episode self-improvement), prior
        # reasoning (cognitive continuity), state diff (what just changed),
        # and a sticky goal anchor (long-horizon goal preservation in a
        # small context window). All fields land in `metadata` — never alter
        # the scenario JSON, never invent new content via an external LLM.
        visible_state = self._redacted_visible_state()
        try:
            recent_actions = format_recent_actions_lines(
                self._actions_taken, k=RECENT_ACTIONS_K
            )
        except Exception:
            recent_actions = []
        try:
            loop_detected, loop_warning = detect_loop(
                self._actions_taken, k=LOOP_DETECT_K
            )
        except Exception:
            loop_detected, loop_warning = False, None
        last_reasoning = extract_last_reasoning(
            self._actions_taken, max_len=LAST_REASONING_MAX
        )
        try:
            state_diff = compute_state_diff(self._prev_visible_state, visible_state)
        except Exception:
            state_diff = {}
        # Production-grade safety signals — derived deterministically from
        # visible_state + pending confirmations. Mirrors what a real DPI
        # platform (DigiLocker SDK, IRCTC API, UPI risk engine) would surface
        # to any agent integrating with it. Empty for scenarios that don't
        # trigger any rule (most T1/T2). Module: server/safety_signals.py
        try:
            safety_concerns = extract_safety_concerns(
                visible_state,
                list(self._pending_confirmations),
                user_message=self._state.user_message,
            )
        except Exception:
            safety_concerns = []

        obs = VivekaObservation(
            episode_id=self._state.episode_id or "",
            step=self._state.step_count,
            user_message=self._state.user_message,
            user_language=self._state.user_language,  # type: ignore[arg-type]
            available_services=ALL_SERVICES,  # type: ignore[arg-type]
            last_action_result=self._last_action_result,
            # NOTE: agent receives the REDACTED view, not the full snapshot.
            # The full snapshot (`_snapshot_services()`) is used only by the
            # grader for `task_completion` state-matching. This split is the
            # core of the τ-bench information-gating pattern.
            visible_state=visible_state,
            pending_confirmations=list(self._pending_confirmations),
            user_response=last_reply,
            message=message,
            done=done,
            reward=reward,
            metadata={
                "step_count": self._state.step_count,
                "scenario_id": self._state.scenario_id,
                "reward_signals": signals,
                # Memory channel — consumed by prompts.build_user_prompt.
                "goal_entities": list(self._goal_entities),
                "recent_actions": recent_actions,
                "loop_detected": loop_detected,
                "loop_warning": loop_warning,
                "last_reasoning": last_reasoning,
                "state_diff": state_diff,
                # Production-grade safety / business-rule warnings (T4 anchor).
                "safety_concerns": safety_concerns,
            },
        )
        # Update diff baseline AFTER constructing the obs. The next call sees
        # this state as "prev" and emits diff against it.
        self._prev_visible_state = visible_state
        return obs


def _values_match(expected: Any, current: Any) -> bool:
    if isinstance(expected, bool) or isinstance(current, bool):
        return expected == current
    if isinstance(expected, (int, float)) and isinstance(current, (int, float)):
        return abs(float(expected) - float(current)) <= 0.01
    return expected == current


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
