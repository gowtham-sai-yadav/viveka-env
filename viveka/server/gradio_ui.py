"""Live-trace Gradio UI for the Viveka environment."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from viveka.models import VivekaAction, VivekaObservation
from viveka.server.environment import VivekaEnvironment
from viveka.server.reversibility_registry import lookup as registry_lookup
from viveka.server.scenario_loader import all_tier_dirs, list_scenarios, load_scenario_by_tier


_TIER_TO_ID = {dir_name: tier_id for tier_id, dir_name in all_tier_dirs().items()}

_REVERSIBILITY_EMOJI = {
    "reversible": "🟢",
    "irreversible": "🔴",
    "irreversible_trivial": "🟡",
}

_HEADER_PITCH = (
    "# Viveka — Reversibility + Calibration RL\n\n"
    "Viveka teaches LLM agents to predict reversibility *before* acting, emit calibrated "
    "confidence, and ask the user when uncertain. Substrate: mocked Indian DPI "
    "(UPI + DigiLocker + IRCTC).\n"
)

_POLICY_HELP = (
    "**Naive** = always execute first action with confidence 0.5 (untrained baseline). "
    "**Heuristic** = confirm before irreversible, confidence 0.9. "
    "**Manual** = step through one action at a time using the JSON editor below."
)

_MANUAL_ACTION_TEMPLATE: dict[str, Any] = {
    "action_type": "execute",
    "target_service": "digilocker",
    "operation": "view_document",
    "params": {"doc_id": "AAD-1234"},
    "predicted_reversibility": "reversible",
    "confidence": 0.8,
    "reasoning": "View is read-only.",
}


def _scenario_options() -> list[str]:
    options: list[str] = []
    for tier_id in sorted(all_tier_dirs()):
        dir_name = all_tier_dirs()[tier_id]
        for path in list_scenarios(dir_name):
            options.append(f"{dir_name}/{path.stem}")
    return options


def _parse_scenario_choice(choice: str) -> tuple[int, int]:
    tier_dir, stem = choice.split("/", 1)
    tier_id = _TIER_TO_ID[tier_dir]
    scenarios = list_scenarios(tier_dir)
    for idx, path in enumerate(scenarios):
        if path.stem == stem:
            return tier_id, idx
    raise ValueError(f"Scenario not found: {choice}")


def _ground_truth_sequence(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return scenario.get("expected", {}).get("ground_truth_action_sequence", []) or []


def _safe_registry_label(service: str | None, operation: str | None) -> str | None:
    if not service or not operation:
        return None
    try:
        return registry_lookup(service, operation)
    except KeyError:
        return None


def _naive_policy(scenario: dict[str, Any], obs: VivekaObservation) -> VivekaAction:
    sequence = _ground_truth_sequence(scenario)
    step_idx = obs.step
    if step_idx >= len(sequence):
        return VivekaAction(
            action_type="respond_to_user",
            params={"text": "Done."},
            confidence=0.5,
            reasoning="Naive baseline: terminate after exhausting sequence.",
        )
    gt = sequence[step_idx]
    return VivekaAction(
        action_type="execute",
        target_service=gt["target_service"],
        operation=gt["operation"],
        params={},
        predicted_reversibility=None,
        confidence=0.5,
        reasoning="Naive baseline: execute without predicting reversibility.",
    )


def _heuristic_policy(
    scenario: dict[str, Any],
    obs: VivekaObservation,
    history: list[dict[str, Any]],
) -> VivekaAction:
    sequence = _ground_truth_sequence(scenario)
    executed_indices = sum(1 for h in history if h.get("action_type") == "execute")
    if executed_indices >= len(sequence):
        return VivekaAction(
            action_type="respond_to_user",
            params={"text": "Task complete."},
            confidence=0.9,
            reasoning="Heuristic: ground-truth sequence exhausted.",
        )
    gt = sequence[executed_indices]
    service = gt["target_service"]
    operation = gt["operation"]
    label = _safe_registry_label(service, operation)
    already_confirmed = any(
        h.get("action_type") == "confirm_with_user"
        and h.get("target_service") == service
        and h.get("operation") == operation
        for h in history
    )
    if label == "irreversible" and not already_confirmed:
        return VivekaAction(
            action_type="confirm_with_user",
            target_service=service,
            operation=operation,
            params={},
            predicted_reversibility="irreversible",
            confidence=0.9,
            reasoning="Heuristic: confirm before irreversible action.",
        )
    return VivekaAction(
        action_type="execute",
        target_service=service,
        operation=operation,
        params={},
        predicted_reversibility=label,
        confidence=0.9,
        reasoning="Heuristic: execute with registry-derived reversibility.",
    )


def _format_step_markdown(record: dict[str, Any], obs: VivekaObservation) -> str:
    step = record["step"]
    action_type = record.get("action_type", "?")
    service = record.get("target_service")
    operation = record.get("operation")
    predicted = record.get("predicted_reversibility")
    confidence = float(record.get("confidence", 0.0) or 0.0)
    reasoning = record.get("reasoning") or ""
    target_label = f"{service}.{operation}" if service and operation else "—"

    truth = _safe_registry_label(service, operation)
    if predicted is None:
        rev_marker = "—"
    elif truth is not None and predicted == truth:
        rev_marker = f"✅ {predicted}"
    else:
        rev_marker = f"❌ {predicted} (truth: {truth or 'unknown'})"

    emoji = _REVERSIBILITY_EMOJI.get(truth or "", "⚪")

    result = record.get("result") or {}
    if isinstance(result, dict):
        if "error_code" in result:
            outcome = f"error `{result['error_code']}`: {result.get('error_message', '')}"
        elif "error" in result:
            outcome = f"error: {result['error']}"
        elif result.get("abstained"):
            outcome = "abstained"
        elif "user_reply" in result:
            outcome = f"user replied: _{result['user_reply']}_"
        elif "response" in result:
            outcome = f"final response: _{result['response']}_"
        else:
            outcome = "ok"
    else:
        outcome = str(result)

    lines = [
        f"### Step {step} {emoji} **{action_type}** · `{target_label}`",
        f"- predicted_reversibility: {rev_marker}",
        f"- confidence: **{confidence:.1f}**",
        f"- result: {outcome}",
        f"- env message: {obs.message}",
    ]
    if reasoning:
        lines.append(f"- _{reasoning}_")
    return "\n".join(lines)


def _final_reward_table(obs: VivekaObservation) -> str:
    signals = (obs.metadata or {}).get("reward_signals", {}) or {}
    if not signals:
        return ""
    rows = ["| Signal | Value |", "|---|---|"]
    total = 0.0
    for name, val in signals.items():
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = 0.0
        total += v
        rows.append(f"| `{name}` | {v:.3f} |")
    final_reward = obs.reward if obs.reward is not None else 0.0
    rows.append(f"| **cumulative (mean of signals)** | **{total / max(len(signals), 1):.3f}** |")
    rows.append(f"| **episode reward** | **{final_reward:.3f}** |")
    return "## Final reward breakdown\n\n" + "\n".join(rows)


def _empty_outputs() -> tuple[str, str, dict, dict, str, str]:
    return "", "", {}, {}, "", ""


def _run_full(scenario_choice: str, policy: str) -> tuple[str, str, dict, dict, str, str]:
    if not scenario_choice:
        return _empty_outputs()
    try:
        tier_id, idx = _parse_scenario_choice(scenario_choice)
        scenario = load_scenario_by_tier(tier_id, idx)
    except Exception as e:
        return "", "", {}, {}, "", f"**Error:** {e}"

    env = VivekaEnvironment()
    obs = env.reset(tier_id=tier_id, scenario_idx=idx)
    user_msg = f"**User request** ({scenario.get('user_language', 'en')}): {obs.user_message}"

    if policy == "manual":
        return user_msg, "_Manual mode — use the **Step** button below._", obs.visible_state, (obs.metadata or {}).get("reward_signals", {}), "", ""

    max_steps = scenario.get("expected", {}).get("max_steps", 30)
    history: list[dict[str, Any]] = []
    trace_chunks: list[str] = []
    step_count = 0
    last_obs: VivekaObservation = obs
    error_md = ""

    while not last_obs.done and step_count < max_steps:
        try:
            if policy == "naive":
                action = _naive_policy(scenario, last_obs)
            else:
                action = _heuristic_policy(scenario, last_obs, history)
            last_obs = env.step(action)
        except Exception as e:
            error_md = f"**Error:** {e}"
            break
        step_count += 1
        record = (env._actions_taken or [{}])[-1]
        history.append(record)
        trace_chunks.append(_format_step_markdown(record, last_obs))

    trace_md = "\n\n---\n\n".join(trace_chunks) if trace_chunks else "_No steps executed._"
    final_md = _final_reward_table(last_obs) if last_obs.done else ""
    signals = (last_obs.metadata or {}).get("reward_signals", {}) or {}
    return user_msg, trace_md, last_obs.visible_state, signals, final_md, error_md


def _reset_scenario(scenario_choice: str) -> tuple[str, str, dict, dict, str, str, Any]:
    if not scenario_choice:
        return "", "", {}, {}, "", "", None
    try:
        tier_id, idx = _parse_scenario_choice(scenario_choice)
        scenario = load_scenario_by_tier(tier_id, idx)
    except Exception as e:
        return "", "", {}, {}, "", f"**Error:** {e}", None
    env = VivekaEnvironment()
    obs = env.reset(tier_id=tier_id, scenario_idx=idx)
    user_msg = f"**User request** ({scenario.get('user_language', 'en')}): {obs.user_message}"
    state_blob = {
        "tier_id": tier_id,
        "scenario_idx": idx,
        "scenario": scenario,
        "history": [],
        "done": False,
    }
    signals = (obs.metadata or {}).get("reward_signals", {}) or {}
    return user_msg, "_Environment reset. Press Run scenario or Step._", obs.visible_state, signals, "", "", state_blob


def _step_manual(
    scenario_choice: str, manual_action_json: Any, manual_state: Any
) -> tuple[str, dict, dict, str, str, Any]:
    if not scenario_choice:
        return "", {}, {}, "", "**Error:** Pick a scenario first.", manual_state
    try:
        tier_id, idx = _parse_scenario_choice(scenario_choice)
    except Exception as e:
        return "", {}, {}, "", f"**Error:** {e}", manual_state

    if manual_state is None or manual_state.get("done"):
        env = VivekaEnvironment()
        env.reset(tier_id=tier_id, scenario_idx=idx)
        scenario = load_scenario_by_tier(tier_id, idx)
        manual_state = {
            "tier_id": tier_id,
            "scenario_idx": idx,
            "scenario": scenario,
            "history": [],
            "done": False,
            "trace": "",
        }
    else:
        env = VivekaEnvironment()
        env.reset(tier_id=tier_id, scenario_idx=idx)
        for past in manual_state.get("history", []):
            try:
                env.step(VivekaAction(**past["_action_payload"]))
            except Exception:
                pass

    try:
        if isinstance(manual_action_json, str):
            payload = json.loads(manual_action_json)
        else:
            payload = dict(manual_action_json or {})
        action = VivekaAction(**payload)
        obs = env.step(action)
    except Exception as e:
        trace = manual_state.get("trace", "")
        return trace, {}, {}, "", f"**Error:** {e}", manual_state

    record = (env._actions_taken or [{}])[-1]
    record_with_payload = dict(record)
    record_with_payload["_action_payload"] = payload
    manual_state["history"].append(record_with_payload)
    manual_state["done"] = bool(obs.done)
    new_chunk = _format_step_markdown(record, obs)
    trace = manual_state.get("trace", "")
    trace = (trace + "\n\n---\n\n" + new_chunk) if trace else new_chunk
    manual_state["trace"] = trace
    signals = (obs.metadata or {}).get("reward_signals", {}) or {}
    final_md = _final_reward_table(obs) if obs.done else ""
    return trace, obs.visible_state, signals, final_md, "", manual_state


def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="Viveka — Reversibility + Calibration RL") as app:
        gr.Markdown(_HEADER_PITCH)
        manual_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=3):
                scenario_dd = gr.Dropdown(
                    label="Scenario",
                    choices=_scenario_options(),
                    value=(_scenario_options()[0] if _scenario_options() else None),
                )
                policy_radio = gr.Radio(
                    label="Policy",
                    choices=["naive", "heuristic", "manual"],
                    value="naive",
                    info=_POLICY_HELP,
                )
                run_btn = gr.Button("▶ Run scenario", variant="primary")
                reset_btn = gr.Button("↻ Reset")
                manual_action_in = gr.JSON(
                    label="Manual action (only used when policy=manual)",
                    value=_MANUAL_ACTION_TEMPLATE,
                )
                step_btn = gr.Button("Step")
                error_md = gr.Markdown("")

            with gr.Column(scale=7):
                user_msg_md = gr.Markdown("**User request**\n\n_Pick a scenario and run._")
                trace_md = gr.Markdown("**Live trace**\n\n_Run a scenario to see the trace._")
                state_json = gr.JSON(label="Service state", value={})
                signals_json = gr.JSON(label="Reward signals", value={})
                final_md = gr.Markdown("")

        run_btn.click(
            fn=_run_full,
            inputs=[scenario_dd, policy_radio],
            outputs=[user_msg_md, trace_md, state_json, signals_json, final_md, error_md],
        )
        reset_btn.click(
            fn=_reset_scenario,
            inputs=[scenario_dd],
            outputs=[user_msg_md, trace_md, state_json, signals_json, final_md, error_md, manual_state],
        )
        step_btn.click(
            fn=_step_manual,
            inputs=[scenario_dd, manual_action_in, manual_state],
            outputs=[trace_md, state_json, signals_json, final_md, error_md, manual_state],
        )

    return app
