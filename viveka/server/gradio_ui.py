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


def _infer_params(scenario: dict[str, Any], service: str, operation: str) -> dict[str, Any]:
    """Pick realistic params for the heuristic so operations actually succeed against the mock state.

    Falls back to {} when initial_state lacks the entity. Naive policy intentionally
    bypasses this — empty-params failures are the baseline.
    """
    initial = scenario.get("initial_state", {}) or {}

    if service == "digilocker":
        dgl = initial.get("digilocker", {}) or {}
        docs = dgl.get("documents", []) or []
        consents = dgl.get("consents", []) or []
        first_doc_id = docs[0].get("doc_id") if docs else None
        first_active_consent = next(
            (c.get("consent_id") for c in consents if c.get("status") == "active"),
            consents[0].get("consent_id") if consents else None,
        )
        if operation in {"view_document", "fetch_document", "delete_document"}:
            return {"doc_id": first_doc_id} if first_doc_id else {}
        if operation == "share_document":
            return {"doc_id": first_doc_id, "recipient": "hdfc-bank"} if first_doc_id else {}
        if operation == "issue_consent_token":
            return (
                {
                    "doc_id": first_doc_id,
                    "audience": "hdfc-bank",
                    "scope": ["aadhaar.read"],
                    "ttl_minutes": 30,
                }
                if first_doc_id
                else {}
            )
        if operation == "revoke_consent":
            return {"consent_id": first_active_consent} if first_active_consent else {}
        return {}

    if service == "irctc":
        irctc = initial.get("irctc", {}) or {}
        catalogue = irctc.get("catalogue", []) or []
        bookings = irctc.get("bookings", []) or []
        avail = irctc.get("availability", {}) or {}
        first_train = catalogue[0] if catalogue else None
        first_pnr = bookings[0].get("pnr") if bookings else None
        if operation == "search_trains":
            if first_train:
                return {
                    "from_station": first_train.get("from_station", ""),
                    "to_station": first_train.get("to_station", ""),
                }
            return {}
        if operation == "check_seat_availability":
            if first_train:
                tn = first_train.get("train_no", "")
                cls = next(iter(avail.get(tn, {}).keys()), "SL")
                return {"train_no": tn, "class": cls}
            return {}
        if operation in {"check_pnr", "cancel_booking"}:
            return {"pnr": first_pnr} if first_pnr else {}
        if operation == "modify_booking":
            return {"pnr": first_pnr, "class": "3A"} if first_pnr else {}
        if operation == "book_ticket":
            if first_train:
                tn = first_train.get("train_no", "")
                cls_with_seats = next(
                    (cls for cls, n in avail.get(tn, {}).items() if n > 0),
                    "SL",
                )
                return {
                    "train_no": tn,
                    "class": cls_with_seats,
                    "passengers": [{"name": "Demo Passenger", "age": 30, "gender": "M"}],
                }
            return {}
        return {}

    if service == "upi":
        upi = initial.get("upi", {}) or {}
        balance = float(upi.get("balance", 0))
        contacts = upi.get("contacts", {}) or {}
        mandates = upi.get("mandates", []) or []
        cards = upi.get("cards", []) or []
        txns = upi.get("transactions", []) or []
        first_vpa = next(iter(contacts.values()), None)
        first_pending_mandate = next(
            (m.get("mandate_id") for m in mandates if m.get("status") == "pending"),
            mandates[0].get("mandate_id") if mandates else None,
        )
        first_card_last4 = cards[0].get("last4") if cards else None
        first_txn_id = txns[0].get("transaction_ref_id") if txns else None
        if operation == "send_money":
            if first_vpa:
                return {"payee_vpa": first_vpa, "amount": min(500.0, max(balance / 4, 100.0))}
            return {}
        if operation in {"approve_mandate", "reject_mandate"}:
            return {"mandate_id": first_pending_mandate} if first_pending_mandate else {}
        if operation == "block_card":
            return {"card_last4": first_card_last4} if first_card_last4 else {}
        if operation == "raise_dispute":
            return {"transaction_ref_id": first_txn_id} if first_txn_id else {}
        if operation == "lookup_vpa":
            return {"vpa": first_vpa} if first_vpa else {}
        return {}

    return {}


def _naive_policy(scenario: dict[str, Any], obs: VivekaObservation) -> VivekaAction:
    raw_sequence = _ground_truth_sequence(scenario)
    sequence: list[dict[str, Any]] = []
    for gt in raw_sequence:
        if sequence and (
            sequence[-1].get("target_service"),
            sequence[-1].get("operation"),
        ) == (gt.get("target_service"), gt.get("operation")):
            continue
        sequence.append(gt)
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
    raw_sequence = _ground_truth_sequence(scenario)
    sequence: list[dict[str, Any]] = []
    for gt in raw_sequence:
        if sequence and (
            sequence[-1].get("target_service"),
            sequence[-1].get("operation"),
        ) == (gt.get("target_service"), gt.get("operation")):
            continue
        sequence.append(gt)
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
    params = _infer_params(scenario, service, operation)
    if label == "irreversible" and not already_confirmed:
        return VivekaAction(
            action_type="confirm_with_user",
            target_service=service,
            operation=operation,
            params=params,
            predicted_reversibility="irreversible",
            confidence=0.9,
            reasoning="Heuristic: confirm before irreversible action.",
        )
    return VivekaAction(
        action_type="execute",
        target_service=service,
        operation=operation,
        params=params,
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


def _run_episode_silent(
    scenario: dict[str, Any], tier_id: int, scenario_idx: int, policy_name: str
) -> tuple[VivekaObservation, list[dict[str, Any]]]:
    env = VivekaEnvironment()
    obs = env.reset(tier_id=tier_id, scenario_idx=scenario_idx)
    history: list[dict[str, Any]] = []
    max_steps = scenario.get("expected", {}).get("max_steps", 30)
    step_count = 0
    while not obs.done and step_count < max_steps:
        try:
            if policy_name == "naive":
                action = _naive_policy(scenario, obs)
            else:
                action = _heuristic_policy(scenario, obs, history)
            obs = env.step(action)
        except Exception:
            break
        step_count += 1
        record = (env._actions_taken or [{}])[-1]
        history.append(record)
    return obs, history


_TIER_LABEL = {
    "t1_easy": "T1 easy",
    "t2_medium": "T2 medium",
    "t3_hard": "T3 hard / Hinglish",
    "t4_adversarial": "T4 adversarial",
}


def _compare_all_scenarios() -> str:
    options = _scenario_options()
    if not options:
        return "_No scenarios available._"

    by_tier: dict[str, list[tuple[str, float, float, float]]] = {k: [] for k in _TIER_LABEL}
    grand_naive = 0.0
    grand_heur = 0.0
    grand_n = 0

    for choice in options:
        try:
            tier_id, idx = _parse_scenario_choice(choice)
            scenario = load_scenario_by_tier(tier_id, idx)
        except Exception:
            continue
        try:
            naive_obs, _ = _run_episode_silent(scenario, tier_id, idx, "naive")
            heur_obs, _ = _run_episode_silent(scenario, tier_id, idx, "heuristic")
        except Exception:
            continue
        nr = naive_obs.reward if naive_obs.reward is not None else 0.0
        hr = heur_obs.reward if heur_obs.reward is not None else 0.0
        tier_dir = choice.split("/", 1)[0]
        by_tier.setdefault(tier_dir, []).append((choice, nr, hr, hr - nr))
        grand_naive += nr
        grand_heur += hr
        grand_n += 1

    if grand_n == 0:
        return "_No scenarios ran successfully._"

    out: list[str] = ["## 🆚 Naive vs Heuristic — full bench (per-tier breakdown)", ""]
    for tier_dir, label in _TIER_LABEL.items():
        rows_t = by_tier.get(tier_dir, [])
        if not rows_t:
            continue
        nt = sum(r[1] for r in rows_t) / len(rows_t)
        ht = sum(r[2] for r in rows_t) / len(rows_t)
        dt = ht - nt
        sign = "+" if dt > 0 else ""
        out.append(f"### {label} ({len(rows_t)} scenarios)")
        out.append("")
        out.append("| Scenario | Naive | Heuristic | Δ |")
        out.append("|---|---|---|---|")
        for choice, nr, hr, delta in rows_t:
            short = choice.split("/", 1)[1]
            d_sign = "+" if delta > 0 else ""
            out.append(f"| `{short}` | {nr:.3f} | {hr:.3f} | {d_sign}{delta:.3f} |")
        out.append(f"| **{label} mean** | **{nt:.3f}** | **{ht:.3f}** | **{sign}{dt:.3f}** |")
        out.append("")

    grand_naive_mean = grand_naive / grand_n
    grand_heur_mean = grand_heur / grand_n
    grand_delta = grand_heur_mean - grand_naive_mean
    grand_sign = "+" if grand_delta > 0 else ""
    out.append("---")
    out.append("")
    out.append(
        f"### Overall — {grand_n} scenarios · Naive `{grand_naive_mean:.3f}` · "
        f"Heuristic `{grand_heur_mean:.3f}` · **Δ `{grand_sign}{grand_delta:.3f}`**"
    )
    return "\n".join(out)


def _compare_policies(scenario_choice: str) -> str:
    if not scenario_choice:
        return "_Pick a scenario first._"
    try:
        tier_id, idx = _parse_scenario_choice(scenario_choice)
        scenario = load_scenario_by_tier(tier_id, idx)
    except Exception as e:
        return f"**Error:** {e}"

    naive_obs, naive_hist = _run_episode_silent(scenario, tier_id, idx, "naive")
    heur_obs, heur_hist = _run_episode_silent(scenario, tier_id, idx, "heuristic")

    naive_signals = (naive_obs.metadata or {}).get("reward_signals", {}) or {}
    heur_signals = (heur_obs.metadata or {}).get("reward_signals", {}) or {}
    naive_reward = naive_obs.reward if naive_obs.reward is not None else 0.0
    heur_reward = heur_obs.reward if heur_obs.reward is not None else 0.0
    delta = heur_reward - naive_reward

    if delta > 0:
        verdict = f"heuristic wins by **{abs(delta):.3f}**"
    elif delta < 0:
        verdict = f"naive wins by **{abs(delta):.3f}**"
    else:
        verdict = "tie"

    rows = [
        f"## 🆚 Policy comparison · `{scenario_choice}`",
        "",
        f"- **Naive baseline:** reward `{naive_reward:.3f}` · {len(naive_hist)} steps",
        f"- **Heuristic policy:** reward `{heur_reward:.3f}` · {len(heur_hist)} steps",
        "",
        f"### Δ reward: `{'+' if delta >= 0 else ''}{delta:.3f}` — {verdict}",
        "",
        "| Reward signal | Naive | Heuristic | Δ |",
        "|---|---|---|---|",
    ]
    keys = sorted(set(naive_signals) | set(heur_signals))
    for k in keys:
        n = float(naive_signals.get(k, 0.0) or 0.0)
        h = float(heur_signals.get(k, 0.0) or 0.0)
        d = h - n
        sign = "+" if d > 0 else ""
        rows.append(f"| `{k}` | {n:.3f} | {h:.3f} | {sign}{d:.3f} |")

    return "\n".join(rows)


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
                compare_btn = gr.Button("🆚 Compare both policies", variant="secondary")
                compare_all_btn = gr.Button("🆚 Compare on all scenarios")
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
                comparison_md = gr.Markdown("")
                comparison_all_md = gr.Markdown("")

        run_btn.click(
            fn=_run_full,
            inputs=[scenario_dd, policy_radio],
            outputs=[user_msg_md, trace_md, state_json, signals_json, final_md, error_md],
        )
        compare_btn.click(
            fn=_compare_policies,
            inputs=[scenario_dd],
            outputs=[comparison_md],
        )
        compare_all_btn.click(
            fn=_compare_all_scenarios,
            inputs=[],
            outputs=[comparison_all_md],
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
