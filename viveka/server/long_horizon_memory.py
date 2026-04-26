"""Long-horizon memory orchestration for multi-turn agents.

Theme alignment: Long-Horizon Planning + Instruction Following.

This module provides the agent-facing memory channel that lets a 1.5B+
parameter agent maintain coherent context across an entire 30-step
episode within a tight (~1.5k token) prompt budget. The mechanism:

    1. Goal anchor (sticky every step) — entity bullets parsed
       deterministically from scenario.initial_state. Keeps the user's
       intent visible across all turns.
    2. Rolling action log (last K=5) — compact rendering of recent
       (action, outcome) pairs. Lets the agent observe its own trace.
    3. Loop detection — flags identical-action repetition so the agent
       can break out via abstain / respond_to_user / different op.
    4. Reasoning echo — surfaces the agent's own prior reasoning as
       cognitive continuity (within-episode self-reflection, Theme 4 supporting).
    5. State diff — delta vs previous observation; highlights what
       changed without re-reading the full state JSON.

Design constraints:
    - Pure derivation from scenario data + the env's existing action trace.
      No external LLM calls. No scenario-file modification. No invented
      narrative.
    - Every helper is wrapped in safe-empty fallbacks at the env layer
      (see VivekaEnvironment._make_observation) so a helper bug never
      breaks an observation.
    - Rendered into observation.metadata as separate keys; consumed by
      viveka.prompts.build_user_prompt and rendered into structured
      sections (## GOAL_ANCHOR, ## RECENT_ACTIONS, etc.).
"""

from __future__ import annotations

import json
from typing import Any

# ── Tunables ────────────────────────────────────────────────────────────────
# Length of rolling action log surfaced to the agent each step. K=5 gives
# full recent context without bloating the 1.5B-model 1.5k-token user-prompt.
RECENT_ACTIONS_K = 5
# Threshold for loop detection. If the last N actions are identical
# (action_type, target_service, operation, params), a warning fires.
LOOP_DETECT_K = 3
# Truncation for the prior-reasoning echo. Keeps the prompt small.
LAST_REASONING_MAX = 200
# Fields excluded from state_diff because they tick on every observation
# (clock-like) and would pollute the diff with non-actionable noise.
DIFF_NOISY_KEYS = {"now_iso"}


# ── Internal utilities ──────────────────────────────────────────────────────

def _short(v: Any, n: int = 20) -> str:
    """Truncate any value to a string of length n with an ellipsis."""
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_repr(v: Any, n: int = 40) -> str:
    """Compact JSON-like single-line repr of a value, truncated to n chars."""
    try:
        s = json.dumps(v, default=str)
    except Exception:
        s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


# ── Public API ──────────────────────────────────────────────────────────────

def format_recent_actions_lines(
    actions_taken: list[dict[str, Any]],
    k: int = RECENT_ACTIONS_K,
) -> list[str]:
    """Render the last k actions as compact lines for the agent to attend to.

    This is the within-episode memory channel — without it, the agent has
    no way to detect that it just looped on the same failing action.
    Surfaced via observation.metadata["recent_actions"]. Empty when no
    actions have been taken yet (step 1).
    """
    if not actions_taken:
        return []
    recent = actions_taken[-k:]
    lines: list[str] = []
    for a in recent:
        at = a.get("action_type", "?")
        svc = a.get("target_service") or "-"
        op = a.get("operation") or "-"
        result = a.get("result") or {}
        err = result.get("error_code")
        if err:
            outcome = f"ERR={err}"
        elif result.get("abstained"):
            outcome = "abstained"
        elif result:
            outcome = "ok"
        else:
            outcome = ""
        params = a.get("params") or {}
        try:
            params_compact = ", ".join(
                f"{key}={_short(params[key], 20)}"
                for key in list(params.keys())[:2]
            )
        except Exception:
            params_compact = ""
        step = a.get("step", "?")
        lines.append(f"  step{step}: {at} {svc}.{op}({params_compact}) → {outcome}")
    return lines


def detect_loop(
    actions_taken: list[dict[str, Any]],
    k: int = LOOP_DETECT_K,
) -> tuple[bool, str | None]:
    """Detect when the last k actions are byte-identical (action_type, service,
    operation, params). This is the loop visible in baseline traces — the
    agent re-tries the same failed view_document() forever because it has
    no working memory. Surfaced via metadata so the prompt can warn the
    agent and the agent can break out via abstain / respond_to_user / a
    different operation.
    """
    if len(actions_taken) < k:
        return False, None
    sigs: list[tuple[Any, ...]] = []
    for a in actions_taken[-k:]:
        try:
            params_str = json.dumps(a.get("params") or {}, sort_keys=True, default=str)
        except Exception:
            params_str = str(a.get("params") or {})
        sigs.append((
            a.get("action_type"),
            a.get("target_service"),
            a.get("operation"),
            params_str,
        ))
    if len(set(sigs)) == 1:
        return True, (
            f"WARNING: identical action repeated {k}+ times — try a different "
            f"operation, abstain, or respond_to_user."
        )
    return False, None


def extract_last_reasoning(
    actions_taken: list[dict[str, Any]],
    max_len: int = LAST_REASONING_MAX,
) -> str | None:
    """Pull the most-recent action's `reasoning` field for cognitive echo.

    The agent's own prior thought becomes input to its next decision —
    within-episode self-improvement via reflection. Returns None on step 1
    (no prior action) and on empty/missing reasoning (avoid empty echoes).
    """
    if not actions_taken:
        return None
    reasoning = actions_taken[-1].get("reasoning")
    if not isinstance(reasoning, str):
        return None
    reasoning = reasoning.strip()
    if not reasoning:
        return None
    return reasoning if len(reasoning) <= max_len else reasoning[: max_len - 1] + "…"


def extract_goal_entities(initial_state: dict[str, Any]) -> list[str]:
    """Surface salient entities from scenario.initial_state as a flat bullet
    list. Pure deterministic extraction — no narrative invention, no LLM,
    no scenario file modification. Caps at 5 bullets to keep prompt tight.

    These serve as sticky salience anchors across the whole episode so the
    agent doesn't lose track of who/what it's transacting on (e.g., "two
    contacts named Rohit — disambiguate before send_money").
    """
    if not isinstance(initial_state, dict) or not initial_state:
        return []
    bullets: list[str] = []

    upi = initial_state.get("upi") or {}
    if isinstance(upi, dict):
        contacts = upi.get("contacts") or {}
        if isinstance(contacts, dict) and contacts:
            names = ", ".join(list(contacts.keys())[:3])
            bullets.append(f"UPI contacts: {names}")
        payer = upi.get("payer_vpa")
        if payer:
            bullets.append(f"User VPA: {payer}")
        fraud = upi.get("fraud_vpa")
        if isinstance(fraud, list) and fraud:
            bullets.append(f"Flagged-fraud VPAs: {len(fraud)}")

    dgl = initial_state.get("digilocker") or {}
    if isinstance(dgl, dict):
        docs = dgl.get("documents") or []
        if isinstance(docs, list) and docs:
            types = ", ".join(
                str(d.get("doc_type", "?")) for d in docs[:3] if isinstance(d, dict)
            )
            if types:
                bullets.append(f"DigiLocker docs: {types}")

    irctc = initial_state.get("irctc") or {}
    if isinstance(irctc, dict):
        bookings = irctc.get("bookings") or []
        if isinstance(bookings, list) and bookings:
            pnrs = ", ".join(
                str(b.get("pnr", "?")) for b in bookings[:2] if isinstance(b, dict)
            )
            if pnrs:
                bullets.append(f"IRCTC bookings (PNR): {pnrs}")

    bnk = initial_state.get("banking") or {}
    if isinstance(bnk, dict):
        bens = bnk.get("beneficiaries") or []
        if isinstance(bens, list) and bens:
            names = ", ".join(
                str(b.get("name", "?")) for b in bens[:2] if isinstance(b, dict)
            )
            if names:
                bullets.append(f"Beneficiaries: {names}")

    tel = initial_state.get("telecom") or {}
    if isinstance(tel, dict):
        sims = tel.get("sims") or []
        if isinstance(sims, list) and sims:
            msisdns = ", ".join(
                str(s.get("msisdn", "?")) for s in sims[:2] if isinstance(s, dict)
            )
            if msisdns:
                bullets.append(f"Telecom SIMs: {msisdns}")

    return bullets[:5]


def compute_state_diff(
    prev: dict[str, Any] | None,
    curr: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Compact diff between two redacted-snapshot dicts.

    Returns {service: {field: "before → after"}} for keys whose values
    changed since the previous observation. Capped at ~5 total entries to
    keep the prompt small. Skips known-noisy keys (clock-like fields).
    On the first observation (prev is None), returns {} — there's no
    baseline yet.
    """
    if prev is None or not isinstance(prev, dict) or not isinstance(curr, dict):
        return {}
    diff: dict[str, dict[str, str]] = {}
    total = 0
    for svc in set(prev.keys()) | set(curr.keys()):
        prev_svc = prev.get(svc) or {}
        curr_svc = curr.get(svc) or {}
        if not isinstance(prev_svc, dict) or not isinstance(curr_svc, dict):
            continue
        svc_diff: dict[str, str] = {}
        keys = (set(prev_svc.keys()) | set(curr_svc.keys())) - DIFF_NOISY_KEYS
        for key in keys:
            before = prev_svc.get(key)
            after = curr_svc.get(key)
            if before == after:
                continue
            if isinstance(before, list) and isinstance(after, list):
                if len(before) != len(after):
                    svc_diff[key] = f"{len(before)} → {len(after)} items"
            else:
                svc_diff[key] = f"{_short_repr(before, 40)} → {_short_repr(after, 40)}"
            total += 1
            if total >= 5:
                break
        if svc_diff:
            diff[svc] = svc_diff
        if total >= 5:
            break
    return diff
