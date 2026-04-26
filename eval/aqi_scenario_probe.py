"""Build the Viveka-domain AQI probe set from real scenarios.

Replaces the 20 hand-written aligned/misaligned prompts in `aqi_probe.py`
with the actual scenario user_messages: T1+T2 reversible/safe ops are the
"aligned" cluster; T4 adversarial traps are the "misaligned" cluster.

Why this is better:
- Domain-tight: probes the exact distribution the model was trained on
- Larger N: ~30 safe + ~25 unsafe = 55 prompts (vs the paper's 50+50 floor)
- Auditable: every prompt traces to a scenario file, no hand-crafted cherry-picks

References:
- Borah, Sharma, Khanna, Shirawalmath et al. (EMNLP 2025), arXiv:2506.13901
  Section 5: "evaluated on a 500-pair subset" — 50/50 is the validated floor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from viveka.server.scenario_loader import all_tier_dirs, list_scenarios


def build_probe_set(scenarios_dir: Path | None = None) -> tuple[list[str], list[int], list[str]]:
    """Return (prompts, labels, scenario_ids).
    label 0 = aligned (T1 + T2 reversible/easy)
    label 1 = misaligned (T4 adversarial trap)
    """
    tier_dirs = all_tier_dirs()
    safe: list[tuple[str, str]] = []
    unsafe: list[tuple[str, str]] = []

    for tier_id, tier_dir in tier_dirs.items():
        for path in list_scenarios(tier_dir):
            with open(path) as f:
                raw = json.load(f)
            user_msg = raw.get("user_message", "").strip()
            sid = raw.get("scenario_id", path.stem)
            if not user_msg:
                continue
            if tier_id in (1, 2):
                safe.append((user_msg, sid))
            elif tier_id == 4:
                unsafe.append((user_msg, sid))

    prompts = [p for p, _ in safe] + [p for p, _ in unsafe]
    labels = [0] * len(safe) + [1] * len(unsafe)
    sids = [s for _, s in safe] + [s for _, s in unsafe]
    return prompts, labels, sids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="eval/probe_set_viveka.json")
    args = p.parse_args()

    prompts, labels, sids = build_probe_set()
    payload = {
        "prompts": prompts,
        "labels": labels,
        "scenario_ids": sids,
        "n_aligned": int(sum(1 for l in labels if l == 0)),
        "n_misaligned": int(sum(1 for l in labels if l == 1)),
        "source": "Viveka T1+T2 (safe) vs T4 (adversarial)",
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {args.output}  "
        f"n_aligned={payload['n_aligned']}  n_misaligned={payload['n_misaligned']}"
    )


if __name__ == "__main__":
    main()
