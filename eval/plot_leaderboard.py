"""Viveka leaderboard — frontier closed models vs our trained open models.

Reads:
  eval/baseline_claude_haiku.json
  eval/baseline_claude_sonnet.json
  eval/baseline_gpt_4o_mini_per_tier21.json
  eval/baseline_gpt5.2_3per_tier.json
  + hardcoded open-model numbers from earlier Kaggle eval logs

Writes:
  eval/plots/leaderboard.png    (clean style only; we do not produce an xkcd
                                 variant here because the leaderboard is the
                                 quantitative claim a judge will compare against
                                 their priors — hand-drawn aesthetics undersell
                                 the point.)

Design notes:
  - Horizontal bar, sorted by mean reward.
  - One row per policy. We deliberately omit Llama-3.2-3B (both base and
    trained) until the trained sealed-eval pass completes; including a row
    for a result we do not yet have would either lie or invite "where's
    the trained number?" follow-up.
  - Frontier closed models, our trained open models, and frozen open models
    use three distinct colour bands so the visual story is "closed > our
    trained > frozen".
  - T4 (adversarial) per-policy mean is annotated at the right of each bar
    so the safety-tier story reads at a glance — even Claude Sonnet's T4
    is 0.44.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLOR_FRONTIER = "#cc6633"
COLOR_FRONTIER_LITE = "#e8a570"
COLOR_TRAINED = "#1f5fa1"
COLOR_TRAINED_LITE = "#5b8ec9"
COLOR_FROZEN = "#b3b3b3"

FRONTIER_FILES = [
    ("Claude Sonnet 4.6", "eval/baseline_claude_sonnet.json", COLOR_FRONTIER),
    ("Claude Haiku 4.5",  "eval/baseline_claude_haiku.json",  COLOR_FRONTIER_LITE),
    ("GPT-4o-mini",       "eval/baseline_gpt_4o_mini_per_tier21.json", "#7a6cd1"),
    ("GPT-5.2",           "eval/baseline_gpt5.2_3per_tier.json", "#a99bdb"),
]

# Open-model numbers from sealed eval (n=20, 5 per tier × T1–T4) on the
# weighted-average grader. All three architectures fully evaluated.
OPEN_MODELS = [
    # (name, mean, T4_mean, color, kind)
    ("Viveka-Qwen-2.5-1.5B (trained)", 0.231, 0.199, COLOR_TRAINED,      "trained"),
    ("Viveka-Llama-3.2-3B (trained)",  0.165, 0.089, "#2ca02c",           "trained"),
    ("Viveka-Llama-3.2-1B (trained)",  0.131, 0.000, COLOR_TRAINED_LITE, "trained"),
    ("Llama-3.2-1B (frozen)",          0.289, 0.310, COLOR_FROZEN,       "frozen"),
    ("Qwen-2.5-1.5B (frozen)",         0.211, 0.290, COLOR_FROZEN,       "frozen"),
    ("Llama-3.2-3B (frozen)",          0.145, 0.126, "#d6d6d6",           "frozen"),
]


def _per_tier_from_json(path: Path) -> tuple[float, dict[int, float]]:
    d = json.loads(path.read_text())
    by_tier: dict[int, list[float]] = collections.defaultdict(list)
    for ep in d.get("scenarios", []):
        by_tier[ep.get("tier_id", 0)].append(float(ep.get("reward", 0.0)))
    tier_means = {t: sum(v) / len(v) for t, v in by_tier.items()}
    return float(d["mean_reward"]), tier_means


def plot_leaderboard(output_png: Path) -> None:
    rows: list[tuple[str, float, float, str, str]] = []  # (name, mean, t4, color, kind)

    for name, path, color in FRONTIER_FILES:
        mean, tiers = _per_tier_from_json(Path(path))
        rows.append((name, mean, tiers.get(4, 0.0), color, "frontier"))

    for name, mean, t4, color, kind in OPEN_MODELS:
        rows.append((name, mean, t4, color, kind))

    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)

    names = [r[0] for r in rows_sorted]
    means = [r[1] for r in rows_sorted]
    t4s = [r[2] for r in rows_sorted]
    colors = [r[3] for r in rows_sorted]
    kinds = [r[4] for r in rows_sorted]

    y = np.arange(len(rows_sorted))
    bar_h = 0.62

    ax.barh(y, means, height=bar_h, color=colors, edgecolor="white", linewidth=1.2)

    for i, (m, t4) in enumerate(zip(means, t4s)):
        ax.text(m + 0.01, y[i], f"{m:.3f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color="#1a1a1a")
        ax.text(m + 0.085, y[i], f"  · T4 {t4:.2f}",
                va="center", ha="left", fontsize=9, color="#666666")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Mean episode reward (sealed eval, weighted-average grader)", fontsize=10.5)
    ax.set_title(
        "Viveka Leaderboard — Frontier closed models vs trained open-source",
        fontsize=13, fontweight="bold", pad=14,
    )

    ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FRONTIER,    label="Frontier (closed-source)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TRAINED,     label="Viveka-trained (open, GRPO LoRA)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FROZEN,      label="Frozen baseline (open, no training)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=9,
              framealpha=0.95, edgecolor="#cccccc")

    ax.text(
        0.012, -1.05,
        "Frontier scored on n=12 (3/tier).  Open-source scored on n=20 (5/tier).  T4 = adversarial planted-trap tier.",
        transform=ax.transData, ha="left", va="top",
        fontsize=8, color="#666666", fontstyle="italic",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {output_png}")
    for r in rows_sorted:
        name, mean, t4, _, kind = r
        print(f"  [{kind:10s}] {name:36s}  mean={mean:.3f}  T4={t4:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-png", type=Path, default=Path("eval/plots/leaderboard.png"))
    args = p.parse_args()
    plot_leaderboard(args.output_png)


if __name__ == "__main__":
    main()
