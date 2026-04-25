"""Plot AQI delta between base and trained Qwen2-0.5B-Instruct.

Reads two JSONs produced by `aqi_probe.py` and emits `aqi_delta.png`
with a grouped bar chart over {DBS, Dunn, XBI, CHI, AQI}.

Usage:
    python eval/aqi_delta.py \
        --base eval/aqi_base.json \
        --trained eval/aqi_trained.json \
        --output eval/aqi_delta.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ["DBS", "Dunn", "XBI", "CHI", "AQI"]
# Direction: True = higher is better (Dunn, CHI, AQI); False = lower is better (DBS, XBI)
HIGHER_IS_BETTER = {"DBS": False, "Dunn": True, "XBI": False, "CHI": True, "AQI": True}


def _safe(v: float) -> float:
    if v is None or not np.isfinite(v):
        return 0.0
    return float(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--trained", required=True)
    p.add_argument("--output", default="eval/aqi_delta.png")
    args = p.parse_args()

    base = json.loads(Path(args.base).read_text())["metrics"]
    trained = json.loads(Path(args.trained).read_text())["metrics"]

    base_vals = np.array([_safe(base.get(m)) for m in METRICS])
    trained_vals = np.array([_safe(trained.get(m)) for m in METRICS])

    x = np.arange(len(METRICS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, base_vals, width, label="Base", color="#888")
    ax.bar(x + width / 2, trained_vals, width, label="GRPO-trained", color="#2a7")

    # annotate deltas with arrow direction
    for i, m in enumerate(METRICS):
        delta = trained_vals[i] - base_vals[i]
        good = (delta > 0) == HIGHER_IS_BETTER[m]
        sign = "+" if delta >= 0 else ""
        ax.text(
            x[i],
            max(base_vals[i], trained_vals[i]) * 1.02,
            f"{sign}{delta:.3f}\n{'OK' if good else 'BAD'}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=("#2a7" if good else "#c33"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("metric value")
    ax.set_title(
        "AQI (Borah et al., EMNLP 2025) — Qwen2-0.5B-Instruct base vs GRPO-trained\n"
        "AQI = lambda*(1/XBI) + (1-lambda)*log1p(CHI),   lambda=0.5"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
