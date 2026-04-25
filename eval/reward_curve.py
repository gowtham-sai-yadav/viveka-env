"""GRPO reward curve with random/untrained baseline overlay (rubric-explicit).

Reads:
  --training-log    JSONL written by viveka.server.training_log_callback
  --baseline-json   JSON written by inference.py --policy random (mean_reward + scenarios)

Writes:
  --output-png      PNG with rolling-mean line, raw alpha=0.25 scatter,
                    horizontal baseline + shaded ±1σ band.

Run:
  python eval/reward_curve.py \
      --training-log runs/qwen05b/training_log.jsonl \
      --baseline-json eval/random.json \
      --output-png eval/plots/reward_curve.png \
      --smooth-window 10 \
      --title "GRPO Training — Qwen2-0.5B-Instruct on Viveka (200 episodes)"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_baseline(path: Path) -> tuple[float, float, int]:
    """Return (mean, std, n) from inference.py random-policy output."""
    payload = json.loads(path.read_text())
    if "mean_reward" not in payload and "random" in payload:
        payload = payload["random"]
    rewards = [float(s["reward"]) for s in payload["scenarios"]]
    if not rewards:
        return float(payload.get("mean_reward", 0.0)), 0.0, 0
    arr = np.array(rewards, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0)), len(arr)


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window - 1
    yp = np.concatenate([np.full(pad, y[0]), y])
    return np.convolve(yp, kernel, mode="valid")


def _extract_reward_series(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        v = r.get("reward")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        xs.append(int(r.get("episode", r.get("step", 0))))
        ys.append(float(v))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def plot_reward_curve(
    training_log: Path,
    baseline_json: Path,
    output_png: Path,
    smooth_window: int,
    title: str,
) -> None:
    rows = _read_jsonl(training_log)
    if not rows:
        raise SystemExit(f"empty training log: {training_log}")

    xs, ys = _extract_reward_series(rows)
    if len(ys) == 0:
        raise SystemExit("no `reward` field in any training-log row")

    base_mean, base_std, base_n = _load_baseline(baseline_json)
    smooth = _rolling_mean(ys, smooth_window)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)

    # Raw per-step scatter (low alpha) so judges see the noise.
    ax.scatter(xs, ys, s=10, alpha=0.25, color="#1f77b4", label=f"per-step reward (n={len(ys)})")

    # Smoothed learning curve — the headline line.
    ax.plot(xs, smooth, color="#1f77b4", linewidth=2.2, label=f"rolling mean (window={smooth_window})")

    # Baseline overlay — required by rubric.
    ax.axhline(
        base_mean,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"random baseline mean = {base_mean:.3f} (n={base_n})",
    )
    if base_std > 0:
        ax.fill_between(
            xs,
            base_mean - base_std,
            base_mean + base_std,
            color="#d62728",
            alpha=0.12,
            label=f"random baseline ±1σ ({base_std:.3f})",
        )

    ax.set_xlabel("episode" if "episode" in rows[0] else "step", fontsize=11)
    ax.set_ylabel("reward (Viveka grade_episode, [0, 1])", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    final_mean = float(np.mean(ys[-max(1, len(ys) // 10) :]))
    delta = final_mean - base_mean
    ax.text(
        0.02,
        0.97,
        f"final-decile mean: {final_mean:.3f}\nΔ vs random: {delta:+.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_png} ({len(ys)} points, baseline={base_mean:.3f}, final={final_mean:.3f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-log", type=Path, required=True)
    p.add_argument("--baseline-json", type=Path, required=True)
    p.add_argument("--output-png", type=Path, default=Path("eval/plots/reward_curve.png"))
    p.add_argument("--smooth-window", type=int, default=10)
    p.add_argument(
        "--title",
        default="GRPO Training — Qwen2-0.5B-Instruct on Viveka (200 episodes)",
    )
    args = p.parse_args()
    plot_reward_curve(
        args.training_log,
        args.baseline_json,
        args.output_png,
        args.smooth_window,
        args.title,
    )


if __name__ == "__main__":
    main()
