"""Combined reward curves — Qwen 1.5B vs Llama 1B vs Llama 3B.

Reads:
  --qwen-log     runs/qwen_v6/training_log.jsonl
  --llama-log    runs/llama_v3/training_log.jsonl
  --llama3b-log  runs/llama3b_v1/training_log.jsonl   (optional)

Writes:
  --output-png   eval/plots/reward_curves_combined.png

The headline plot for the rubric: shows three architectures trained on identical
GRPO config. Qwen climbed past zero (after EOS-list fix), Llama-1B plateaued
deeply negative (capacity ceiling), and Llama-3B climbed cleanly without needing
the EOS fix.
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


def _extract_series(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        xs.append(int(r.get("step", r.get("episode", 0))))
        ys.append(float(v))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window - 1
    yp = np.concatenate([np.full(pad, y[0]), y])
    return np.convolve(yp, kernel, mode="valid")


def plot_combined(
    qwen_log: Path,
    llama_log: Path,
    output_png: Path,
    smooth_window: int = 3,
    xkcd: bool = False,
    llama3b_log: Path | None = None,
) -> None:
    qwen_rows = _read_jsonl(qwen_log)
    llama_rows = _read_jsonl(llama_log)
    llama3b_rows = _read_jsonl(llama3b_log) if llama3b_log and llama3b_log.exists() else []

    qx, qy = _extract_series(qwen_rows, "reward")
    lx, ly = _extract_series(llama_rows, "reward")
    l3x, l3y = (_extract_series(llama3b_rows, "reward")
                if llama3b_rows else (np.array([]), np.array([])))

    qy_smooth = _smooth(qy, smooth_window)
    ly_smooth = _smooth(ly, smooth_window)
    l3y_smooth = _smooth(l3y, smooth_window) if len(l3y) > 0 else l3y

    output_png.parent.mkdir(parents=True, exist_ok=True)

    if xkcd:
        plt.xkcd(scale=1.0, length=100, randomness=2)
    fig, (ax_main, ax_clip) = plt.subplots(
        2, 1, figsize=(11, 8), dpi=200, gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True
    )

    QWEN_COLOR = "#1f77b4"
    LLAMA_COLOR = "#888888"
    LLAMA3B_COLOR = "#2ca02c"

    ax_main.scatter(qx, qy, s=18, alpha=0.35, color=QWEN_COLOR)
    ax_main.plot(qx, qy_smooth, color=QWEN_COLOR, linewidth=2.4,
                 label=f"Qwen2.5-1.5B (final={qy[-1]:+.3f}, peak={qy.max():+.3f})")

    ax_main.scatter(lx, ly, s=18, alpha=0.35, color=LLAMA_COLOR)
    ax_main.plot(lx, ly_smooth, color=LLAMA_COLOR, linewidth=2.4,
                 label=f"Llama-3.2-1B (final={ly[-1]:+.3f}, peak={ly.max():+.3f})")

    if len(l3y) > 0:
        ax_main.scatter(l3x, l3y, s=18, alpha=0.35, color=LLAMA3B_COLOR)
        ax_main.plot(l3x, l3y_smooth, color=LLAMA3B_COLOR, linewidth=2.4,
                     label=f"Llama-3.2-3B (final={l3y[-1]:+.3f}, peak={l3y.max():+.3f})")

    ax_main.axhline(0.0, color="#000000", linestyle="-", linewidth=0.8, alpha=0.5)
    floor_style = "-" if xkcd else ":"
    ax_main.axhline(-1.0, color="#d62728", linestyle=floor_style, linewidth=1.0, alpha=0.6,
                    label="reward floor (parser fails)")

    ax_main.set_ylabel("Reward (per-step mean across G=4 rollouts)", fontsize=11)
    ax_main.set_title(
        "Viveka GRPO Training — Three Architectures, Identical Config\n"
        "Qwen 1.5B climbed past zero after EOS-list fix; Llama 1B plateaued; Llama 3B climbed cleanly",
        fontsize=12,
    )
    ax_main.grid(True, alpha=0.3, linestyle="-" if xkcd else ":")
    ax_main.legend(loc="lower right", frameon=True, fontsize=10)

    all_mins = [-1.0, qy.min(), ly.min()] + ([l3y.min()] if len(l3y) > 0 else [])
    all_maxes = [0.3, qy.max(), ly.max()] + ([l3y.max()] if len(l3y) > 0 else [])
    y_min = min(all_mins) - 0.05
    y_max = max(all_maxes) + 0.05
    ax_main.set_ylim(y_min, y_max)

    qx_clip, qy_clip = _extract_series(qwen_rows, "clipped_ratio")
    lx_clip, ly_clip = _extract_series(llama_rows, "clipped_ratio")
    l3x_clip, l3y_clip = (_extract_series(llama3b_rows, "clipped_ratio")
                          if llama3b_rows else (np.array([]), np.array([])))
    if len(qy_clip) > 0:
        ax_clip.plot(qx_clip, qy_clip, color=QWEN_COLOR, linewidth=2.0, marker="o", markersize=4,
                     label=f"Qwen 1.5B (final={qy_clip[-1]:.3f})")
    if len(ly_clip) > 0:
        ax_clip.plot(lx_clip, ly_clip, color=LLAMA_COLOR, linewidth=2.0, marker="s", markersize=4,
                     label=f"Llama 1B (final={ly_clip[-1]:.3f})")
    if len(l3y_clip) > 0:
        ax_clip.plot(l3x_clip, l3y_clip, color=LLAMA3B_COLOR, linewidth=2.0, marker="^", markersize=4,
                     label=f"Llama 3B (final={l3y_clip[-1]:.3f})")
    ax_clip.set_ylabel("Clipped ratio\n(lower = healthier)", fontsize=10)
    ax_clip.set_xlabel("Training step", fontsize=11)
    ax_clip.set_ylim(0.0, 1.05)
    ax_clip.grid(True, alpha=0.3, linestyle="-" if xkcd else ":")
    ax_clip.legend(loc="upper right", frameon=True, fontsize=9)

    qy_final = float(qy[-1])
    ly_final = float(ly[-1])
    box_lines = [
        f"Qwen 1.5B final:  {qy_final:+.3f}",
        f"Llama 1B final:   {ly_final:+.3f}",
    ]
    if len(l3y) > 0:
        l3y_final = float(l3y[-1])
        box_lines.append(f"Llama 3B final:   {l3y_final:+.3f}")
    ax_main.text(
        0.02,
        0.97,
        "\n".join(box_lines),
        transform=ax_main.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_png}")
    print(f"  Qwen 1.5B:  n={len(qy)} steps, final={qy_final:+.4f}, peak={qy.max():+.4f}")
    print(f"  Llama 1B:   n={len(ly)} steps, final={ly_final:+.4f}, peak={ly.max():+.4f}")
    if len(l3y) > 0:
        print(f"  Llama 3B:   n={len(l3y)} steps, final={float(l3y[-1]):+.4f}, peak={l3y.max():+.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-log", type=Path, default=Path("runs/qwen_v6/training_log.jsonl"))
    p.add_argument("--llama-log", type=Path, default=Path("runs/llama_v3/training_log.jsonl"))
    p.add_argument("--llama3b-log", type=Path, default=Path("runs/llama3b_v1/training_log.jsonl"),
                   help="Llama 3B training log (set to /dev/null to skip)")
    p.add_argument("--output-png", type=Path, default=Path("eval/plots/reward_curves_combined.png"))
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--xkcd", action="store_true",
                   help="Render in xkcd / hand-drawn style for the README hero image")
    args = p.parse_args()
    plot_combined(
        args.qwen_log, args.llama_log, args.output_png, args.smooth_window,
        args.xkcd, args.llama3b_log,
    )


if __name__ == "__main__":
    main()
