"""Loss curves for Qwen 1.5B / Llama 1B / Llama 3B trained on identical GRPO config.

Reads:
  --qwen-log     runs/qwen_v6/training_log.jsonl
  --llama-log    runs/llama_v3/training_log.jsonl
  --llama3b-log  runs/llama3b_v1/training_log.jsonl   (optional)

Writes:
  --output-png   eval/plots/loss_curves.png

Loss values are the GRPO surrogate loss (TRL `loss` field, written every 5 steps
by the training_log_callback). Negative values are normal — GRPO loss is the
signed advantage-weighted policy ratio; sign tells you "did the policy lean
toward higher-reward completions" but magnitude is what matters for stability.

Submission rule explicitly requires both a loss curve AND a reward curve as
committed image files. Reward curve lives in plot_combined_curves.py.
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


def plot_loss(
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

    qx, qy = _extract_series(qwen_rows, "loss")
    lx, ly = _extract_series(llama_rows, "loss")
    l3x, l3y = (
        _extract_series(llama3b_rows, "loss")
        if llama3b_rows
        else (np.array([]), np.array([]))
    )

    qy_smooth = _smooth(qy, smooth_window)
    ly_smooth = _smooth(ly, smooth_window)
    l3y_smooth = _smooth(l3y, smooth_window) if len(l3y) > 0 else l3y

    output_png.parent.mkdir(parents=True, exist_ok=True)

    if xkcd:
        plt.xkcd(scale=1.0, length=100, randomness=2)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)

    QWEN_COLOR = "#1f77b4"
    LLAMA_COLOR = "#888888"
    LLAMA3B_COLOR = "#2ca02c"

    ax.scatter(qx, qy, s=18, alpha=0.35, color=QWEN_COLOR)
    ax.plot(
        qx, qy_smooth, color=QWEN_COLOR, linewidth=2.4,
        label=f"Qwen2.5-1.5B (final loss={qy[-1]:+.4f})",
    )

    ax.scatter(lx, ly, s=18, alpha=0.35, color=LLAMA_COLOR)
    ax.plot(
        lx, ly_smooth, color=LLAMA_COLOR, linewidth=2.4,
        label=f"Llama-3.2-1B (final loss={ly[-1]:+.4f})",
    )

    if len(l3y) > 0:
        ax.scatter(l3x, l3y, s=18, alpha=0.35, color=LLAMA3B_COLOR)
        ax.plot(
            l3x, l3y_smooth, color=LLAMA3B_COLOR, linewidth=2.4,
            label=f"Llama-3.2-3B (final loss={l3y[-1]:+.4f})",
        )

    ax.axhline(0.0, color="#000000", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("GRPO surrogate loss\n(per logging step, mean across G=4 rollouts)", fontsize=11)
    ax.set_title(
        "Viveka GRPO Training Loss — Three Architectures, Identical Config\n"
        "GRPO loss is signed; near-zero = stable, large negative = strong gradient",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, linestyle="-" if xkcd else ":")
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_png}")
    print(f"  Qwen 1.5B:  n={len(qy)}  first={qy[0]:+.4f}  final={qy[-1]:+.4f}")
    print(f"  Llama 1B:   n={len(ly)}  first={ly[0]:+.4f}  final={ly[-1]:+.4f}")
    if len(l3y) > 0:
        print(f"  Llama 3B:   n={len(l3y)}  first={l3y[0]:+.4f}  final={l3y[-1]:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen-log", type=Path, default=Path("runs/qwen_v6/training_log.jsonl"))
    p.add_argument("--llama-log", type=Path, default=Path("runs/llama_v3/training_log.jsonl"))
    p.add_argument(
        "--llama3b-log", type=Path, default=Path("runs/llama3b_v1/training_log.jsonl"),
        help="Llama 3B training log (set to /dev/null to skip)",
    )
    p.add_argument("--output-png", type=Path, default=Path("eval/plots/loss_curves.png"))
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument(
        "--xkcd", action="store_true",
        help="Render in xkcd / hand-drawn style (matches reward curve aesthetic)",
    )
    args = p.parse_args()
    plot_loss(
        args.qwen_log, args.llama_log, args.output_png,
        args.smooth_window, args.xkcd, args.llama3b_log,
    )


if __name__ == "__main__":
    main()
