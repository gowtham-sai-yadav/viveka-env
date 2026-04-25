"""JSONL logging callback for TRL GRPOTrainer — one line per logging step.

Schema written per line:
  {step, episode, wall_time, reward, reward_std, kl, grad_norm,
   completion_length, entropy, clip_ratio, learning_rate, epoch, ...}

The plotter `eval/reward_curve.py` consumes this file. We do NOT depend on
W&B / Trackio being available — JSONL is the canonical artifact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
except ImportError:  # transformers is a [train] extra; allow module import for tests.
    TrainerCallback = object  # type: ignore[misc,assignment]
    TrainerControl = TrainerState = TrainingArguments = object  # type: ignore[misc,assignment]

_REWARD_KEYS = ("reward", "rewards/mean", "train/reward")
_LEN_KEYS = ("completions/mean_length", "completion_length", "train/completion_length")


def _first_present(logs: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in logs:
            return float(logs[k])
    return None


class TrainingLogCallback(TrainerCallback):
    """Append one JSON dict per `on_log` call. Robust to missing keys."""

    def __init__(self, jsonl_path: str | Path, episodes_per_step: int = 1) -> None:
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)
        self._t0 = time.time()
        self._episodes_per_step = episodes_per_step

    def on_log(  # noqa: D401 — TRL/HF callback contract
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or not state.is_world_process_zero:
            return

        step = int(state.global_step)
        record = {
            "step": step,
            "episode": step * self._episodes_per_step,
            "wall_time": round(time.time() - self._t0, 3),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "reward": _first_present(logs, _REWARD_KEYS),
            "reward_std": logs.get("reward_std") or logs.get("train/reward_std"),
            "kl": logs.get("kl") or logs.get("train/kl"),
            "entropy": logs.get("entropy"),
            "grad_norm": logs.get("grad_norm"),
            "completion_length": _first_present(logs, _LEN_KEYS),
            "clipped_ratio": logs.get("completions/clipped_ratio"),
            "learning_rate": logs.get("learning_rate"),
            "loss": logs.get("loss"),
        }
        # Also surface per-reward-function components for diagnostics.
        for k, v in logs.items():
            if k.startswith("rewards/") and k.endswith("/mean"):
                record[k] = float(v)
        self._fh.write(json.dumps(record, default=str) + "\n")

    def on_train_end(  # noqa: D401
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if hasattr(self, "_fh") and not self._fh.closed:
            self._fh.close()
