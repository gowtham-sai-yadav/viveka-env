"""Generate a synthetic 200-row training_log.jsonl + matching baseline JSON.

Used by tests/test_reward_curve.py and as a smoke fixture before the real
overnight run produces real artifacts.

Curve shape: starts near random baseline (~0.20), climbs sigmoidally to ~0.85
with realistic per-step noise. Mirrors what a healthy GRPO run on Viveka
should look like.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).parent


def synth_training_log(n: int = 200, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for step in range(n):
        # Sigmoid from ~0.20 → ~0.85 over n episodes, centered at 40% of run.
        progress = (step - n * 0.4) / (n * 0.18)
        mean = 0.20 + 0.65 / (1.0 + math.exp(-progress))
        noise = rng.gauss(0.0, 0.06)
        reward = max(0.0, min(1.0, mean + noise))
        rows.append(
            {
                "step": step,
                "episode": step,
                "wall_time": round(step * 8.4, 2),
                "epoch": round(step / 50.0, 4),
                "reward": round(reward, 4),
                "reward_std": round(0.18 - 0.10 * (step / n) + abs(rng.gauss(0, 0.01)), 4),
                "kl": round(0.02 + 0.03 * (step / n) + abs(rng.gauss(0, 0.005)), 5),
                "entropy": round(2.1 - 0.6 * (step / n), 4),
                "grad_norm": round(0.8 + abs(rng.gauss(0, 0.2)), 4),
                "completion_length": round(118 + rng.gauss(0, 8), 1),
                "clipped_ratio": round(max(0.0, 0.05 - 0.03 * (step / n)), 4),
                "learning_rate": 5e-6,
                "loss": round(-reward * 0.5 + rng.gauss(0, 0.02), 4),
                "rewards/reversibility/mean": round(min(1.0, 0.30 + reward * 0.7), 4),
                "rewards/task_completion/mean": round(min(1.0, 0.15 + reward * 0.85), 4),
                "rewards/confidence_brier/mean": round(min(1.0, 0.40 + reward * 0.55), 4),
            }
        )
    return rows


def synth_baseline(seed: int = 13) -> dict:
    rng = random.Random(seed)
    scenarios = []
    for i in range(30):
        scenarios.append(
            {
                "tier": (i % 4) + 1,
                "idx": i,
                "reward": round(max(0.0, min(1.0, rng.gauss(0.205, 0.085))), 4),
                "length": rng.randint(2, 8),
            }
        )
    mean = sum(s["reward"] for s in scenarios) / len(scenarios)
    return {
        "policy_name": "random",
        "n_scenarios": len(scenarios),
        "mean_reward": round(mean, 4),
        "scenarios": scenarios,
    }


def main() -> None:
    log_path = HERE / "training_log.jsonl"
    base_path = HERE / "baseline_random.json"
    with log_path.open("w") as f:
        for row in synth_training_log():
            f.write(json.dumps(row) + "\n")
    base_path.write_text(json.dumps(synth_baseline(), indent=2))
    print(f"wrote {log_path} (200 rows)")
    print(f"wrote {base_path}")


if __name__ == "__main__":
    main()
