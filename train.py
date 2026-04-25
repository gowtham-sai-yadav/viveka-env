"""GRPO training: Qwen2-0.5B-Instruct + Unsloth 4-bit QLoRA on Viveka OpenEnv.

Usage:
  python train.py --dry-run                                    # build everything, no GPU touch
  python train.py --smoke                                      # 10 episodes, gradient checks
  python train.py --episodes 200 --output-dir runs/v1          # full run
  python train.py --tier-mix "1:0.4,2:0.4,4:0.2" --no-wandb
  python train.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 200   # stretch
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from viveka.models import VivekaAction
from viveka.server.environment import VivekaEnvironment

SYSTEM_PROMPT = (
    "You are Viveka, an Indian DPI assistant for UPI, DigiLocker, and IRCTC. "
    "Reason briefly, then call exactly one tool. "
    "Prefer reversible actions; confirm_with_user before any irreversible action; "
    "abstain when uncertain. Always pass predicted_reversibility "
    "('reversible' | 'irreversible' | 'irreversible_trivial') and confidence in [0,1]."
)


# ── env adapter: one tool per action_type ─────────────────────────────────
# TRL v1 environment_factory expects a class with reset() + tool methods. The
# trainer routes the model's tool calls to these methods. Each method
# constructs a VivekaAction (Pydantic, extra=forbid) and dispatches via env.step.


class VivekaToolEnv:
    """One env instance per generation. Stateless across reset()."""

    def __init__(self) -> None:
        self.env = VivekaEnvironment()
        self.reward = 0.0
        self.done = False
        self._steps = 0
        self._signals: dict[str, float] = {}

    def reset(self, **kwargs: Any) -> str:
        tier_id = int(kwargs.get("tier_id", 1))
        scenario_idx = int(kwargs.get("scenario_idx", 0))
        obs = self.env.reset(tier_id=tier_id, scenario_idx=scenario_idx)
        self.reward = 0.0
        self.done = False
        self._steps = 0
        self._signals = {}
        return obs.user_message or "Scenario loaded."

    def execute(
        self,
        target_service: str,
        operation: str,
        params: dict[str, Any] | None = None,
        predicted_reversibility: str = "reversible",
        confidence: float = 0.5,
        reasoning: str = "",
    ) -> str:
        """Execute a service operation. Use only after assessing reversibility.

        Args:
            target_service: 'upi' | 'digilocker' | 'irctc'.
            operation: registered op name (e.g. 'check_balance', 'send_money').
            params: operation-specific dict.
            predicted_reversibility: 'reversible' | 'irreversible' | 'irreversible_trivial'.
            confidence: float in [0, 1].
            reasoning: 1-line justification.
        """
        return self._dispatch(
            "execute", target_service, operation, params or {}, predicted_reversibility, confidence, reasoning
        )

    def confirm_with_user(
        self,
        target_service: str,
        operation: str,
        params: dict[str, Any] | None = None,
        predicted_reversibility: str = "irreversible",
        confidence: float = 0.5,
        reasoning: str = "",
    ) -> str:
        """Ask the human to confirm before an irreversible action."""
        return self._dispatch(
            "confirm_with_user",
            target_service,
            operation,
            params or {},
            predicted_reversibility,
            confidence,
            reasoning,
        )

    def ask_user(self, question: str, confidence: float = 0.5, reasoning: str = "") -> str:
        """Ask a clarifying question when info is missing."""
        return self._dispatch("ask_user", None, None, {"question": question}, None, confidence, reasoning)

    def abstain(self, reasoning: str = "", confidence: float = 0.5) -> str:
        """Abstain when stakes are high and info is low."""
        return self._dispatch("abstain", None, None, {}, None, confidence, reasoning)

    def respond_to_user(self, text: str, confidence: float = 0.7, reasoning: str = "") -> str:
        """Final answer to the user; ends the episode."""
        return self._dispatch("respond_to_user", None, None, {"text": text}, None, confidence, reasoning)

    def _dispatch(
        self,
        action_type: str,
        target_service: str | None,
        operation: str | None,
        params: dict[str, Any],
        predicted_reversibility: str | None,
        confidence: float,
        reasoning: str,
    ) -> str:
        if self.done:
            return "Episode already terminated."
        try:
            action = VivekaAction(
                action_type=action_type,  # type: ignore[arg-type]
                target_service=target_service,  # type: ignore[arg-type]
                operation=operation,
                params=params,
                predicted_reversibility=predicted_reversibility,  # type: ignore[arg-type]
                confidence=float(max(0.0, min(1.0, confidence))),
                reasoning=str(reasoning)[:500],
            )
        except Exception as e:  # noqa: BLE001 - schema-violation surfaces as in-band error
            return f"Action validation error: {e}"
        obs = self.env.step(action)
        self._steps += 1
        if obs.metadata and "reward_signals" in obs.metadata:
            self._signals = dict(obs.metadata["reward_signals"])
        if obs.done:
            self.done = True
            self.reward = float(obs.reward or 0.0)
        return obs.message or "(no message)"


# ── reward function: TRL v1 signature ─────────────────────────────────────


def reward_func(environments, **kwargs):
    """TRL v1 GRPO reward: scalar per env. Per-step signals exposed via log_metric."""
    log_metric = kwargs.get("log_metric")
    rewards = [float(env.reward) for env in environments]
    if log_metric is not None and environments:
        agg: dict[str, list[float]] = {}
        for env in environments:
            for k, v in (env._signals or {}).items():
                agg.setdefault(k, []).append(float(v))
        for k, vs in agg.items():
            log_metric(f"signal/{k}_mean", sum(vs) / len(vs))
        log_metric("episode/mean_steps", sum(e._steps for e in environments) / len(environments))
    return rewards


# ── dataset: tier-mixed prompts ────────────────────────────────────────────


def parse_tier_mix(s: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for part in s.split(","):
        k, v = part.split(":")
        out[int(k)] = float(v)
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


def build_dataset(tier_mix: dict[int, float], n: int, seed: int = 0):
    """Construct a TRL-compatible Dataset of prompts. Imports lazy."""
    from datasets import Dataset

    rng = random.Random(seed)
    tiers = list(tier_mix.keys())
    weights = [tier_mix[t] for t in tiers]
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        tier = rng.choices(tiers, weights=weights, k=1)[0]
        scenario_idx = rng.randrange(0, 100)
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"You are paged to a tier-{tier} Indian DPI scenario. Investigate and act.",
                    },
                ],
                "tier_id": tier,
                "scenario_idx": scenario_idx,
            }
        )
    return Dataset.from_list(rows)


# ── main ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--tier-mix", default="1:0.4,2:0.4,4:0.2")
    p.add_argument("--output-dir", default="runs/grpo_v1")
    p.add_argument(
        "--dry-run", action="store_true", help="build env+dataset, print config, exit (no GPU touch)"
    )
    p.add_argument(
        "--smoke", action="store_true", help="10-episode sanity run with NaN guards and gradient checks"
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def print_config(args: argparse.Namespace, tier_mix: dict[int, float]) -> None:
    cfg = {
        "model": args.model,
        "episodes": args.episodes,
        "tier_mix": tier_mix,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "no_wandb": args.no_wandb,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
    }
    print("[config]", json.dumps(cfg, indent=2))


def smoke_check_env(args: argparse.Namespace) -> None:
    """Construct VivekaToolEnv, drive 1 trivial trajectory, confirm reward fires."""
    env = VivekaToolEnv()
    msg = env.reset(tier_id=1, scenario_idx=0)
    print(f"[smoke] reset OK: {msg[:80]}")
    env.execute("upi", "check_balance", {}, "reversible", 0.9, "read-only")
    print(f"[smoke] step OK, signals={list(env._signals.keys())[:4]}")
    env.respond_to_user("done", 0.9, "task complete")
    print(f"[smoke] terminal reward={env.reward:.4f}")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.episodes = 10

    random.seed(args.seed)
    tier_mix = parse_tier_mix(args.tier_mix)
    print_config(args, tier_mix)

    if args.dry_run:
        smoke_check_env(args)
        try:
            ds = build_dataset(tier_mix, n=args.episodes, seed=args.seed)
            print(f"[dry-run] dataset built: {len(ds)} prompts, columns={list(ds.column_names)}")
        except ImportError:
            print(
                f"[dry-run] datasets lib not installed; would build {args.episodes} prompts (mix={tier_mix})"
            )
        print("[dry-run] OK. Skipping model + trainer (no GPU touch).")
        return

    # Heavy imports are lazy: keeps --dry-run usable on CPU-only laptops.
    try:
        import torch
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: F401
    except ImportError as e:  # noqa: BLE001
        print(f"[error] training extras not installed: {e}")
        print('  Install with: uv sync --extra train  (or: pip install -e ".[train]" + unsloth)')
        sys.exit(2)

    torch.manual_seed(args.seed)

    class NaNGuard(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if not logs:
                return
            gn = logs.get("grad_norm")
            if gn is None:
                return
            try:
                gn_f = float(gn)
            except (TypeError, ValueError):
                return
            if gn_f != gn_f or gn_f in (float("inf"), float("-inf")):  # NaN/Inf
                print(f"[NaNGuard] non-finite grad_norm={gn_f} step={state.global_step} HALTING")
                control.should_training_stop = True
            elif gn_f > 10.0:
                print(f"[NaNGuard] WARN grad_norm={gn_f:.3f} step={state.global_step}")

    print(f"[load] {args.model} via Unsloth 4-bit ...")
    max_seq = 1280
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq,
        load_in_4bit=True,
        fast_inference=False,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = build_dataset(tier_mix, n=args.episodes, seed=args.seed)

    bf16 = is_bfloat16_supported()
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=768,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        beta=0.0,
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=4,
        logging_steps=1 if args.smoke else 5,
        bf16=bf16,
        fp16=not bf16,
        report_to=("none" if args.no_wandb else "wandb"),
        seed=args.seed,
        num_train_epochs=1,
    )

    from viveka.server.training_log_callback import TrainingLogCallback

    log_path = Path(args.output_dir) / "training_log.jsonl"
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=VivekaToolEnv,
        callbacks=[NaNGuard(), TrainingLogCallback(log_path)],
    )

    print(
        f"[train] {args.episodes} episodes, G={cfg.num_generations}, "
        f"bs={cfg.per_device_train_batch_size}x{cfg.gradient_accumulation_steps}"
    )
    trainer.train()

    out = Path(args.output_dir) / "lora"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[done] LoRA saved -> {out}")


if __name__ == "__main__":
    main()
