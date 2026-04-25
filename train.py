"""GRPO training: Qwen2-0.5B-Instruct + Unsloth 4-bit QLoRA on Viveka OpenEnv.

Usage:
  python train.py --dry-run                                    # build everything, no GPU touch
  python train.py --smoke                                      # 10 episodes, gradient checks
  python train.py --episodes 200 --output-dir runs/v1          # full run
  python train.py --tier-mix "1:0.4,2:0.4,4:0.2" --no-wandb
  python train.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 200   # stretch
"""

from __future__ import annotations

import importlib.util
import sys
import types

# ── Stubs for broken transitive imports (MUST run BEFORE trl import in main()) ──
# TRL 0.24's import_utils calls importlib.util.find_spec on these names, which
# raises ValueError if __spec__ is None — so stubs need a real ModuleSpec.

class _DummyModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

def _install_stub(modname: str, dummy: bool = False) -> None:
    if modname in sys.modules:
        return
    m = _DummyModule(modname) if dummy else types.ModuleType(modname)
    m.__spec__ = importlib.util.spec_from_loader(modname, None)
    sys.modules[modname] = m

# llm_blender — broken on transformers 4.45+ (TRANSFORMERS_CACHE removed)
try:
    import llm_blender as _llm_blender  # noqa: F401
except Exception:  # noqa: BLE001
    _install_stub("llm_blender")
    sys.modules["llm_blender"].Blender = type("Blender", (), {})

# mergekit — its pydantic model has a torch.Tensor field that pydantic 2.13
# refuses to schema-generate. Catch-all dummy submodules so TRL's package
# init can do `from mergekit.X import Y` without actually loading mergekit.
for _name in ["mergekit", "mergekit.merge_methods", "mergekit.io",
              "mergekit.config", "mergekit.architecture", "mergekit.options",
              "mergekit.merge", "mergekit.plan", "mergekit.graph"]:
    _install_stub(_name, dummy=True)

import argparse
import json
import random
from pathlib import Path
from typing import Any

from viveka.models import VivekaAction
from viveka.server.environment import VivekaEnvironment

SYSTEM_PROMPT = (
    "You are Viveka, an Indian DPI assistant for UPI, DigiLocker, and IRCTC. "
    "Reason briefly, then output EXACTLY ONE JSON object (no prose, no code fences) "
    "matching this schema:\n"
    '{"action_type": "execute"|"confirm_with_user"|"ask_user"|"abstain"|"respond_to_user", '
    '"target_service": "upi"|"digilocker"|"irctc"|null, '
    '"operation": "<op_name>"|null, '
    '"params": {...}, '
    '"predicted_reversibility": "reversible"|"irreversible"|"irreversible_trivial"|null, '
    '"confidence": <float 0..1>, '
    '"reasoning": "<one short line>"}\n'
    "Prefer reversible actions; confirm_with_user before any irreversible action; "
    "abstain when uncertain."
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


# ── reward function: TRL 0.24 signature ───────────────────────────────────
# TRL 0.24 GRPOTrainer drops the experimental environment_factory hook and
# expects a stateless reward_funcs callable: takes prompts + completions,
# returns one scalar per completion. We parse a JSON action out of each
# completion, replay it through a fresh VivekaEnvironment scoped to the
# scenario the prompt referenced, and return the env's terminal reward.

import re

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# VivekaAction has model_config = ConfigDict(extra="forbid"), so any extra
# field the LLM emits ("thoughts", "explanation", etc.) raises ValidationError.
# We strip parsed dicts down to known fields before constructing the action.
_VIVEKA_ACTION_FIELDS = frozenset(
    ["action_type", "target_service", "operation", "params",
     "predicted_reversibility", "confidence", "reasoning"]
)


def _parse_action(text: str) -> dict | None:
    """Extract the first JSON object from completion text. None if unparseable."""
    cleaned = _FENCE_RE.sub("", text.strip())
    m = _JSON_RE.search(cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _score_completion(text: str, tier_id: int, scenario_idx: int) -> float:
    """Replay one completion against a fresh env. Returns terminal reward.
    Schema violations and dispatch failures map to -1.0 per CLAUDE.md."""
    parsed = _parse_action(text)
    if not parsed:
        return -1.0
    filtered = {k: v for k, v in parsed.items() if k in _VIVEKA_ACTION_FIELDS}
    try:
        action = VivekaAction(**filtered)
    except Exception:  # noqa: BLE001 — pydantic ValidationError + bad enums
        return -1.0
    try:
        env = VivekaEnvironment()
        env.reset(tier_id=tier_id, scenario_idx=scenario_idx)
        obs = env.step(action)
        # If the model's single action didn't terminate the episode, force a
        # respond_to_user so the rubric can compute terminal reward.
        if not obs.done:
            terminal = VivekaAction(
                action_type="respond_to_user",
                target_service=None,
                operation=None,
                params={"text": "done"},
                predicted_reversibility=None,
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning="auto-terminate after single-action rollout",
            )
            obs = env.step(terminal)
    except Exception:  # noqa: BLE001 — env failures are real signal, not crashes
        return -1.0
    return float(obs.reward or 0.0)


def reward_func(prompts=None, completions=None, **kwargs) -> list[float]:
    """TRL 0.24 reward_funcs callable. Dataset columns arrive via kwargs."""
    if not completions:
        return []
    n = len(completions)
    tier_ids = kwargs.get("tier_id") or [1] * n
    scenario_idxs = kwargs.get("scenario_idx") or [0] * n
    rewards: list[float] = []
    for completion, tier_id, scenario_idx in zip(completions, tier_ids, scenario_idxs):
        # TRL passes either list[dict] (chat format) or str (text format).
        if isinstance(completion, list) and completion:
            text = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
        else:
            text = str(completion)
        rewards.append(_score_completion(text, int(tier_id), int(scenario_idx)))
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

    from viveka.server.scenario_loader import all_tier_dirs, list_scenarios

    # Per-tier real scenario counts. Without this, randrange(0, 100) means
    # ~85% of training samples hit the empty-stub fallback in env.reset() —
    # zero learning signal. Bound to the actual count per tier instead.
    tier_dirs = all_tier_dirs()
    tier_counts = {tid: max(1, len(list_scenarios(d))) for tid, d in tier_dirs.items()}

    rng = random.Random(seed)
    tiers = list(tier_mix.keys())
    weights = [tier_mix[t] for t in tiers]
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        tier = rng.choices(tiers, weights=weights, k=1)[0]
        scenario_idx = rng.randrange(0, tier_counts.get(tier, 1))
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
