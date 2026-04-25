"""Smoke test for train.py --dry-run path. No GPU, no train extras required."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_train_dry_run_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(REPO / "train.py"), "--dry-run", "--episodes", "3", "--no-wandb"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[smoke] terminal reward=" in proc.stdout
    assert "[dry-run] OK" in proc.stdout


def test_train_smoke_check_env_directly():
    """Import + drive a trajectory through VivekaToolEnv without subprocess."""
    sys.path.insert(0, str(REPO))
    try:
        from train import VivekaToolEnv
    finally:
        sys.path.pop(0)

    env = VivekaToolEnv()
    msg = env.reset(tier_id=1, scenario_idx=0)
    assert isinstance(msg, str) and len(msg) > 0
    assert env.done is False

    env.execute("upi", "check_balance", {}, "reversible", 0.9, "read-only probe")
    assert env._steps == 1
    assert env.done is False
    assert env._signals != {}

    env.respond_to_user("done", 0.9, "task complete")
    assert env.done is True
    assert 0.0 <= env.reward <= 1.0


def test_train_imports_without_train_extras():
    """train.py must be importable without unsloth/trl/torch installed."""
    sys.path.insert(0, str(REPO))
    try:
        import importlib

        if "train" in sys.modules:
            del sys.modules["train"]
        m = importlib.import_module("train")
        assert hasattr(m, "VivekaToolEnv")
        assert hasattr(m, "reward_func")
        assert hasattr(m, "build_dataset")
        assert hasattr(m, "main")
    finally:
        sys.path.pop(0)
