"""Unit test for eval/reward_curve.py — uses the synthetic fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "eval" / "fixtures"
sys.path.insert(0, str(ROOT))

from eval.fixtures.make_synthetic_training_log import (  # noqa: E402
    synth_baseline,
    synth_training_log,
)
from eval.reward_curve import (  # noqa: E402
    _extract_reward_series,
    _load_baseline,
    _read_jsonl,
    _rolling_mean,
    plot_reward_curve,
)


@pytest.fixture(scope="module")
def fixture_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("rcurve")
    log_path = d / "training_log.jsonl"
    base_path = d / "baseline_random.json"
    with log_path.open("w") as f:
        for row in synth_training_log():
            f.write(json.dumps(row) + "\n")
    base_path.write_text(json.dumps(synth_baseline(), indent=2))
    return log_path, base_path, d


def test_jsonl_roundtrip_has_200_rows(fixture_paths):
    log_path, _, _ = fixture_paths
    rows = _read_jsonl(log_path)
    assert len(rows) == 200
    assert all("reward" in r for r in rows)
    assert all(0.0 <= r["reward"] <= 1.0 for r in rows)


def test_baseline_loader_returns_mean_std_n(fixture_paths):
    _, base_path, _ = fixture_paths
    mean, std, n = _load_baseline(base_path)
    assert n == 30
    assert 0.10 < mean < 0.30, f"random baseline should be ~0.20, got {mean}"
    assert std > 0.02


def test_reward_series_climbs(fixture_paths):
    log_path, _, _ = fixture_paths
    rows = _read_jsonl(log_path)
    xs, ys = _extract_reward_series(rows)
    assert len(xs) == 200
    early = ys[:20].mean()
    late = ys[-20:].mean()
    assert late > early + 0.4, f"synthetic curve must climb >0.4: {early=}, {late=}"


def test_rolling_mean_shape_preserved():
    import numpy as np

    y = np.linspace(0.0, 1.0, 50)
    smoothed = _rolling_mean(y, window=10)
    assert smoothed.shape == y.shape
    # Monotone-increasing input → smoothed is also monotone non-decreasing.
    assert all(smoothed[i + 1] >= smoothed[i] - 1e-9 for i in range(len(smoothed) - 1))


def test_rolling_mean_window_one_is_identity():
    import numpy as np

    y = np.array([0.1, 0.4, 0.2, 0.9])
    assert (_rolling_mean(y, window=1) == y).all()


def test_plot_reward_curve_writes_png(fixture_paths):
    log_path, base_path, d = fixture_paths
    out = d / "out.png"
    plot_reward_curve(log_path, base_path, out, smooth_window=10, title="test")
    assert out.exists()
    assert out.stat().st_size > 10_000  # non-trivial PNG


def test_plot_handles_missing_reward_field(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"step": 0, "kl": 0.01}) + "\n")
    base = tmp_path / "base.json"
    base.write_text(json.dumps(synth_baseline()))
    with pytest.raises(SystemExit, match="no `reward` field"):
        plot_reward_curve(log, base, tmp_path / "x.png", 10, "t")
