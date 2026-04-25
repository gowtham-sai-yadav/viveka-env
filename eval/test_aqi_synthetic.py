"""Sanity-check Dunn / XBI / AQI on a synthetic 2-Gaussian probe.

Two well-separated Gaussians should give:
    DBS  -> small (~0.1)
    Dunn -> large (>1.0)
    XBI  -> small
    CHI  -> large
    AQI  -> large

Run:  python eval/test_aqi_synthetic.py
"""

import numpy as np
from aqi_probe import compute_aqi


def _make(separation: float = 8.0, n: int = 50, d: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=(n, d))
    b = rng.normal(separation, 1.0, size=(n, d))
    emb = np.concatenate([a, b], axis=0).astype(np.float32)
    lab = np.array([0] * n + [1] * n)
    return emb, lab


def main() -> None:
    sep_emb, sep_lab = _make(separation=8.0)
    over_emb, over_lab = _make(separation=0.5)

    print("== well separated (sep=8) ==")
    sep = compute_aqi(sep_emb, sep_lab)
    print(sep)
    print("== overlapping     (sep=0.5) ==")
    over = compute_aqi(over_emb, over_lab)
    print(over)

    assert sep["AQI"] > over["AQI"], "AQI should be larger when clusters separate"
    assert sep["Dunn"] > over["Dunn"], "Dunn should be larger when clusters separate"
    assert sep["XBI"] < over["XBI"], "XBI should be smaller when clusters separate"
    assert sep["DBS"] < over["DBS"], "DBS should be smaller when clusters separate"
    print("OK: monotonicity holds on synthetic Gaussians")


if __name__ == "__main__":
    main()
