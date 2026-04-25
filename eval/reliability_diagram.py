"""Publication-quality reliability diagrams for Viveka policies.

Reads `inference.py` JSON dumps (per-policy or multi-policy bundle) and emits an
overlay reliability plot with ECE/MCE annotations. Brier-scored RLCR rewards
mean the trained policy should hug the diagonal; base Qwen will not.

Usage:
  python eval/reliability_diagram.py \
      --inputs eval/qwen_base.json,eval/viveka_trained.json \
      --output-png docs/reliability.png \
      --bin-strategy equal-width --n-bins 10
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ─── colour palette (cb-friendly: Wong 2011) ──────────────────────────────
_PALETTE = {
    "qwen_base": "#888888",
    "qwen": "#888888",
    "random": "#D55E00",
    "gpt4o": "#009E73",
    "gpt4o_mini": "#009E73",
    "viveka_trained": "#1F77B4",
    "viveka": "#1F77B4",
    "trained": "#1F77B4",
}
_FALLBACK_COLORS = ["#1F77B4", "#888888", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]

_EXEC_TYPES = {"execute", "confirm_with_user"}
_VALID_REVS = {"reversible", "irreversible", "irreversible_trivial"}


# ─── extraction ───────────────────────────────────────────────────────────


def extract_pairs(trajectory: list[dict]) -> list[tuple[float, int]]:
    """Return (confidence, correctness) pairs from a per-step trajectory.

    Filters to execute / confirm_with_user actions where predicted_reversibility
    is set AND ground_truth_reversibility is recorded by the env. Other action
    types (ask_user, abstain, respond_to_user) carry no reversibility prediction
    and are excluded — including them as "correctness=0" would slander policies
    that correctly abstain on uncertainty.
    """
    pairs: list[tuple[float, int]] = []
    for step in trajectory:
        if step.get("action_type") not in _EXEC_TYPES:
            continue
        pred = step.get("predicted_reversibility")
        if pred not in _VALID_REVS:
            continue
        # Two shapes supported: explicit `correctness` (0/1) OR
        # `ground_truth_reversibility` for derivation.
        if "correctness" in step:
            corr = int(bool(step["correctness"]))
        elif "ground_truth_reversibility" in step:
            corr = int(step["ground_truth_reversibility"] == pred)
        else:
            continue
        conf = step.get("confidence")
        if conf is None:
            continue
        c = float(conf)
        if not (0.0 <= c <= 1.0) or not np.isfinite(c):
            continue
        pairs.append((c, corr))
    return pairs


def _pairs_from_bundle(obj: dict, fallback_name: str) -> tuple[str, list[tuple[float, int]]]:
    """Pull (policy_name, pairs) from one policy block of inference.py output."""
    name = obj.get("policy_name", fallback_name)
    pairs: list[tuple[float, int]] = []
    for sc in obj.get("scenarios", []):
        traj = sc.get("trajectory")
        if traj:
            pairs.extend(extract_pairs(traj))
    return name, pairs


def load_inputs(paths: Iterable[Path]) -> dict[str, list[tuple[float, int]]]:
    """Load one-or-more inference JSONs into {policy_name: pairs}.

    Supports both shapes inference.py emits:
      - single policy: {"policy_name": ..., "scenarios": [...]}
      - bundle: {pname: {"policy_name": ..., "scenarios": [...]}, ...}
    """
    out: dict[str, list[tuple[float, int]]] = {}
    for p in paths:
        with open(p) as f:
            obj = json.load(f)
        if "scenarios" in obj and "policy_name" in obj:
            name, pairs = _pairs_from_bundle(obj, p.stem)
            out[name] = pairs
        else:
            for k, v in obj.items():
                if isinstance(v, dict) and "scenarios" in v:
                    name, pairs = _pairs_from_bundle(v, k)
                    out[name] = pairs
    return out


# ─── calibration metrics ──────────────────────────────────────────────────


def expected_calibration_error(
    pairs: list[tuple[float, int]],
    n_bins: int = 10,
    strategy: str = "equal-width",
) -> dict:
    """ECE = sum_m (|B_m|/N) * |acc(B_m) - conf(B_m)|  (Guo et al. 2017).

    Also returns MCE (max gap) and per-bin centers/acc/conf/count for plotting.
    `strategy="equal-frequency"` uses quantile boundaries (Nixon et al. 2019,
    "Adaptive Calibration Error") — recommended by ICLR Blogposts 2025 when
    confidences cluster, since modern models pile mass into a few bins.
    """
    if not pairs:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "n": 0,
            "bin_edges": np.linspace(0, 1, n_bins + 1),
            "bin_centers": np.zeros(n_bins),
            "bin_acc": np.full(n_bins, np.nan),
            "bin_conf": np.full(n_bins, np.nan),
            "bin_count": np.zeros(n_bins, dtype=int),
        }

    confs = np.asarray([c for c, _ in pairs], dtype=float)
    corrs = np.asarray([y for _, y in pairs], dtype=float)
    n = len(confs)

    if strategy == "equal-frequency":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(confs, qs))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    actual_bins = len(edges) - 1
    # right-closed except last bin (np convention); clip into range first.
    idx = np.clip(np.digitize(confs, edges[1:-1], right=False), 0, actual_bins - 1)

    bin_acc = np.full(actual_bins, np.nan)
    bin_conf = np.full(actual_bins, np.nan)
    bin_count = np.zeros(actual_bins, dtype=int)
    for m in range(actual_bins):
        sel = idx == m
        cnt = int(sel.sum())
        bin_count[m] = cnt
        if cnt > 0:
            bin_acc[m] = corrs[sel].mean()
            bin_conf[m] = confs[sel].mean()

    populated = bin_count > 0
    gaps = np.abs(bin_acc[populated] - bin_conf[populated])
    weights = bin_count[populated] / n
    ece = float(np.sum(weights * gaps)) if populated.any() else float("nan")
    mce = float(np.max(gaps)) if populated.any() else float("nan")

    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "ece": ece,
        "mce": mce,
        "n": n,
        "bin_edges": edges,
        "bin_centers": centers,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_count": bin_count,
        "strategy": strategy,
    }


# ─── plotting ─────────────────────────────────────────────────────────────


_HERO_RC = {
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _color_for(name: str, fallback_idx: int) -> str:
    return _PALETTE.get(name, _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)])


def _draw_axes(ax: plt.Axes) -> None:
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="_perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy (reversibility correct)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)


def plot_reliability(
    policies: dict[str, list[tuple[float, int]]],
    output_png: str,
    title: str = "Viveka Calibration",
    n_bins: int = 10,
    strategy: str = "equal-width",
) -> dict[str, dict]:
    """Render reliability diagram(s).

    - >=2 policies: overlay on a single axis (base-vs-trained delta is the
      story; bars become noisy with overlay so we draw connected accuracy
      curves with translucent confidence bars in policy colour).
    - 1 policy: single axis with bars (Guo-et-al. style).
    """
    metrics = {
        name: expected_calibration_error(pairs, n_bins=n_bins, strategy=strategy)
        for name, pairs in policies.items()
    }

    with plt.rc_context(_HERO_RC):
        fig, ax = plt.subplots(figsize=(7, 7))
        _draw_axes(ax)

        items = list(policies.items())
        if len(items) == 1:
            name, _ = items[0]
            m = metrics[name]
            color = _color_for(name, 0)
            pop = m["bin_count"] > 0
            centers = m["bin_centers"][pop]
            confs = m["bin_conf"][pop]
            accs = m["bin_acc"][pop]
            width = (m["bin_edges"][1] - m["bin_edges"][0]) * 0.9 if len(m["bin_edges"]) > 1 else 0.09
            ax.bar(
                centers,
                confs,
                width=width,
                color="#BBBBBB",
                alpha=0.45,
                edgecolor="white",
                label="mean confidence",
                zorder=2,
            )
            ax.bar(
                centers,
                accs,
                width=width,
                color=color,
                alpha=0.85,
                edgecolor="white",
                label="empirical accuracy",
                zorder=3,
            )
            for x, c, a in zip(centers, confs, accs, strict=False):
                ax.plot([x, x], [min(c, a), max(c, a)], color="#222", linewidth=1.0, alpha=0.6, zorder=4)
        else:
            for i, (name, _) in enumerate(items):
                m = metrics[name]
                color = _color_for(name, i)
                pop = m["bin_count"] > 0
                if not pop.any():
                    continue
                centers = m["bin_centers"][pop]
                accs = m["bin_acc"][pop]
                counts = m["bin_count"][pop]
                sizes = 30 + 220 * (counts / counts.max())
                ax.plot(
                    centers,
                    accs,
                    color=color,
                    linewidth=2.2,
                    alpha=0.95,
                    marker="o",
                    markersize=0,
                    zorder=3 + i,
                    label=f"{name}  (ECE={m['ece']:.3f}, n={m['n']})",
                )
                ax.scatter(
                    centers,
                    accs,
                    s=sizes,
                    color=color,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=4 + i,
                )

        ax.set_title(title, pad=12)
        ax.legend(loc="lower right", frameon=False)

        # ECE / MCE annotation block (upper-left). Single-policy keeps it tidy;
        # multi-policy already shows ECE in the legend, so we annotate MCE only.
        lines = []
        for name, m in metrics.items():
            if np.isnan(m["ece"]):
                lines.append(f"{name}: no data")
            elif len(items) == 1:
                lines.append(f"ECE = {m['ece']:.3f}\nMCE = {m['mce']:.3f}\nn = {m['n']}")
            else:
                lines.append(f"{name}: MCE={m['mce']:.3f}")
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        )

        out = Path(output_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return {
        k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
        for k, v in metrics.items()
    }


# ─── synthetic sanity test ────────────────────────────────────────────────


def _synthetic_demo(out_png: str = "eval/_synthetic_reliability.png") -> None:
    rng = np.random.default_rng(0)
    # Base: overconfident — confidence ~ U[0.5, 1.0], accuracy ~ 0.5 regardless.
    base_confs = rng.uniform(0.5, 1.0, size=400)
    base_corrs = (rng.uniform(0, 1, size=400) < 0.5).astype(int)
    base = list(zip(base_confs.tolist(), base_corrs.tolist(), strict=False))
    # Trained: accuracy ≈ confidence (perfectly calibrated).
    tr_confs = rng.uniform(0.0, 1.0, size=400)
    tr_corrs = (rng.uniform(0, 1, size=400) < tr_confs).astype(int)
    trained = list(zip(tr_confs.tolist(), tr_corrs.tolist(), strict=False))
    metrics = plot_reliability(
        {"qwen_base": base, "viveka_trained": trained},
        output_png=out_png,
        title="Viveka Calibration — synthetic sanity",
    )
    print(
        json.dumps({k: {"ece": v["ece"], "mce": v["mce"], "n": v["n"]} for k, v in metrics.items()}, indent=2)
    )


# ─── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", help="comma-separated paths to inference.py JSON dumps")
    p.add_argument("--output-png", default="docs/reliability.png")
    p.add_argument("--title", default="Viveka Calibration")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--bin-strategy", choices=["equal-width", "equal-frequency"], default="equal-width")
    p.add_argument("--synthetic", action="store_true", help="emit a sanity-check plot from synthetic data")
    args = p.parse_args()

    if args.synthetic:
        _synthetic_demo(args.output_png)
        return

    if not args.inputs:
        raise SystemExit("--inputs required (or pass --synthetic)")
    paths = [Path(s.strip()) for s in args.inputs.split(",") if s.strip()]
    policies = load_inputs(paths)
    if not policies:
        raise SystemExit(f"no policy data found in {paths}")
    metrics = plot_reliability(
        policies,
        output_png=args.output_png,
        title=args.title,
        n_bins=args.n_bins,
        strategy=args.bin_strategy,
    )
    summary = {k: {"ece": v["ece"], "mce": v["mce"], "n": v["n"]} for k, v in metrics.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
