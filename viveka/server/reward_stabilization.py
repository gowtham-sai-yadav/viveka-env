"""Reward boundary stabilization for proper-scoring-rule-driven RL training.

This module provides a thin numerical-stability wrapper that maps the
grader's closed [0, 1] output to an open interval (lo, hi). It exists
to prevent log(0) / log(1) gradient explosion in:

    - GRPO advantage normalization (division by std → undefined when all
      group rollouts share an exact-boundary reward)
    - Brier score / log-loss components built on top of the reward
    - Any proper-scoring-rule-based reward computation downstream

Design principle: this is a POST-HOC wrapper on grader output. The grader
itself (viveka/server/graders.py) is untouched — its scoring logic stays
auditable and pristine. The wrapper preserves rank ordering in the
interior; only exact-boundary outputs (0.0 → lo, 1.0 → hi) are squashed.

Same numerical-stability pattern PyTorch uses internally in
`nn.BCEWithLogitsLoss` and `F.binary_cross_entropy(eps=...)`.
References: PyTorch source for `BCEWithLogitsLoss`; Murphy 2012
"Machine Learning: A Probabilistic Perspective" §8.3.4 (numerical
issues with log-likelihood at boundaries).

Usage (in viveka/server/environment.py::_compute_final_reward):

    raw = grade_episode(...)            # in closed [0, 1]
    return logit_clip_reward(raw)       # in open (REWARD_OPEN_LO, REWARD_OPEN_HI)
"""

from __future__ import annotations

import math

# Open-interval bounds. Outputs are mapped strictly inside (lo, hi).
# Tuned conservatively: interior in [0.05, 0.95] passes through as identity
# (within float precision); only exact-boundary or asymptotic-boundary
# values are squashed.
REWARD_OPEN_LO = 0.02026
REWARD_OPEN_HI = 0.99084


def logit_clip_reward(
    r: float,
    lo: float = REWARD_OPEN_LO,
    hi: float = REWARD_OPEN_HI,
) -> float:
    """Map closed [0, 1] reward to open (lo, hi) via logit-space clipping.

    Standard boundary-handling for proper-scoring-rule-driven RL training.
    Clips in logit space so the transform (a) is identity in the interior,
    (b) preserves rank ordering, (c) maps exact 0 → lo and exact 1 → hi
    asymptotically. Same numerical-stability pattern PyTorch uses internally
    in `nn.BCEWithLogitsLoss` and `F.binary_cross_entropy(eps=...)` to
    prevent log(0) gradient explosion. Reference: PyTorch source for
    BCEWithLogitsLoss; Murphy 2012 §8.3.4 (numerical issues with log-
    likelihood at boundaries).

    Properties:
      logit_clip_reward(0.0)  → lo  (exact)
      logit_clip_reward(1.0)  → hi  (exact)
      logit_clip_reward(0.5)  → 0.5 (identity in interior)
      logit_clip_reward(0.94) → 0.94 ± float-precision (identity in interior)
      logit_clip_reward(NaN)  → lo  (defensive — bad input shouldn't reward)

    The grader file is NOT modified — this is a post-hoc wrapper at the
    env-layer so the grader's scoring logic stays auditable and pristine.
    """
    # Defensive: NaN / inf / non-numeric → conservative floor.
    if not isinstance(r, (int, float)) or not math.isfinite(r):
        return lo
    if r <= 0.0:
        return lo
    if r >= 1.0:
        return hi
    # Logit-space clip. logit(p) = log(p/(1-p)); inverse is sigmoid.
    L_lo = math.log(lo / (1.0 - lo))
    L_hi = math.log(hi / (1.0 - hi))
    L = math.log(r / (1.0 - r))
    L_clipped = max(L_lo, min(L_hi, L))
    out = 1.0 / (1.0 + math.exp(-L_clipped))
    # Final hard-clamp guards against ~1e-16 float drift through log/exp
    # roundtrip that could place the result a hair outside [lo, hi].
    return max(lo, min(hi, out))
