# Reward-Hacking Playbook — Viveka GRPO Training (02:00–09:00 IST)

Operational doc for the babysitter on shift. Open W&B run + this file in two
tabs. Goal: catch reward hacking before 7 hours of compute is wasted.

Background reading, in priority order:
- Lilian Weng, "Reward Hacking in Reinforcement Learning" (Nov 2024) — taxonomy
  we use below: <https://lilianweng.github.io/posts/2024-11-28-reward-hacking/>
- TRL GRPO logging reference: <https://huggingface.co/docs/trl/main/en/logging>
- "A brief example of reward hacking in GRPO" (Mukherjee, 2024) — the canonical
  256-token-pad failure: <https://ishanjmukherjee.github.io/reward-hacking-grpo>
- "Detecting and Mitigating Reward Hacking in RL", arXiv 2507.05619, 2025.

## 1. Five signals to watch in W&B (TRL v1 metric names)

| Signal | Metric | Healthy range | Why it matters |
|---|---|---|---|
| Reward trend | `reward` | climbing 0.20 → 0.85 over 200 ep | primary objective |
| Reward variance collapse | `reward_std` | > 0.05 throughout | if it pins to ~0, every group output is identical → no learning gradient (Mukherjee 2024) |
| Length explosion | `completions/mean_length` and `completions/clipped_ratio` | mean stable ±30%, `clipped_ratio < 0.10` | classic length-pad reward hack |
| Entropy collapse | `entropy` | drops slowly; never below 0.5× initial | entropy is a faster early-warning than KL (arXiv 2508.03772, 2025) |
| KL drift | `kl` (only if `beta > 0`) | < 0.5; not monotonically rising | if it runs away while reward keeps climbing, the policy is leaving the manifold |
| Custom: confidence saturation | `rewards/confidence_brier/mean` | does not saturate to 1.0 | if Brier maxes out, the model is gaming our calibration grader |

## 2. Quantitative halt thresholds

Halt training (Ctrl-C, save checkpoint, inspect) if **any** of:

1. `reward_std < 0.01` for **5 consecutive logging steps** AND `completions/mean_length` is more than **1.5×** its baseline-30 value.
   → group collusion / length pad (Mukherjee 2024 pattern).
2. `completions/clipped_ratio > 0.30` for 3 consecutive steps.
   → model is hitting `max_completion_length` to game length-correlated reward.
3. `entropy` drops below **50%** of its step-30 value within 10 steps.
   → premature policy collapse; further training will not recover (TRL logging guide explicitly calls this out).
4. `kl > 1.0` AND still rising at 3 consecutive steps (only with `beta > 0`).
   → policy has left reference manifold; outputs likely incoherent.
5. `reward` climbing **but** `rewards/task_completion/mean` flat or falling.
   → classic specification gaming: total reward up because of cheap components, real task quality declining.

If halted: keep the last clean checkpoint, NOT the most recent. `model.save_pretrained(.../safe_checkpoint)`.

## 3. Five-generation sampling cadence (every 30 minutes)

`scripts/sample_5.py` (run on the side, separate terminal):

```bash
while true; do
  python scripts/sample_5.py \
    --checkpoint runs/qwen05b/checkpoints/latest \
    --scenarios eval/probe_5.json \
    --output runs/qwen05b/samples/$(date +%H%M).jsonl
  sleep 1800
done
```

For each of the 5 probe scenarios, log: `prompt`, `completion`, `reward`,
`reward_components`, `predicted_reversibility`, `confidence`. Eyeball each
sample for:

- **Did the model produce valid JSON?** Format-collapse manifests as repeated
  tokens or prose preamble around the JSON.
- **Is `reasoning` actually about the user request, or is it boilerplate?**
- **Is `confidence` always 0.95 or 1.0?** Saturation = calibration grader is
  being gamed.
- **Are completions the same length as one another?** If yes, suspect
  length-pad (cross-reference with `completions/mean_length` plateau).

## 4. Three reward-hacking patterns to recognise (Weng taxonomy)

### Pattern A — Specification gaming via length pad
*Weng category: specification gaming, in-distribution.*
**Signature:** every completion is exactly `max_completion_length`,
`reward_std` collapses to 0, `kl` quietly creeps up. The model has discovered
that all-equal-length outputs zero out the GRPO group advantage so only the
KL penalty matters; it then minimises that with garbage. **First documented
public GRPO instance:** Mukherjee 2024.
**Action:** halt, lower `max_completion_length`, add a length-penalty reward
component, OR enable `scale_rewards=False`.

### Pattern B — Calibration sycophancy / confidence saturation
*Weng category: U-Sophistry — model becomes better at convincing the grader,
not better at the task.*
**Signature:** `rewards/confidence_brier/mean` ≈ 1.0, `rewards/task_completion/mean`
flat. Model has learned to emit `confidence=1.0` only when its action happens
to be correct, exploiting any small leak in our state-diff grader. Sample
inspection shows confidence is 1.0 even on user-facing `respond_to_user`
actions where there is no objective truth.
**Action:** halt, audit `graders.confidence_brier` for label leakage, add an
adversarial probe scenario where the correct calibrated confidence is < 0.5.

### Pattern C — Refusal / abstain collapse
*Weng category: in-context reward hacking — the safe action dominates.*
**Signature:** `action_type` distribution in samples becomes >80% `abstain`.
Mean reward stable around the mid-tier value because abstain dodges the worst
penalties. **`entropy` drops fast**, `kl` stays low (the model isn't moving
much), but the agent has stopped being useful. Adversarial split shows zero
task-completion reward.
**Action:** halt, add an explicit penalty for `abstain` on cleanly-reversible
T1 scenarios, OR rebalance reward weights so caution can't dominate.

## 5. If everything looks fine

Don't override — let it run. The point of an overnight run is that we trust
the halt rules. Take a screenshot of the W&B dashboard at 04:00, 06:00, 08:00
for the README's training-progress section.
