# Viveka — Pitch Scripts

> Tight, on-brand pitches to use in mentor rounds and the final demo. Edit to your voice but keep the rubric anchors (innovation 40 / storytelling 30 / reward curves 20 / pipeline 10).

## 60-second pitch (mentor rounds 1 & 2)

> Viveka — Sanskrit for *the wisdom to discriminate*. We train LLM agents to predict whether an action is reversible *before* they take it, and emit a calibrated confidence on every prediction. The substrate is mocked Indian Digital Public Infrastructure: UPI for money, DigiLocker for documents, IRCTC for trains. Six-component reward, all deterministic state checks plus a Brier proper scoring rule on confidence — mathematically un-gameable, no LLM-as-judge. Two hero visuals: reward curve, base vs trained on the same axes, and a reliability diagram showing calibration improving alongside reward. Stack is TRL GRPO + Unsloth 4-bit on Qwen2-0.5B, with 1.5B as a stretch run. By Sunday 5 PM you'll see a Hinglish agent that asks "are you sure?" before sending ₹50,000 to a flagged VPA — instead of guessing.

## 30-second hard pitch (mentor round 3 / demo opening)

> Agents today guess on irreversible actions. Viveka teaches them to ask. Substrate: real Indian DPI — UPI, DigiLocker, IRCTC. Reward: six deterministic components, including a Brier score on confidence. Outputs: reward curve and reliability diagram. Trained on Qwen2-0.5B with TRL GRPO. Empty competitive lane on GitHub.

## 90-second video script (final submission)

- **0–15s:** problem framing — "AI nearly sends ₹50,000 to wrong UPI ID. AI deletes wrong file. AI cancels wrong train. The shared failure: agents can't tell what's reversible."
- **15–45s:** live demo — same Hinglish prompt to untrained Qwen vs trained Viveka. Untrained executes blindly. Viveka pauses, predicts irreversibility, asks the user.
- **45–70s:** results — reward curve with random/untrained baseline overlay (innovation 20%), reliability diagram before/after (calibration improves alongside reward).
- **70–90s:** pitch line + URLs — "Viveka. The wisdom to discriminate. github.com/gowtham-sai-yadav/viveka-env"

## Per-judge framing notes

- **Sanyam Bhutani (presenter + judge):** lead with verifiable rewards + clean PyTorch-native GRPO loop. Drop the "deterministic state checks, no LLM-as-judge" line — directly anti-game-able like his TorchForge agenda.
- **Adarsh Shirawalmath:** lead with proper-scoring-rule on confidence + reliability diagram. His EMNLP 2025 AQI paper is literally about catching mis-calibrated alignment.
- **Aashay Sachdeva (Sarvam):** lead with RLVR + Indic substrate. Don't overclaim "sovereign" — frame as "real DPI APIs, real business rules."
- **Adithya Kolavi / Nilesh Pandey / Deepa Dhevannan:** lead with Indian context — UPI ₹1L mandate cap, IRCTC tatkal windows, DigiLocker consent token expiry. Real, not toy.
- **Red Hat trio (Ayush / Parshant / Arkadip):** lead with production-systems realism — error codes, schema-strict Pydantic, fault tolerance in the env step loop.
- **Soumik Rakshit (ml-intern):** lead with self-curriculum if it lands as stretch; otherwise the agentic asking-loop.
