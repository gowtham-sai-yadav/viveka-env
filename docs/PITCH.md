# Viveka — Pitch Scripts

> Tight, on-brand pitches for mentor rounds and the final demo. Rubric anchors: innovation 40 / storytelling 30 / reward curves 20 / pipeline 10. Numbers in `<ANGLE_BRACKETS>` fill in once `eval/holdout_eval.py` lands real values.

## 60-second pitch (mentor rounds 1 & 2)

> Viveka — Sanskrit for *the wisdom to discriminate*. We train LLM agents to predict whether an action is reversible *before* they take it, and emit a calibrated confidence on every prediction. Substrate: mocked Indian Digital Public Infrastructure — UPI for money, DigiLocker for documents, IRCTC for trains, with real NPCI/RBI error codes and business rules. Six-component reward, all deterministic state checks plus a Brier proper scoring rule on confidence — mathematically un-game-able, no LLM-as-judge anywhere. Two hero visuals: reward curve, base vs trained on the same axes, and a reliability diagram showing calibration improving alongside reward. Stack is TRL v1 GRPO + Unsloth 4-bit QLoRA on Qwen2-0.5B-Instruct, with 1.5B as a stretch run. By Sunday 5 PM you'll see a Hinglish agent that asks "are you sure?" before sending ₹50,000 to a flagged VPA — instead of guessing.

## 30-second hard pitch (mentor round 3 / demo opening)

**Primary** (when reward curve has landed):

> Agents today guess on irreversible actions. Viveka teaches them to ask. Substrate: real Indian DPI — UPI, DigiLocker, IRCTC, with real error codes (`UPI:5031` mandate cap, `IRCTC:E2032` tatkal closed). Reward: six deterministic components, including a Brier score on confidence — Adarsh-paper anti-game-able. Reward curve: random `<R>`, frozen Qwen `<Q>`, Viveka `<V>`. Reliability ECE dropped from `<X1>` to `<X2>`. T4 adversarial safety: `<S1>%` → `<S2>%`. Empty competitive lane on GitHub.

**Fallback** (curve still rising at Round 3):

> Viveka — wisdom to discriminate. Reward curve direction is clear and still climbing; let me show what's already locked: the reliability diagram. Same Qwen2-0.5B, base versus mid-training. ECE dropped from `<X1>` to `<X2>` on a sealed eval set. Calibration improving without LLM-as-judge — Adarsh's EMNLP 2025 anti-game-able territory. Six deterministic reward components, real Indian DPI error codes, Hinglish demo on the Space. Curve will land before submission; calibration story is already won.

---

## 90-second video — shot list

Voiceover at natural 1.0×, ~150 wpm. Burned-in captions for accessibility. Hook in first 3-5 sec. Total target 88 s (2 s buffer under 90 s cap). Output 1280×720, MP4 H.264, 6 Mbps + AAC 192 kbps.

| Time | Screen | Voiceover | Capture | Hero overlay |
|---|---|---|---|---|
| 0:00–0:05 | **HOOK** split-screen: Hinglish prompt "Send rs5000 to mom for medicines" → red flash "FRAUD-WATCHLIST HIT, UPI:5050" | "AI agents guess on irreversible actions. Watch." | static slide → 1-frame red flash | red badge + `UPI:5050` |
| 0:05–0:25 | Terminal A (untrained Qwen2-0.5B): same prompt → executes blindly, predicted_reversibility=`reversible` (WRONG), confidence 0.91 | "Today's frozen Qwen executes a UPI transfer to a flagged VPA. It mis-labels it reversible — at 91% confidence." | vhs `.tape` recording, zoom on confidence + reversibility | `predicted_reversibility=reversible ✗` `confidence=0.91` |
| 0:25–0:45 | Terminal B (trained Viveka) + Gradio UI: same prompt → predicts `irreversible`, conf 0.88, emits `confirm_with_user`; UI shows "Are you sure? ₹5000 to vpa@flagged" in Hinglish | "Viveka pauses, predicts irreversibility, surfaces the watchlist hit, and asks the user — in Hinglish, the user's language." | screen-cap Gradio + terminal side-by-side | `predicted_reversibility=irreversible ✓` `confidence=0.88` |
| 0:45–1:05 | Full-screen reward curve PNG (`docs/reward_curve.png`) with three lines + Ken-Burns pan-zoom | "Six deterministic reward components — including a Brier score on confidence. No LLM-as-judge. Random scores `<R>%`, frozen Qwen `<Q>%`, Viveka `<V>%`." | static PNG with motion + caption labels | `random=<R>%`, `base=<Q>%`, `viveka=<V>%` |
| 1:05–1:20 | Reliability diagram (`docs/reliability.png`) → cut to AQI delta bars (`eval/aqi_delta.png`) → cut to T4 adversarial table | "Calibration improves alongside reward. ECE drops from `<X1>` to `<X2>`. On adversarial scenarios — fraud watchlists, expired tatkal, revoked DigiLocker consent — safety jumps from `<S1>%` to `<S2>%`." | three quick cuts ~5 s each | `ECE: <X1> → <X2>`, `T4 safety: <S1>% → <S2>%` |
| 1:20–1:30 | Title card: "Viveka — wisdom to discriminate. UPI · DigiLocker · IRCTC. github.com/gowtham-sai-yadav/viveka-env" + QR code | "Viveka. The wisdom to discriminate. Repo, Space, model card linked below." | static end card; logo fade | tagline + URLs + QR |

---

## Recording protocol

Recording window: 14:00–16:00 IST on 2026-04-26. Submit at 16:00, deadline 17:00.

1. **Tools (Day 1 evening):** install [`vhs`](https://github.com/charmbracelet/vhs) (`brew install vhs`) for terminal takes, OBS Studio 30.x for Gradio capture, iMovie for stitching. Avoid Loom (heavy compression).
2. **Terminal theme (vhs `.tape` config):** `Set FontSize 22`, `Set FontFamily "JetBrains Mono"`, `Set Theme "Catppuccin Frappe"`, `Set Width 1280`, `Set Height 720`, `Set CursorBlink false`, `Set TypingSpeed 50ms`, `Set Framerate 60`.
3. **Pre-recording (13:30 IST):** regenerate `docs/reward_curve.png`, `docs/reliability.png`, `eval/aqi_delta.png` with REAL data. Lock numbers into a `recording_numbers.txt` so the voiceover script has truth values.
4. **Take 1 — untrained Qwen** (`docs/video/takes/01_untrained.tape`): pipe Hinglish prompt to frozen Qwen2-0.5B; expect blind execute. Two takes, pick clean.
5. **Take 2 — trained Viveka** (`docs/video/takes/02_viveka.tape`): same prompt to trained checkpoint; expect `confirm_with_user` with VPA fraud reason surfaced.
6. **Take 3 — Gradio UI** (OBS, browser 1280×720, Chrome window-only, hide bookmarks bar): type Hinglish prompt, click submit, capture confirm modal. Mouseposé for click rings.
7. **Take 4 — plot zoom-ins** (Keynote/iMovie Ken Burns): full-screen PNGs with annotation arrows.
8. **Voiceover:** Debashis or Gowtham (whoever has clearer Indian English on the day); record in Voice Memos with WIRED earbuds (NOT MacBook built-in). Two takes per beat.
9. **Captions:** burn in via iMovie title overlay or Submagic auto-captions.
10. **Stitch in iMovie:** voiceover track 1, video track 2, plot stills track 3, captions track 4. Skip background music (90 s is too tight).
11. **Click highlights:** OBS `Show Mouse Clicks` filter or post-edit yellow circles in iMovie at click moments.
12. **Export:** MP4 H.264, 1280×720 @ 30 fps, 6 Mbps video, AAC 192 kbps.
13. **Sanity playback:** watch on phone (small-screen test) and on a 13" laptop. Bump font size if unreadable.
14. **Upload by 15:30 IST as Unlisted YouTube;** copy link into README + Devpost. Don't make Public — comment moderation distracts during judging.
15. **Backup:** commit the `.mp4` to `docs/video/viveka_demo.mp4` and push (Git LFS if >100 MB) so YouTube outage doesn't sink submission.

**Skipped intentionally:** background music, animated logo intros, slide decks. (Devpost guidelines + 90-s budget.)

---

## Per-judge talking-point coverage (11 judges × 3 beats)

Beats: **P** = problem (0:00–0:25), **E** = evidence (0:25–1:20), **A** = ask (1:20–1:30).

| Judge | Cluster | P | E | A |
|---|---|---|---|---|
| Sanyam Bhutani | Reproducibility / TorchForge | UPI watchlist hit shown via deterministic state | Reward curve, six components, no LLM-as-judge | Repo URL — Quick Start in 5 cmds |
| Yash Marathe | Reproducibility | Same | Reward curve overlay | Repo + HF Space + model card URLs |
| Adarsh Shirawalmath | Calibration / safety | Confidence 0.91 on a wrong call | **Reliability diagram + AQI delta** | "ECE `<X1>` → `<X2>`, T4 safety `<S1>%` → `<S2>%`" |
| Deepa Dhevannan | Calibration / safety | Same | Reliability + adversarial scenarios | T4 safety jump number |
| Ayush (Red Hat) | Production | UPI:5050 surfaced as real error code | Schema-strict Pydantic in terminal output | Production-realism close |
| Parshant (Red Hat) | Production | Same | Same | Same |
| Arkadip (Red Hat) | Production | Same | Same | Same |
| Aashay Sachdeva (Sarvam) | Indic | **Hinglish prompt as hero example** | Gradio UI Hinglish confirm flow | "Real DPI, real business rules" |
| Adithya Kolavi | Indic | Hinglish + ₹50,000 framing | UPI mandate cap, IRCTC tatkal references | Indic close |
| Nilesh Pandey | Indic | Same | Same | Same |
| Soumik Rakshit | Production / agentic | Frozen-vs-trained ask-loop contrast | `confirm_with_user` action live | Repo + Space + agentic-loop call-out |

Every judge sees at least one beat from their cluster. Reliability diagram + Hinglish demo carry the heaviest cluster load (5 + 3 judges).

---

## Risk register — 14:00-16:00 recording window

| # | Risk | P | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Training run not finished by 14:00 | M | High | Pre-render template at 13:00 with `<placeholders>`; have 30-min plot-overlay swap path. Fallback voiceover script (above). Hard cut-off: if curve still flat at 14:30, pivot narrative to "calibration already locked" and feature reliability + AQI as hero. |
| 2 | Plots not generated / aqi_probe.py crashes | M | M | Run `eval/holdout_eval.py` + `eval/reward_curve.py` + `eval/aqi_probe.py` at 13:00 as smoke. If AQI fails, drop AQI bar and extend reliability shot to 8 s. Reward curve + reliability is minimum viable hero. |
| 3 | No quiet recording space at venue | H | H | Pre-book a breakout room for 14:00–14:30, OR record voiceover audio-only in a stairwell with Voice Memos + AirPods Pro mic; layer over screen captures in post. Test the spot at 13:30. |
| 4 | Mic quality bad | H | H | Use WIRED earbuds (3.5mm or USB-C), not Bluetooth (BT degrades to 8 kHz). Record 10-s test, listen on headphones BEFORE real takes. Submagic AI cleanup as fallback. |
| 5 | YouTube upload fails / processing stuck at 15:45 | L | Critical | Upload by 15:30 as Unlisted at 720p. Backup: commit `.mp4` to `docs/video/` + Git LFS; link both URLs in README. Devpost accepts a direct repo-hosted MP4 if YouTube fails. |
| 6 | Video drifts > 92 s | M | M | Timecode-check at every edit pass. Target 88 s, hard cap 92 s. |

---

## Per-judge framing notes (preserved from earlier passes)

- **Sanyam Bhutani:** lead with verifiable rewards + clean PyTorch-native GRPO loop. Drop "deterministic state checks, no LLM-as-judge" — TorchForge agenda.
- **Adarsh Shirawalmath:** lead with proper-scoring-rule on confidence + reliability diagram. His EMNLP 2025 AQI paper is *literally* about catching mis-calibrated alignment.
- **Aashay Sachdeva (Sarvam):** lead with RLVR + Indic substrate. Don't overclaim "sovereign" — frame as "real DPI APIs, real business rules."
- **Adithya Kolavi / Nilesh Pandey / Deepa Dhevannan:** lead with Indian context — UPI ₹1L mandate cap, IRCTC tatkal windows, DigiLocker consent token expiry. Real, not toy.
- **Red Hat trio (Ayush / Parshant / Arkadip):** lead with production-systems realism — error codes, schema-strict Pydantic, fault tolerance in env step loop.
- **Soumik Rakshit:** lead with agentic asking-loop and trained-vs-frozen JSON-failure rate exposure (the methodology-honesty story).
