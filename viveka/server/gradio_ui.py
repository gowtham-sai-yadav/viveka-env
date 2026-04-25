"""Placeholder Gradio UI for Viveka. Real trace view lands in Phase 3."""

from __future__ import annotations

import gradio as gr


def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="Viveka — Reversibility + Calibration RL") as app:
        gr.Markdown(
            "## Viveka\n\n"
            "An OpenEnv RL environment teaching agents to predict reversibility, "
            "emit calibrated confidence, and ask the user before irreversible actions.\n\n"
            "_Live trace UI coming online in Phase 3._"
        )
        gr.Markdown("**API endpoints:** `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/health`.")
    return app
