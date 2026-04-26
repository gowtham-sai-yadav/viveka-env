"""FastAPI application for the Viveka environment."""

from __future__ import annotations

from typing import Any

import gradio as gr
from openenv.core.env_server.http_server import create_app

from viveka.models import VivekaAction, VivekaObservation
from viveka.server.environment import VivekaEnvironment
from viveka.server.graders import grade_episode
from viveka.server.gradio_ui import create_gradio_app
from viveka.server.scenario_loader import all_tier_dirs, list_scenarios, load_scenario_by_tier

app = create_app(
    env=VivekaEnvironment,
    action_cls=VivekaAction,
    observation_cls=VivekaObservation,
    env_name="viveka_env",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/tasks")
async def get_tasks() -> dict[str, Any]:
    tier_descriptions = {
        1: ("Easy", "Single service, single action, explicit request."),
        2: ("Medium", "Mixed reversible + irreversible actions, mild ambiguity."),
        3: ("Hard", "Hinglish, multi-step, multi-service, real ambiguity."),
        4: ("Adversarial", "Planted traps: refund-window cancellations, fraud VPAs, hardlinked deletes."),
    }
    tasks = []
    for tier_id, dir_name in all_tier_dirs().items():
        difficulty, description = tier_descriptions[tier_id]
        tasks.append({
            "tier_id": tier_id,
            "name": dir_name,
            "difficulty": difficulty,
            "num_scenarios": len(list_scenarios(dir_name)),
            "description": description,
        })
    return {
        "tasks": tasks,
        "action_schema": VivekaAction.model_json_schema(),
    }


@app.post("/grader")
async def run_grader(body: dict[str, Any]) -> dict[str, Any]:
    tier_id = int(body.get("tier_id", 1))
    scenario_idx = int(body.get("scenario_idx", 0))
    scenario = load_scenario_by_tier(tier_id, scenario_idx)
    score = grade_episode(
        scenario=scenario,
        actions_taken=body.get("actions_taken", []),
        services_state=body.get("services_state", {}),
        user_responses=body.get("user_responses", []),
        pending_confirmations=body.get("pending_confirmations", []),
        done_action_type=body.get("done_action_type"),
    )
    return {"score": score, "tier_id": tier_id, "scenario_idx": scenario_idx}


gradio_app = create_gradio_app()
# Mount at root: HF Space iframe loads `/`, so Gradio must answer there directly.
# OpenEnv's REST routes (/health, /tasks, /grader, /docs, /reset, /step, etc.)
# are registered with explicit decorators above and match before Gradio's catch-all.
app = gr.mount_gradio_app(app, gradio_app, path="/")


def main() -> None:
    import uvicorn

    uvicorn.run("viveka.server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()