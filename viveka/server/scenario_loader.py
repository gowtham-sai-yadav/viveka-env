"""Load and validate Viveka scenario JSON files."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from viveka.server.reversibility_registry import lookup


_TIER_DIRS = {
    1: "t1_easy",
    2: "t2_medium",
    3: "t3_hard",
    4: "t4_adversarial",
}


class GroundTruthAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_service: Literal["upi", "digilocker", "irctc", "banking", "telecom"]
    operation: str
    reversibility: Literal["reversible", "irreversible", "irreversible_trivial"]


class ExpectedSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ground_truth_action_sequence: list[GroundTruthAction] = Field(default_factory=list)
    post_state: dict[str, dict[str, Any]] = Field(default_factory=dict)
    max_steps: int = 30
    must_ask: bool = False
    must_not_execute: list[list[str]] = Field(default_factory=list)


class ScenarioSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    tier_id: int = Field(ge=1, le=4)
    title: str
    user_message: str
    user_language: Literal["en", "hi-en"]
    initial_state: dict[str, dict[str, Any]] = Field(default_factory=dict)
    user_oracle: dict[str, str] = Field(default_factory=dict)
    expected: ExpectedSchema


def validate_scenario(raw: dict[str, Any]) -> ScenarioSchema:
    parsed = ScenarioSchema.model_validate(raw)
    for idx, gt in enumerate(parsed.expected.ground_truth_action_sequence):
        try:
            true_label = lookup(gt.target_service, gt.operation)
        except KeyError as e:
            raise ValueError(
                f"Scenario '{parsed.scenario_id}' ground_truth_action_sequence[{idx}] "
                f"references unknown operation {gt.target_service}.{gt.operation}: {e}"
            ) from e
        if true_label != gt.reversibility:
            raise ValueError(
                f"Scenario '{parsed.scenario_id}' ground_truth_action_sequence[{idx}] "
                f"reversibility mismatch for {gt.target_service}.{gt.operation}: "
                f"asserted '{gt.reversibility}', registry says '{true_label}'"
            )
    return parsed


def _find_scenarios_dir() -> Path:
    source_dir = Path(__file__).resolve().parent.parent / "scenarios"
    if source_dir.exists() and any(source_dir.iterdir()):
        return source_dir
    try:
        pkg = importlib.resources.files("viveka.scenarios")
        pkg_path = Path(str(pkg))
        if pkg_path.exists():
            return pkg_path
    except (ImportError, TypeError):
        pass
    return source_dir


SCENARIOS_DIR = _find_scenarios_dir()


def list_scenarios(tier_dir: str) -> list[Path]:
    d = SCENARIOS_DIR / tier_dir
    if not d.exists():
        return []
    return sorted(d.glob("scenario_*.json"))


def _validate_or_raise(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    try:
        validate_scenario(raw)
    except ValidationError as e:
        raise ValidationError.from_exception_data(
            title=f"{path}: {e.title}",
            line_errors=e.errors(),  # type: ignore[arg-type]
        ) from e
    return raw


def load_scenario(tier_dir: str, scenario_name: str) -> dict[str, Any]:
    path = SCENARIOS_DIR / tier_dir / f"{scenario_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    try:
        validate_scenario(raw)
    except ValidationError as e:
        raise ValueError(f"{path}: scenario failed schema validation: {e}") from e
    return raw


def load_scenario_by_tier(tier_id: int, scenario_idx: int = 0) -> dict[str, Any]:
    tier_dir = _TIER_DIRS.get(tier_id)
    if tier_dir is None:
        raise ValueError(f"Invalid tier_id: {tier_id}. Must be 1-4.")
    scenarios = list_scenarios(tier_dir)
    if not scenarios:
        raise FileNotFoundError(f"No scenarios found for {tier_dir}")
    if scenario_idx >= len(scenarios):
        raise IndexError(f"Scenario index {scenario_idx} out of range (have {len(scenarios)})")
    path = scenarios[scenario_idx]
    with open(path) as f:
        raw = json.load(f)
    try:
        validate_scenario(raw)
    except ValidationError as e:
        raise ValueError(f"{path}: scenario failed schema validation: {e}") from e
    return raw


def all_tier_dirs() -> dict[int, str]:
    return dict(_TIER_DIRS)
