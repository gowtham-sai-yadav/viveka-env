"""Load and validate Viveka scenario JSON files."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any


_TIER_DIRS = {
    1: "t1_easy",
    2: "t2_medium",
    3: "t3_hard",
    4: "t4_adversarial",
}


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


def load_scenario(tier_dir: str, scenario_name: str) -> dict[str, Any]:
    path = SCENARIOS_DIR / tier_dir / f"{scenario_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_scenario_by_tier(tier_id: int, scenario_idx: int = 0) -> dict[str, Any]:
    tier_dir = _TIER_DIRS.get(tier_id)
    if tier_dir is None:
        raise ValueError(f"Invalid tier_id: {tier_id}. Must be 1-4.")
    scenarios = list_scenarios(tier_dir)
    if not scenarios:
        raise FileNotFoundError(f"No scenarios found for {tier_dir}")
    if scenario_idx >= len(scenarios):
        raise IndexError(f"Scenario index {scenario_idx} out of range (have {len(scenarios)})")
    with open(scenarios[scenario_idx]) as f:
        return json.load(f)


def all_tier_dirs() -> dict[int, str]:
    return dict(_TIER_DIRS)
