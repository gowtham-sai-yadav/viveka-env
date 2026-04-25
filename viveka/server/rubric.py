"""VivekaRubric — trajectory rubric wrapping the episode grader."""

from __future__ import annotations

from typing import Any, List, Tuple

from openenv.core.rubrics.trajectory import TrajectoryRubric

from viveka.server.graders import grade_episode


class VivekaRubric(TrajectoryRubric):
    def __init__(self):
        super().__init__(intermediate_reward=0.0)
        self._env_ref: Any = None

    def set_env(self, env: Any) -> None:
        self._env_ref = env

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if self._env_ref is None:
            return 0.0
        env = self._env_ref
        return grade_episode(
            scenario=env._scenario,
            actions_taken=env._actions_taken,
            services_state=env._snapshot_services(),
            user_responses=env._user_responses,
            pending_confirmations=[pc.model_dump() for pc in env._pending_confirmations],
            done_action_type=env._done_action_type,
        )

    def compute_step_rewards(self) -> List[float]:
        if not self._trajectory:
            return []
        final_score = self.score_trajectory(self._trajectory)
        n = len(self._trajectory)
        return [final_score / n] * n

    def reset(self) -> None:
        super().reset()
