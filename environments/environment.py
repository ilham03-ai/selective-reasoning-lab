from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EnvironmentConfig


@dataclass
class StepResult:
    observation: int | None
    reward: float
    done: bool
    info: dict


class SensorDiagnosisEnv:
    """A tiny partially observable environment with optional information gathering."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng = np.random.default_rng()
        self.hidden_state: int | None = None
        self.observations: list[int] = []
        self.remaining_inspects = config.max_inspects
        self.done = False

    def reset(self, rng: np.random.Generator | None = None, hidden_state: int | None = None) -> list[int]:
        if rng is not None:
            self.rng = rng
        self.hidden_state = int(hidden_state) if hidden_state is not None else self._sample_hidden_state()
        self.observations = []
        self.remaining_inspects = self.config.max_inspects
        self.done = False
        for _ in range(self.config.initial_observations):
            self.observations.append(self._sample_observation())
        return list(self.observations)

    def inspect(self) -> StepResult:
        if self.done:
            raise RuntimeError("Episode already terminated.")
        if self.remaining_inspects <= 0:
            raise RuntimeError("No remaining inspect actions.")
        self.remaining_inspects -= 1
        observation = self._sample_observation()
        self.observations.append(observation)
        return StepResult(
            observation=observation,
            reward=-self.config.inspect_cost,
            done=False,
            info={"remaining_inspects": self.remaining_inspects},
        )

    def act(self, predicted_state: int) -> StepResult:
        if self.done:
            raise RuntimeError("Episode already terminated.")
        self.done = True
        correct = int(predicted_state == self.hidden_state)
        reward = self.config.correct_reward if correct else self.config.wrong_reward
        return StepResult(
            observation=None,
            reward=reward,
            done=True,
            info={"correct": bool(correct), "hidden_state": int(self.hidden_state)},
        )

    def abstain(self) -> StepResult:
        if self.done:
            raise RuntimeError("Episode already terminated.")
        self.done = True
        return StepResult(
            observation=None,
            reward=self.config.abstain_penalty,
            done=True,
            info={"correct": None, "hidden_state": int(self.hidden_state)},
        )

    def _sample_hidden_state(self) -> int:
        return int(self.rng.choice(self.config.num_states, p=np.asarray(self.config.prior, dtype=float)))

    def _sample_observation(self) -> int:
        if self.hidden_state is None:
            raise RuntimeError("Environment must be reset before sampling observations.")
        probs = np.asarray(self.config.observation_matrix[self.hidden_state], dtype=float)
        return int(self.rng.choice(self.config.num_observations, p=probs))
