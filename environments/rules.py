from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from config import EnvironmentConfig

ACT = 0
INSPECT = 1
ABSTAIN = 2

ACTION_NAMES = {
    ACT: "act",
    INSPECT: "inspect",
    ABSTAIN: "abstain",
}


def action_name(action_id: int) -> str:
    return ACTION_NAMES[int(action_id)]


def counts_from_history(history: list[int] | np.ndarray, num_observations: int) -> tuple[int, ...]:
    counts = np.bincount(np.asarray(history, dtype=int), minlength=num_observations)
    return tuple(int(x) for x in counts.tolist())


def posterior_from_counts(
    counts: tuple[int, ...] | np.ndarray,
    observation_matrix: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    counts_array = np.asarray(counts, dtype=float)
    log_prior = np.log(prior + 1e-12)
    log_likelihood = counts_array[None, :] * np.log(observation_matrix + 1e-12)
    log_posterior = log_prior + log_likelihood.sum(axis=1)
    stabilized = log_posterior - log_posterior.max()
    posterior = np.exp(stabilized)
    posterior /= posterior.sum()
    return posterior


@dataclass
class OracleDecision:
    action: int
    best_state: int
    posterior: np.ndarray
    action_values: np.ndarray
    state_entropy: float


class OraclePolicy:
    """Exact one-step and multi-step lookahead under the known observation model."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.observation_matrix = np.asarray(config.observation_matrix, dtype=float)
        self.prior = np.asarray(config.prior, dtype=float)

    def evaluate_counts(self, counts: tuple[int, ...], remaining_inspects: int) -> OracleDecision:
        value, action_values, best_state, posterior = self._solve(tuple(counts), int(remaining_inspects))
        del value
        posterior_array = np.asarray(posterior, dtype=float)
        action_values_array = np.asarray(action_values, dtype=float)
        entropy = float(-(posterior_array * np.log(posterior_array + 1e-12)).sum())
        return OracleDecision(
            action=int(np.argmax(action_values_array)),
            best_state=int(best_state),
            posterior=posterior_array,
            action_values=action_values_array,
            state_entropy=entropy,
        )

    def evaluate_history(self, history: list[int] | np.ndarray, remaining_inspects: int) -> OracleDecision:
        counts = counts_from_history(history, self.config.num_observations)
        return self.evaluate_counts(counts, remaining_inspects)

    @lru_cache(maxsize=None)
    def _solve(self, counts: tuple[int, ...], remaining_inspects: int) -> tuple[float, tuple[float, ...], int, tuple[float, ...]]:
        posterior = posterior_from_counts(counts, self.observation_matrix, self.prior)
        best_state = int(np.argmax(posterior))
        act_value = float(
            posterior[best_state] * self.config.correct_reward
            + (1.0 - posterior[best_state]) * self.config.wrong_reward
        )
        abstain_value = float(self.config.abstain_penalty)
        inspect_value = -np.inf

        if remaining_inspects > 0:
            predictive = posterior @ self.observation_matrix
            future_value = 0.0
            for obs_id in range(self.config.num_observations):
                next_counts = list(counts)
                next_counts[obs_id] += 1
                future_value += predictive[obs_id] * self._solve(tuple(next_counts), remaining_inspects - 1)[0]
            inspect_value = float(-self.config.inspect_cost + future_value)

        action_values = np.asarray([act_value, inspect_value, abstain_value], dtype=float)
        optimal_value = float(action_values.max())
        return (
            optimal_value,
            tuple(float(x) for x in action_values.tolist()),
            best_state,
            tuple(float(x) for x in posterior.tolist()),
        )
