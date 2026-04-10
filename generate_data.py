from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from config import LabConfig, get_config
from environments import OraclePolicy, counts_from_history
from utils.data_utils import save_dataset
from utils.seed import set_seed


def _sample_full_trajectory(hidden_state: int, config: LabConfig, rng: np.random.Generator) -> list[int]:
    max_len = config.environment.max_sequence_length
    obs_probs = np.asarray(config.environment.observation_matrix[hidden_state], dtype=float)
    return [int(rng.choice(config.environment.num_observations, p=obs_probs)) for _ in range(max_len)]


def _build_split(num_episodes: int, config: LabConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    env_cfg = config.environment
    oracle = OraclePolicy(env_cfg)
    pad_token = env_cfg.num_observations
    max_len = env_cfg.max_sequence_length

    records: dict[str, list[Any]] = {
        "sequences": [],
        "lengths": [],
        "hidden_states": [],
        "oracle_actions": [],
        "oracle_best_states": [],
        "oracle_posteriors": [],
        "oracle_entropies": [],
        "oracle_action_values": [],
        "remaining_inspects": [],
        "trajectory_ids": [],
        "prefix_steps": [],
    }

    for episode_idx in range(num_episodes):
        hidden_state = int(rng.choice(env_cfg.num_states, p=np.asarray(env_cfg.prior, dtype=float)))
        trajectory = _sample_full_trajectory(hidden_state, config, rng)
        for prefix_len in range(1, max_len + 1):
            prefix = trajectory[:prefix_len]
            remaining = max_len - prefix_len
            counts = counts_from_history(prefix, env_cfg.num_observations)
            decision = oracle.evaluate_counts(counts, remaining)
            padded = np.full(max_len, pad_token, dtype=np.int64)
            padded[:prefix_len] = np.asarray(prefix, dtype=np.int64)
            records["sequences"].append(padded)
            records["lengths"].append(prefix_len)
            records["hidden_states"].append(hidden_state)
            records["oracle_actions"].append(decision.action)
            records["oracle_best_states"].append(decision.best_state)
            records["oracle_posteriors"].append(decision.posterior)
            records["oracle_entropies"].append(decision.state_entropy)
            records["oracle_action_values"].append(decision.action_values)
            records["remaining_inspects"].append(remaining)
            records["trajectory_ids"].append(episode_idx)
            records["prefix_steps"].append(prefix_len)

    return {key: np.asarray(value) for key, value in records.items()}


def generate_dataset(config: LabConfig) -> dict[str, Any]:
    train_rng = np.random.default_rng(config.seed)
    val_rng = np.random.default_rng(config.seed + 1)
    test_rng = np.random.default_rng(config.seed + 2)
    dataset = {
        "train": _build_split(config.training.train_episodes, config, train_rng),
        "val": _build_split(config.training.val_episodes, config, val_rng),
        "test": _build_split(config.training.test_episodes, config, test_rng),
        "metadata": config.to_dict(),
    }
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline trajectories for selective reasoning.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = get_config(seed=args.seed)
    set_seed(config.seed)
    dataset = generate_dataset(config)
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(config.paths.dataset_path, dataset)
    print(f"Saved dataset to {config.paths.dataset_path}")


if __name__ == "__main__":
    main()
