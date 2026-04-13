from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from config import LabConfig, get_config
from environments import ABSTAIN, ACT, INSPECT, OraclePolicy, SensorDiagnosisEnv, action_name
from models import SelectiveDecisionModel
from models.uncertainty import mc_dropout_predict
from utils.data_utils import load_dataset
from utils.plotting import plot_baseline_comparison, plot_threshold_tradeoff
from utils.seed import set_seed


@dataclass
class DecisionSnapshot:
    predicted_state: int
    confidence: float
    decision_action: int
    uncertainty: float


def _prepare_sequence(history: list[int], config: LabConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = config.environment.max_sequence_length
    pad_token = config.environment.num_observations
    sequence = np.full(max_len, pad_token, dtype=np.int64)
    sequence[: len(history)] = np.asarray(history, dtype=np.int64)
    sequence_tensor = torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)
    length_tensor = torch.as_tensor([len(history)], dtype=torch.long, device=device)
    return sequence_tensor, length_tensor


def _decision_snapshot(
    model: SelectiveDecisionModel,
    history: list[int],
    config: LabConfig,
    device: torch.device,
    cache: dict[tuple[int, ...], DecisionSnapshot] | None = None,
) -> DecisionSnapshot:
    history_key = tuple(history)
    if cache is not None and history_key in cache:
        return cache[history_key]
    sequence_tensor, length_tensor = _prepare_sequence(history, config, device)
    model.eval()
    with torch.no_grad():
        outputs = model(sequence_tensor, length_tensor)
        mc = mc_dropout_predict(model, sequence_tensor, length_tensor, config.model.mc_dropout_samples)
    probs = outputs["state_probs"][0].cpu().numpy()
    decision_action = int(outputs["decision_logits"].argmax(dim=-1)[0].cpu().item())
    snapshot = DecisionSnapshot(
        predicted_state=int(np.argmax(probs)),
        confidence=float(np.max(probs)),
        decision_action=decision_action,
        uncertainty=float(mc["entropy"][0].cpu().item()),
    )
    if cache is not None:
        cache[history_key] = snapshot
    return snapshot


def _act_vs_abstain_threshold(config: LabConfig) -> float:
    env = config.environment
    numerator = env.abstain_penalty - env.wrong_reward
    denominator = env.correct_reward - env.wrong_reward
    return float(numerator / denominator)


def _fallback_terminal_action(snapshot: DecisionSnapshot, config: LabConfig) -> int:
    threshold = _act_vs_abstain_threshold(config)
    return ACT if snapshot.confidence >= threshold else ABSTAIN


def _run_episode(
    model: SelectiveDecisionModel,
    config: LabConfig,
    policy_name: str,
    oracle: OraclePolicy,
    snapshot_cache: dict[tuple[int, ...], DecisionSnapshot],
    episode_seed: int,
    threshold: float | None = None,
) -> dict[str, float]:
    device = next(model.parameters()).device
    env = SensorDiagnosisEnv(config.environment)
    rng = np.random.default_rng(episode_seed)
    history = env.reset(rng=rng)

    total_reward = 0.0
    act_steps = 0
    inspect_steps = 0
    abstain_steps = 0
    acted_correctly = 0
    chosen_state = None

    fixed_inspect_budget = 2
    while True:
        snapshot = _decision_snapshot(model, history, config, device, cache=snapshot_cache)
        remaining = env.remaining_inspects

        if policy_name == "always_act":
            action = ACT
        elif policy_name == "fixed_inspect_then_act":
            action = INSPECT if len(history) < 1 + fixed_inspect_budget and remaining > 0 else ACT
        elif policy_name == "random_inspect":
            action = INSPECT if remaining > 0 and rng.random() < 0.45 else ACT
        elif policy_name == "learned_selective":
            action = snapshot.decision_action
            if action == INSPECT and remaining <= 0:
                action = _fallback_terminal_action(snapshot, config)
        elif policy_name == "uncertainty_threshold":
            if snapshot.uncertainty <= float(threshold):
                action = ACT
            elif remaining > 0:
                action = INSPECT
            else:
                action = ABSTAIN
        elif policy_name == "oracle":
            oracle_decision = oracle.evaluate_history(history, remaining)
            action = oracle_decision.action
            chosen_state = oracle_decision.best_state
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        if action == ACT:
            result = env.act(snapshot.predicted_state if chosen_state is None else chosen_state)
            total_reward += result.reward
            act_steps += 1
            acted_correctly += int(bool(result.info["correct"]))
            break
        if action == INSPECT:
            result = env.inspect()
            total_reward += result.reward
            inspect_steps += 1
            history.append(int(result.observation))
            continue
        result = env.abstain()
        total_reward += result.reward
        abstain_steps += 1
        break

    return {
        "total_reward": total_reward,
        "act_steps": float(act_steps),
        "inspect_steps": float(inspect_steps),
        "abstain_steps": float(abstain_steps),
        "acted_correctly": float(acted_correctly),
    }


def _aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    act_steps = np.asarray([record["act_steps"] for record in records], dtype=float)
    inspect_steps = np.asarray([record["inspect_steps"] for record in records], dtype=float)
    abstain_steps = np.asarray([record["abstain_steps"] for record in records], dtype=float)
    acted_correctly = np.asarray([record["acted_correctly"] for record in records], dtype=float)
    act_episodes = np.maximum(act_steps.sum(), 1.0)
    total_episodes = max(len(records), 1)
    return {
        "average_reward": float(np.mean([record["total_reward"] for record in records])),
        "act_accuracy": float(acted_correctly.sum() / act_episodes),
        "act_rate": float(act_steps.sum() / total_episodes),
        "inspect_rate": float(inspect_steps.sum() / total_episodes),
        "abstain_rate": float(abstain_steps.sum() / total_episodes),
    }


def _all_histories(num_observations: int, max_length: int) -> list[list[int]]:
    histories: list[list[int]] = []
    for length in range(1, max_length + 1):
        histories.extend([list(history) for history in product(range(num_observations), repeat=length)])
    return histories


def _oracle_alignment_summary(
    model: SelectiveDecisionModel,
    config: LabConfig,
    oracle: OraclePolicy,
    snapshot_cache: dict[tuple[int, ...], DecisionSnapshot],
) -> dict[str, object]:
    device = next(model.parameters()).device
    histories = _all_histories(config.environment.num_observations, config.environment.max_sequence_length)

    action_matches = 0
    state_matches = 0
    acted_state_matches = 0
    learned_act_count = 0
    disagreements: list[dict[str, object]] = []

    for history in histories:
        remaining = config.environment.max_sequence_length - len(history)
        snapshot = _decision_snapshot(model, history, config, device, cache=snapshot_cache)
        oracle_decision = oracle.evaluate_history(history, remaining)
        action_match = snapshot.decision_action == oracle_decision.action
        state_match = snapshot.predicted_state == oracle_decision.best_state

        action_matches += int(action_match)
        state_matches += int(state_match)
        if snapshot.decision_action == ACT:
            learned_act_count += 1
            acted_state_matches += int(state_match)

        if not action_match and len(disagreements) < 8:
            disagreements.append(
                {
                    "history": history,
                    "learned_action": action_name(snapshot.decision_action),
                    "oracle_action": action_name(oracle_decision.action),
                    "predicted_state": snapshot.predicted_state,
                    "oracle_state": oracle_decision.best_state,
                    "confidence": snapshot.confidence,
                    "uncertainty": snapshot.uncertainty,
                }
            )

    dataset = load_dataset(config.paths.dataset_path)["test"]
    test_histories = {
        tuple(dataset["sequences"][idx][: int(dataset["lengths"][idx])].tolist())
        for idx in range(len(dataset["hidden_states"]))
    }

    return {
        "history_space_size": len(histories),
        "test_unique_histories": len(test_histories),
        "action_match_rate": float(action_matches / len(histories)),
        "state_match_rate": float(state_matches / len(histories)),
        "acted_state_match_rate": float(acted_state_matches / max(learned_act_count, 1)),
        "num_disagreements_logged": len(disagreements),
        "sample_disagreements": disagreements,
    }


def evaluate_policies(config: LabConfig) -> dict[str, dict[str, float]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelectiveDecisionModel(config.environment, config.model).to(device)
    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    oracle = OraclePolicy(config.environment)
    snapshot_cache: dict[tuple[int, ...], DecisionSnapshot] = {}
    episode_seeds = [config.seed + 11 + idx for idx in range(config.analysis.policy_eval_episodes)]
    policies = ["always_act", "fixed_inspect_then_act", "random_inspect", "learned_selective", "oracle"]
    summary: dict[str, dict[str, float]] = {}

    for policy_name in policies:
        records = [
            _run_episode(
                model,
                config,
                policy_name=policy_name,
                oracle=oracle,
                snapshot_cache=snapshot_cache,
                episode_seed=episode_seed,
            )
            for episode_seed in episode_seeds
        ]
        summary[policy_name] = _aggregate(records)

    thresholds = np.linspace(0.05, np.log(config.environment.num_states), config.analysis.threshold_grid_size)
    threshold_records = {
        "average_reward": [],
        "act_accuracy": [],
        "act_rate": [],
        "inspect_rate": [],
        "abstain_rate": [],
    }
    for threshold in thresholds:
        records = [
            _run_episode(
                model,
                config,
                policy_name="uncertainty_threshold",
                oracle=oracle,
                snapshot_cache=snapshot_cache,
                episode_seed=episode_seed,
                threshold=float(threshold),
            )
            for episode_seed in episode_seeds
        ]
        aggregated = _aggregate(records)
        for key in threshold_records:
            threshold_records[key].append(aggregated[key])

    plot_baseline_comparison(
        policy_names=policies,
        records={
            "average_reward": [summary[name]["average_reward"] for name in policies],
            "act_accuracy": [summary[name]["act_accuracy"] for name in policies],
            "inspect_rate": [summary[name]["inspect_rate"] for name in policies],
            "abstain_rate": [summary[name]["abstain_rate"] for name in policies],
        },
        path=config.paths.figures_dir / "baseline_policy_comparison.png",
    )
    plot_threshold_tradeoff(
        thresholds=thresholds,
        records={key: np.asarray(value, dtype=float) for key, value in threshold_records.items()},
        path=config.paths.figures_dir / "threshold_tradeoff.png",
    )

    output = {
        "policies": summary,
        "oracle_alignment": _oracle_alignment_summary(
            model=model,
            config=config,
            oracle=oracle,
            snapshot_cache=snapshot_cache,
        ),
        "threshold_sweep": {
            "thresholds": thresholds.tolist(),
            **{key: list(map(float, value)) for key, value in threshold_records.items()},
        },
    }
    with config.paths.policy_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective decision policies.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = get_config(seed=args.seed)
    results = evaluate_policies(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
