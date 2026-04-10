from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from config import LabConfig, get_config
from environments import OraclePolicy, action_name
from models import SelectiveDecisionModel
from models.uncertainty import mc_dropout_predict
from utils.data_utils import load_dataset
from utils.metrics import selective_error_statistics
from utils.plotting import plot_failure_case, plot_trajectory_examples, plot_uncertainty_vs_error
from utils.seed import set_seed


def _single_prefix_prediction(
    model: SelectiveDecisionModel,
    sequence: np.ndarray,
    length: int,
    config: LabConfig,
    device: torch.device,
) -> dict[str, np.ndarray | float | int]:
    sequence_tensor = torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)
    length_tensor = torch.as_tensor([length], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        outputs = model(sequence_tensor, length_tensor)
        mc = mc_dropout_predict(
            model=model,
            sequences=sequence_tensor,
            lengths=length_tensor,
            num_samples=config.model.mc_dropout_samples,
        )
    probs = outputs["state_probs"][0].cpu().numpy()
    mean_probs = mc["mean_probs"][0].cpu().numpy()
    entropy = float(mc["entropy"][0].cpu().item())
    return {
        "predicted_state": int(np.argmax(probs)),
        "confidence": float(np.max(probs)),
        "mean_probs": mean_probs,
        "entropy": entropy,
    }


def analyze_uncertainty(config: LabConfig) -> dict[str, float]:
    set_seed(config.seed)
    dataset = load_dataset(config.paths.dataset_path)["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelectiveDecisionModel(config.environment, config.model).to(device)
    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    oracle = OraclePolicy(config.environment)

    batch_size = config.training.batch_size
    y_true = []
    probs = []
    entropy = []
    for start in range(0, len(dataset["hidden_states"]), batch_size):
        end = start + batch_size
        sequences = torch.as_tensor(dataset["sequences"][start:end], dtype=torch.long, device=device)
        lengths = torch.as_tensor(dataset["lengths"][start:end], dtype=torch.long, device=device)
        y_true.append(dataset["hidden_states"][start:end])
        with torch.no_grad():
            outputs = model(sequences, lengths)
            mc = mc_dropout_predict(model, sequences, lengths, config.model.mc_dropout_samples)
        probs.append(outputs["state_probs"].cpu().numpy())
        entropy.append(mc["entropy"].cpu().numpy())
    y_true_array = np.concatenate(y_true)
    probs_array = np.concatenate(probs)
    entropy_array = np.concatenate(entropy)

    curve = selective_error_statistics(y_true_array, probs_array, entropy_array)
    plot_uncertainty_vs_error(curve, config.paths.figures_dir / "uncertainty_vs_error.png")

    predictions = probs_array.argmax(axis=1)
    confidence = probs_array.max(axis=1)
    wrong_mask = predictions != y_true_array
    full_length_mask = dataset["prefix_steps"] == config.environment.max_sequence_length
    overconfident_wrong = np.where(full_length_mask & wrong_mask & (confidence >= 0.80))[0]
    if len(overconfident_wrong) == 0:
        candidate_indices = np.where(full_length_mask & wrong_mask)[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.where(wrong_mask)[0]
        ranked = candidate_indices[np.argsort(confidence[candidate_indices])[::-1]]
        overconfident_wrong = ranked[:1]

    failure_idx = int(overconfident_wrong[0])
    trajectory_id = int(dataset["trajectory_ids"][failure_idx])
    same_trajectory = np.where(dataset["trajectory_ids"] == trajectory_id)[0]
    prefix_indices = same_trajectory[np.argsort(dataset["prefix_steps"][same_trajectory])]

    failure_trace = {
        "hidden_state": int(dataset["hidden_states"][failure_idx]),
        "predicted_state": int(predictions[failure_idx]),
        "observations": dataset["sequences"][failure_idx][: int(dataset["lengths"][failure_idx])].tolist(),
        "posterior_trace": [],
        "model_prob_trace": [],
        "model_confidence_trace": [],
        "uncertainty_trace": [],
    }
    for idx in prefix_indices:
        prefix_len = int(dataset["lengths"][idx])
        prefix = dataset["sequences"][idx][:prefix_len].tolist()
        oracle_decision = oracle.evaluate_history(prefix, config.environment.max_sequence_length - prefix_len)
        pred = _single_prefix_prediction(model, dataset["sequences"][idx], prefix_len, config, device)
        failure_trace["posterior_trace"].append(oracle_decision.posterior.tolist())
        failure_trace["model_prob_trace"].append(pred["mean_probs"].tolist())
        failure_trace["model_confidence_trace"].append(pred["confidence"])
        failure_trace["uncertainty_trace"].append(pred["entropy"])
    plot_failure_case(failure_trace, config.paths.figures_dir / "failure_case.png")

    example_trajectory_ids = np.unique(dataset["trajectory_ids"])[: config.analysis.example_trajectories]
    examples = []
    for trajectory_id in example_trajectory_ids:
        indices = np.where(dataset["trajectory_ids"] == trajectory_id)[0]
        indices = indices[np.argsort(dataset["prefix_steps"][indices])]
        final_index = int(indices[-1])
        example = {
            "hidden_state": int(dataset["hidden_states"][final_index]),
            "observations": dataset["sequences"][final_index][: int(dataset["lengths"][final_index])].tolist(),
            "posterior_trace": [],
            "final_action_name": action_name(int(dataset["oracle_actions"][final_index])),
        }
        for idx in indices:
            prefix_len = int(dataset["lengths"][idx])
            prefix = dataset["sequences"][idx][:prefix_len].tolist()
            oracle_decision = oracle.evaluate_history(prefix, config.environment.max_sequence_length - prefix_len)
            example["posterior_trace"].append(oracle_decision.posterior.tolist())
        examples.append(example)
    plot_trajectory_examples(examples, config.paths.figures_dir / "trajectory_examples.png")

    summary = {
        "mean_entropy": float(entropy_array.mean()),
        "overconfident_wrong_cases": int(len(overconfident_wrong)),
        "failure_case_index": failure_idx,
    }
    with config.paths.uncertainty_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze uncertainty quality and failure cases.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = get_config(seed=args.seed)
    summary = analyze_uncertainty(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
