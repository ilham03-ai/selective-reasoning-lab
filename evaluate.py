from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import LabConfig, get_config
from models import SelectiveDecisionModel
from models.uncertainty import mc_dropout_predict
from utils.data_utils import load_dataset, to_tensor_dataset
from utils.metrics import classification_metrics, calibration_curve_data
from utils.plotting import plot_calibration_curve
from utils.seed import set_seed


def _collect_predictions(
    model: SelectiveDecisionModel,
    loader: DataLoader,
    config: LabConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    y_true = []
    deterministic_probs = []
    decision_true = []
    decision_pred = []
    entropies = []
    mutual_information = []
    confidence_std = []

    model.eval()
    with torch.no_grad():
        for sequences, lengths, hidden_states, oracle_actions in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            hidden_states = hidden_states.to(device)
            oracle_actions = oracle_actions.to(device)
            outputs = model(sequences, lengths)
            dropout_outputs = mc_dropout_predict(
                model=model,
                sequences=sequences,
                lengths=lengths,
                num_samples=config.model.mc_dropout_samples,
            )
            y_true.append(hidden_states.cpu().numpy())
            deterministic_probs.append(outputs["state_probs"].cpu().numpy())
            decision_true.append(oracle_actions.cpu().numpy())
            decision_pred.append(outputs["decision_logits"].argmax(dim=-1).cpu().numpy())
            entropies.append(dropout_outputs["entropy"].cpu().numpy())
            mutual_information.append(dropout_outputs["mutual_information"].cpu().numpy())
            confidence_std.append(dropout_outputs["confidence_std"].cpu().numpy())

    return {
        "y_true": np.concatenate(y_true),
        "probs": np.concatenate(deterministic_probs),
        "decision_true": np.concatenate(decision_true),
        "decision_pred": np.concatenate(decision_pred),
        "entropy": np.concatenate(entropies),
        "mutual_information": np.concatenate(mutual_information),
        "confidence_std": np.concatenate(confidence_std),
    }


def evaluate_model(config: LabConfig) -> dict[str, float]:
    set_seed(config.seed)
    dataset = load_dataset(config.paths.dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelectiveDecisionModel(config.environment, config.model).to(device)
    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(
        to_tensor_dataset(dataset["test"]),
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    collected = _collect_predictions(model, test_loader, config, device)
    metrics = classification_metrics(collected["y_true"], collected["probs"])
    metrics["decision_accuracy"] = float(
        np.mean(collected["decision_true"] == collected["decision_pred"])
    )
    confidence = collected["probs"].max(axis=1)
    predictions = collected["probs"].argmax(axis=1)
    wrong_mask = predictions != collected["y_true"]
    metrics["overconfident_error_count"] = int(np.sum(wrong_mask & (confidence >= 0.80)))
    metrics["underconfident_correct_count"] = int(np.sum((~wrong_mask) & (confidence <= 0.55)))

    with config.paths.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    calibration = calibration_curve_data(
        y_true=collected["y_true"],
        probs=collected["probs"],
        num_bins=config.analysis.calibration_bins,
    )
    plot_calibration_curve(calibration, config.paths.figures_dir / "calibration_curve.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction and calibration quality.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = get_config(seed=args.seed)
    metrics = evaluate_model(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
