from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import LabConfig, get_config
from generate_data import generate_dataset
from models import SelectiveDecisionModel
from utils.data_utils import load_dataset, save_dataset, to_tensor_dataset
from utils.plotting import plot_training_curves
from utils.seed import set_seed


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    decision_loss_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    state_correct = 0
    decision_correct = 0
    total = 0
    total_loss = 0.0
    state_criterion = nn.CrossEntropyLoss()
    decision_criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for sequences, lengths, hidden_states, oracle_actions in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            hidden_states = hidden_states.to(device)
            oracle_actions = oracle_actions.to(device)
            outputs = model(sequences, lengths)
            state_loss = state_criterion(outputs["state_logits"], hidden_states)
            decision_loss = decision_criterion(outputs["decision_logits"], oracle_actions)
            loss = state_loss + decision_loss_weight * decision_loss
            total_loss += float(loss.item()) * sequences.size(0)
            state_correct += int((outputs["state_logits"].argmax(dim=-1) == hidden_states).sum().item())
            decision_correct += int((outputs["decision_logits"].argmax(dim=-1) == oracle_actions).sum().item())
            total += sequences.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "state_accuracy": state_correct / max(total, 1),
        "decision_accuracy": decision_correct / max(total, 1),
    }


def train_model(config: LabConfig, force_regenerate: bool = False) -> tuple[SelectiveDecisionModel, dict[str, list[float]]]:
    set_seed(config.seed)
    config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    if force_regenerate or not config.paths.dataset_path.exists():
        dataset = generate_dataset(config)
        save_dataset(config.paths.dataset_path, dataset)
    else:
        dataset = load_dataset(config.paths.dataset_path)

    train_loader = DataLoader(
        to_tensor_dataset(dataset["train"]),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        to_tensor_dataset(dataset["val"]),
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelectiveDecisionModel(config.environment, config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    state_criterion = nn.CrossEntropyLoss()
    decision_criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_state_accuracy": [],
        "val_decision_accuracy": [],
    }

    best_val_loss = float("inf")
    for epoch in range(config.training.epochs):
        model.train()
        epoch_losses = []
        for sequences, lengths, hidden_states, oracle_actions in train_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            hidden_states = hidden_states.to(device)
            oracle_actions = oracle_actions.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(sequences, lengths)
            state_loss = state_criterion(outputs["state_logits"], hidden_states)
            decision_loss = decision_criterion(outputs["decision_logits"], oracle_actions)
            loss = state_loss + config.training.decision_loss_weight * decision_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = _evaluate_model(
            model=model,
            loader=val_loader,
            decision_loss_weight=config.training.decision_loss_weight,
            device=device,
        )
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_metrics["loss"])
        history["val_state_accuracy"].append(val_metrics["state_accuracy"])
        history["val_decision_accuracy"].append(val_metrics["decision_accuracy"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                },
                config.paths.checkpoint_path,
            )

    with config.paths.training_history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    plot_training_curves(history, config.paths.figures_dir / "training_curves.png")

    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the selective reasoning model.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--regenerate-data", action="store_true")
    args = parser.parse_args()

    config = get_config(seed=args.seed)
    train_model(config, force_regenerate=args.regenerate_data)
    print(f"Saved checkpoint to {config.paths.checkpoint_path}")


if __name__ == "__main__":
    main()
