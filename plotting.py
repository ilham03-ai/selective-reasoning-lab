from __future__ import annotations

import os
from pathlib import Path

_MPL_DIR = Path(__file__).resolve().parents[1] / "results" / ".matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finalize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _annotate_bars(ax: plt.Axes, values: np.ndarray, fmt: str = "{:.2f}") -> None:
    span = max(np.max(values) - np.min(values), 1e-6)
    offset = 0.04 * span
    for idx, value in enumerate(values):
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(idx, y, fmt.format(value), ha="center", va=va, fontsize=8)


def plot_training_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["val_state_accuracy"], label="state acc")
    axes[1].plot(epochs, history["val_decision_accuracy"], label="decision acc")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    _finalize(path)


def plot_uncertainty_vs_error(curve: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = curve["counts"]
    ax.plot(curve["uncertainty_mean"], curve["error_rate"], marker="o", linewidth=2)
    ax.scatter(curve["uncertainty_mean"], curve["error_rate"], s=25 + counts * 0.3, alpha=0.8)
    for x, y, count in zip(curve["uncertainty_mean"], curve["error_rate"], counts.astype(int)):
        ax.text(x, y + 0.015, f"n={count}", ha="center", fontsize=8)
    ax.set_xlabel("Mean uncertainty")
    ax.set_ylabel("Error rate")
    ax.set_title("Uncertainty vs Error")
    ax.grid(alpha=0.25, linestyle="--")
    _finalize(path)


def plot_calibration_curve(curve: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(curve["bin_confidence"], curve["bin_accuracy"], marker="o")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _finalize(path)


def plot_threshold_tradeoff(thresholds: np.ndarray, records: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(thresholds, records["average_reward"], label="avg reward")
    axes[0].plot(thresholds, records["act_accuracy"], label="act accuracy")
    axes[0].set_xlabel("Uncertainty threshold")
    axes[0].set_title("Utility and Accuracy")
    axes[0].legend()
    axes[1].plot(thresholds, records["act_rate"], label="act rate")
    axes[1].plot(thresholds, records["inspect_rate"], label="inspect rate")
    axes[1].plot(thresholds, records["abstain_rate"], label="abstain rate")
    axes[1].set_xlabel("Uncertainty threshold")
    axes[1].set_title("Policy Mix")
    axes[1].legend()
    _finalize(path)


def plot_baseline_comparison(policy_names: list[str], records: dict[str, list[float]], path: Path) -> None:
    x = np.arange(len(policy_names))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    reward = np.asarray(records["average_reward"], dtype=float)
    act_accuracy = np.asarray(records["act_accuracy"], dtype=float)
    inspect_rate = np.asarray(records["inspect_rate"], dtype=float)
    abstain_rate = np.asarray(records["abstain_rate"], dtype=float)

    axes[0].bar(x, reward, color="#4c78a8")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Average Reward")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policy_names, rotation=20)
    _annotate_bars(axes[0], reward)

    axes[1].bar(x, act_accuracy, color="#f58518")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Accuracy When Acting")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policy_names, rotation=20)
    _annotate_bars(axes[1], act_accuracy)

    axes[2].bar(x, inspect_rate, color="#54a24b")
    axes[2].set_title("Inspect Rate")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(policy_names, rotation=20)
    _annotate_bars(axes[2], inspect_rate)

    axes[3].bar(x, abstain_rate, color="#e45756")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_title("Abstain Rate")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(policy_names, rotation=20)
    _annotate_bars(axes[3], abstain_rate)

    fig.suptitle("Policy Tradeoff Comparison", fontsize=16, y=1.02)
    _finalize(path)


def plot_trajectory_examples(examples: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(len(examples), 1, figsize=(8, 2.5 * len(examples)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, example in zip(axes, examples):
        posteriors = np.asarray(example["posterior_trace"], dtype=float)
        steps = np.arange(1, len(posteriors) + 1)
        for state_idx in range(posteriors.shape[1]):
            ax.plot(steps, posteriors[:, state_idx], marker="o", label=f"state {state_idx}")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("belief")
        ax.set_title(
            f"hidden={example['hidden_state']} obs={example['observations']} final={example['final_action_name']}"
        )
    axes[-1].set_xlabel("Observation prefix length")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper right")
    _finalize(path)


def plot_failure_case(example: dict, path: Path) -> None:
    steps = np.arange(1, len(example["model_confidence_trace"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    model_probs = np.asarray(example["model_prob_trace"], dtype=float)
    for state_idx in range(model_probs.shape[1]):
        axes[0].plot(steps, model_probs[:, state_idx], marker="o", label=f"model state {state_idx}")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Model Class Probabilities")
    axes[0].set_xlabel("Prefix length")
    axes[0].legend()

    axes[1].plot(steps, example["model_confidence_trace"], marker="o", label="confidence")
    axes[1].plot(steps, example["uncertainty_trace"], marker="o", label="entropy")
    axes[1].set_title("Confidence and Uncertainty")
    axes[1].set_xlabel("Prefix length")
    axes[1].legend()

    posteriors = np.asarray(example["posterior_trace"], dtype=float)
    for state_idx in range(posteriors.shape[1]):
        axes[2].plot(steps, posteriors[:, state_idx], marker="o", label=f"oracle state {state_idx}")
    axes[2].set_title(
        f"Oracle Posterior\nhidden={example['hidden_state']} predicted={example['predicted_state']}"
    )
    axes[2].set_xlabel("Prefix length")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend()
    fig.suptitle(
        f"Failure Case | observations={example['observations']} | final confidence={example['model_confidence_trace'][-1]:.2f}",
        fontsize=14,
        y=1.02,
    )
    _finalize(path)
