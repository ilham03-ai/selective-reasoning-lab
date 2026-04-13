from __future__ import annotations

import torch


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    num_samples: int,
) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.train()
    samples = []
    for _ in range(num_samples):
        outputs = model(sequences, lengths)
        samples.append(outputs["state_probs"].unsqueeze(0))
    stacked = torch.cat(samples, dim=0)
    mean_probs = stacked.mean(dim=0)
    entropy = predictive_entropy(mean_probs)
    expected_entropy = predictive_entropy(stacked).mean(dim=0)
    mutual_information = entropy - expected_entropy
    confidence_std = stacked.max(dim=-1).values.std(dim=0)
    if not was_training:
        model.eval()
    return {
        "samples": stacked,
        "mean_probs": mean_probs,
        "entropy": entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "confidence_std": confidence_std,
    }
