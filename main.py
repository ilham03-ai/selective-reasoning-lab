from __future__ import annotations

import json

from analyze_uncertainty import analyze_uncertainty
from config import get_config
from evaluate import evaluate_model
from generate_data import generate_dataset
from policy_eval import evaluate_policies
from train import train_model
from utils.data_utils import save_dataset
from utils.seed import set_seed


def main() -> None:
    config = get_config()
    set_seed(config.seed)
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(config)
    save_dataset(config.paths.dataset_path, dataset)
    train_model(config)
    evaluation_metrics = evaluate_model(config)
    uncertainty_metrics = analyze_uncertainty(config)
    policy_metrics = evaluate_policies(config)

    summary = {
        "prediction": evaluation_metrics,
        "uncertainty": uncertainty_metrics,
        "policy": policy_metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
