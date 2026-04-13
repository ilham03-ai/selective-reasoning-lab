from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentConfig:
    num_states: int = 3
    num_observations: int = 3
    initial_observations: int = 1
    max_inspects: int = 3
    prior: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    observation_matrix: tuple[tuple[float, ...], ...] = (
        (0.70, 0.20, 0.10),
        (0.28, 0.44, 0.28),
        (0.10, 0.20, 0.70),
    )
    correct_reward: float = 1.0
    wrong_reward: float = -2.5
    abstain_penalty: float = -0.25
    inspect_cost: float = 0.07

    @property
    def max_sequence_length(self) -> int:
        return self.initial_observations + self.max_inspects


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 8
    hidden_dim: int = 32
    dropout: float = 0.20
    mc_dropout_samples: int = 24


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    decision_loss_weight: float = 0.7
    train_episodes: int = 5000
    val_episodes: int = 1200
    test_episodes: int = 2000


@dataclass(frozen=True)
class AnalysisConfig:
    calibration_bins: int = 10
    threshold_grid_size: int = 25
    policy_eval_episodes: int = 1500
    example_trajectories: int = 4


@dataclass
class PathsConfig:
    project_root: Path
    results_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    dataset_path: Path = field(init=False)
    checkpoint_path: Path = field(init=False)
    training_history_path: Path = field(init=False)
    metrics_path: Path = field(init=False)
    policy_metrics_path: Path = field(init=False)
    uncertainty_metrics_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.results_dir = self.project_root / "results"
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.figures_dir = self.results_dir / "figures"
        self.dataset_path = self.results_dir / "dataset.npz"
        self.checkpoint_path = self.checkpoint_dir / "decision_model.pt"
        self.training_history_path = self.results_dir / "training_history.json"
        self.metrics_path = self.results_dir / "evaluation_metrics.json"
        self.policy_metrics_path = self.results_dir / "policy_metrics.json"
        self.uncertainty_metrics_path = self.results_dir / "uncertainty_metrics.json"


@dataclass
class LabConfig:
    seed: int
    environment: EnvironmentConfig
    model: ModelConfig
    training: TrainingConfig
    analysis: AnalysisConfig
    paths: PathsConfig

    def to_dict(self) -> dict[str, Any]:
        config_dict = asdict(self)
        config_dict["paths"] = {key: str(value) for key, value in vars(self.paths).items()}
        return config_dict


def get_config(seed: int = 7) -> LabConfig:
    project_root = Path(__file__).resolve().parent
    return LabConfig(
        seed=seed,
        environment=EnvironmentConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        analysis=AnalysisConfig(),
        paths=PathsConfig(project_root=project_root),
    )
