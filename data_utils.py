from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset


def to_tensor_dataset(split: dict[str, np.ndarray]) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(split["sequences"], dtype=torch.long),
        torch.as_tensor(split["lengths"], dtype=torch.long),
        torch.as_tensor(split["hidden_states"], dtype=torch.long),
        torch.as_tensor(split["oracle_actions"], dtype=torch.long),
    )


def save_dataset(path: Path, dataset: dict[str, Any]) -> None:
    arrays = {}
    metadata = dataset["metadata"]
    for split_name in ("train", "val", "test"):
        for key, value in dataset[split_name].items():
            arrays[f"{split_name}_{key}"] = value
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


def load_dataset(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    dataset: dict[str, Any] = {"metadata": json.loads(str(loaded["metadata_json"].item()))}
    for split_name in ("train", "val", "test"):
        split = {}
        prefix = f"{split_name}_"
        for key in loaded.files:
            if key.startswith(prefix):
                split[key.removeprefix(prefix)] = loaded[key]
        dataset[split_name] = split
    return dataset
