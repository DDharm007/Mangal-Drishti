"""Reusable PyTorch DataLoader factories for MANGAL DRISHTI."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.config_utils import get_project_root, load_config
from src.data.dataset import MangalDrishtiDataset


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python stdlib RNGs deterministically per DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: MangalDrishtiDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build a reproducible PyTorch DataLoader with safe hardware defaults."""
    use_pin = bool(pin_memory and torch.cuda.is_available())
    use_persistent = bool(persistent_workers and num_workers > 0)

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": use_pin,
        "drop_last": bool(drop_last),
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = use_persistent

    return DataLoader(**kwargs)


def build_dataloaders(
    config_path: Path | str = Path("configs/data_config.yaml"),
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, DataLoader]:
    """Instantiate train, validation, and test PyTorch DataLoaders from YAML config."""
    root = project_root or get_project_root()
    cfg = load_config(config_path)
    dl_cfg = cfg["dataloader"]
    seed = int(cfg["project"]["random_seed"])

    bs = int(batch_size if batch_size is not None else dl_cfg["batch_size"])
    nw = int(num_workers if num_workers is not None else dl_cfg["num_workers"])
    pin = bool(dl_cfg.get("pin_memory", True))
    pers = bool(dl_cfg.get("persistent_workers", False))

    datasets = {
        split: MangalDrishtiDataset(
            split=split,
            config_path=config_path,
            project_root=root,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        "train": create_dataloader(
            datasets["train"],
            batch_size=bs,
            shuffle=bool(dl_cfg.get("shuffle_train", True)),
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=pers,
            drop_last=bool(dl_cfg.get("drop_last_train", False)),
            seed=seed,
        ),
        "val": create_dataloader(
            datasets["val"],
            batch_size=bs,
            shuffle=bool(dl_cfg.get("shuffle_val", False)),
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=pers,
            drop_last=False,
            seed=seed,
        ),
        "test": create_dataloader(
            datasets["test"],
            batch_size=bs,
            shuffle=bool(dl_cfg.get("shuffle_test", False)),
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=pers,
            drop_last=False,
            seed=seed,
        ),
    }
    return loaders
