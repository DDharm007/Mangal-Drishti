"""Comprehensive automated test suite for MANGAL DRISHTI Step 2 preprocessing.

Tests mask encoding invariants, nearest-neighbor resizing, geometric vs photometric
augmentation safety, leakage-free SCLK splitting, training-only normalization &
class weights, idempotency, and PyTorch Dataset/DataLoader/Loss compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.dataloader import build_dataloaders
from src.data.dataset import MangalDrishtiDataset
from src.data.mask_encoder import MaskEncoder
from src.data.preprocessing import run_preprocessing
from src.data.transforms import SegmentationTransformPipeline


@pytest.fixture(scope="session")
def project_root() -> Path:
    return get_project_root()


@pytest.fixture(scope="session")
def cfg(project_root: Path) -> dict:
    return load_config(project_root / "configs" / "data_config.yaml")


def test_mask_encoder_nearest_neighbor_no_interpolation_artifacts(
    cfg: dict, project_root: Path
) -> None:
    """Verify resizing categorical masks never introduces intermediate values like 1.5 -> 1/2."""
    encoder = MaskEncoder(
        label_keys_path=resolve_path(cfg["paths"]["raw_label_keys"], project_root),
        ignore_index=int(cfg["mask"]["ignore_index"]),
    )
    # Create a synthetic high-contrast checkerboard of {0, 1, 2, 3, 255}
    raw = np.zeros((1024, 1024), dtype=np.uint8)
    raw[:512, :512] = 0
    raw[:512, 512:] = 1
    raw[512:, :512] = 2
    raw[512:, 512:] = 3
    raw[256:768, 256:768] = 255

    for sz in ((256, 256), (512, 512), (768, 768)):
        out = encoder.encode_and_resize(raw, target_size=sz)
        assert out.shape == sz
        assert out.dtype == np.uint8
        assert set(np.unique(out).tolist()).issubset({0, 1, 2, 3, 255})


def test_mask_encoder_rejects_invalid_pixel_values(cfg: dict, project_root: Path) -> None:
    encoder = MaskEncoder(
        label_keys_path=resolve_path(cfg["paths"]["raw_label_keys"], project_root),
        ignore_index=int(cfg["mask"]["ignore_index"]),
    )
    bad_mask = np.zeros((64, 64), dtype=np.uint8)
    bad_mask[10, 10] = 42  # invalid label ID
    with pytest.raises(ValueError, match="Invalid raw pixel IDs"):
        encoder.encode_and_resize(bad_mask, target_size=(32, 32))


def test_transforms_geometric_and_photometric_invariants(cfg: dict) -> None:
    """Verify geometric transforms preserve mask IDs & val/test are strictly deterministic."""
    rng_data = np.random.default_rng(123)
    img = rng_data.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    mask = np.full((512, 512), 255, dtype=np.uint8)
    mask[100:400, 100:400] = 1
    mask[150:250, 150:250] = 3

    train_tf = SegmentationTransformPipeline(cfg, split="train", seed=42)
    for trial in range(10):
        aug_img, aug_mask = train_tf.apply_numpy(
            img.copy(), mask.copy(), rng=np.random.default_rng(100 + trial)
        )
        assert aug_img.shape == (512, 512, 3)
        assert aug_mask.shape == (512, 512)
        assert aug_img.dtype == np.uint8
        assert aug_mask.dtype == np.uint8
        assert set(np.unique(aug_mask).tolist()).issubset({1, 3, 255})

    # Val & test must be 100% deterministic (zero random augmentation)
    for split in ("val", "test"):
        eval_tf = SegmentationTransformPipeline(cfg, split=split, seed=42)
        out_img1, out_msk1 = eval_tf.apply_numpy(img.copy(), mask.copy())
        out_img2, out_msk2 = eval_tf.apply_numpy(img.copy(), mask.copy())
        assert np.array_equal(out_img1, img)
        assert np.array_equal(out_msk1, mask)
        assert np.array_equal(out_img1, out_img2)
        assert np.array_equal(out_msk1, out_msk2)


def test_split_manifest_zero_leakage_and_official_test_integrity(
    cfg: dict, project_root: Path
) -> None:
    manifest = json.loads(
        resolve_path(cfg["paths"]["split_manifest_file"], project_root).read_text(
            encoding="utf-8"
        )
    )
    train_ids = {r["id"] for r in manifest["splits"]["train"]}
    val_ids = {r["id"] for r in manifest["splits"]["val"]}
    test_ids = {r["id"] for r in manifest["splits"]["test"]}

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0

    train_groups = {r["group_id"] for r in manifest["splits"]["train"]}
    val_groups = {r["group_id"] for r in manifest["splits"]["val"]}
    assert len(train_groups & val_groups) == 0

    # All 526 official expert test pairs (322 MSL + 204 MER) must remain in test
    assert len(test_ids) == 526
    assert manifest["counts"]["rejected_pairs"] == 30


def test_metadata_artifacts_exist_and_no_test_leakage_in_stats(
    cfg: dict, project_root: Path
) -> None:
    prep_cfg = json.loads(
        resolve_path(cfg["paths"]["preprocessing_config_file"], project_root).read_text(
            encoding="utf-8"
        )
    )
    assert prep_cfg["normalization"]["statistics_source"] == "processed_train_split_only"
    assert len(prep_cfg["normalization"]["active_mean"]) == 3
    assert len(prep_cfg["normalization"]["active_std"]) == 3

    cw = json.loads(
        resolve_path(cfg["paths"]["class_weights_file"], project_root).read_text(
            encoding="utf-8"
        )
    )
    assert cw["weight_calculation_split"] == "train_only"
    for method, weights in cw["weighting_methods"].items():
        assert set(weights.keys()) == {"0", "1", "2", "3"}
        assert all(w > 0.0 for w in weights.values()), f"Non-positive weight in {method}"


def test_preprocessing_idempotency(cfg: dict, project_root: Path) -> None:
    """Re-running preprocessing with unchanged config must reuse cached outputs."""
    summary = run_preprocessing(cfg, force=False, project_root=project_root)
    assert summary["counts"]["processed_total"] == 2296


def test_pytorch_dataset_dataloader_and_loss_compatibility(
    cfg: dict, project_root: Path
) -> None:
    loaders = build_dataloaders(
        config_path=project_root / "configs" / "data_config.yaml",
        batch_size=4,
        num_workers=0,
        project_root=project_root,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=int(cfg["mask"]["ignore_index"]))
    target_h, target_w = cfg["image"]["target_size"]

    for split_name, loader in loaders.items():
        imgs, masks = next(iter(loader))
        assert imgs.shape == (4, 3, target_h, target_w)
        assert masks.shape == (4, target_h, target_w)
        assert imgs.dtype == torch.float32
        assert masks.dtype == torch.int64
        assert torch.isfinite(imgs).all()

        unique_mask_vals = set(torch.unique(masks).tolist())
        assert unique_mask_vals.issubset({0, 1, 2, 3, 255})

        logits = torch.randn((4, 4, target_h, target_w), requires_grad=True)
        loss = criterion(logits, masks)
        assert torch.isfinite(loss)
        loss.backward()

    # Verify dataset metadata lookup works cleanly
    ds: MangalDrishtiDataset = loaders["test"].dataset  # type: ignore[assignment]
    meta = ds.get_sample_metadata(0)
    assert "id" in meta and "mission" in meta
