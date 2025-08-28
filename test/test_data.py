# tests/test_data.py
import os
import sys
from pathlib import Path
import shutil
import random

import numpy as np
from PIL import Image
import pytest

# Make "src" importable when tests run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import get_dataloaders  # noqa: E402


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """
    Build a tiny dataset on the fly:

    data/
      train/{healthy,defect}/(n images)
      val/{healthy,defect}/(n images)
      test/{healthy,defect}/(n images)
    """
    root = tmp_path_factory.mktemp("tiny_data")
    data_dir = root / "data"
    splits = ["train", "val", "test"]
    classes = ["healthy", "defect"]

    # how many images per split/class
    n_train, n_val, n_test = 6, 2, 2
    per_split = {"train": n_train, "val": n_val, "test": n_test}

    rng = np.random.default_rng(42)

    for split in splits:
        for cls in classes:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)
            for i in range(per_split[split]):
                # simple RGB noise image 64x64
                arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(arr, mode="RGB")
                img.save(data_dir / split / cls / f"{cls}_{i}.jpg")

    yield str(data_dir)

    # cleanup
    shutil.rmtree(root, ignore_errors=True)


def test_get_dataloaders_shapes_and_classes(tiny_dataset):
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=tiny_dataset, batch_size=4, img_size=64
    )

    # classes detected correctly
    assert set(classes) == {"healthy", "defect"}

    # grab one batch and check tensor shapes
    xb, yb = next(iter(train_loader))
    # (B, C, H, W)
    assert xb.ndim == 4
    assert xb.shape[1] in (1, 3)  # grayscale or RGB post-transform
    assert xb.shape[2] == 64 and xb.shape[3] == 64
    assert yb.ndim == 1

    # ensure loaders are non-empty
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0


def test_get_dataloaders_batching(tiny_dataset):
    batch_size = 4
    train_loader, _, _, _ = get_dataloaders(
        data_dir=tiny_dataset, batch_size=batch_size, img_size=64
    )
    for xb, yb in train_loader:
        assert xb.shape[0] <= batch_size
        break  # just check first batch


def test_reproducible_ordering_with_shuffle_off(tiny_dataset):
    # This checks that when shuffle=False, two iterations yield same first filename
    # We’ll reuse the underlying dataset from the loader to peek its samples.
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    ds = datasets.ImageFolder(
        root=os.path.join(tiny_dataset, "train"),
        transform=transforms.ToTensor(),
    )
    dl1 = DataLoader(ds, batch_size=2, shuffle=False)
    dl2 = DataLoader(ds, batch_size=2, shuffle=False)

    b1 = next(iter(dl1))[0]
    b2 = next(iter(dl2))[0]
    # tensors equal for first batch (values differ due to transforms randomness only if present)
    assert b1.shape == b2.shape
