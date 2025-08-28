# tests/test_model.py
import sys
from pathlib import Path

import torch
import pytest

# Make "src" importable when tests run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import build_model  # noqa: E402
from data_loader import get_dataloaders  # noqa: E402


def test_build_model_head_shapes():
    num_classes = 2
    model = build_model(num_classes=num_classes, pretrained=False)
    model.eval()

    # forward a tiny dummy batch (B=2, C=3, H=W=64)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, num_classes)


@pytest.mark.parametrize("num_classes", [2, 3, 4])
def test_model_output_dim_varies_with_num_classes(num_classes):
    model = build_model(num_classes=num_classes, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[-1] == num_classes


def test_tiny_train_loop_cpu(tmp_path):
    """
    Smoke-test: run a very short train loop on a tiny synthetic dataset to ensure
    the pipeline does not crash. We rely on data_loader to create loaders
    given the folder structure built here.
    """
    from PIL import Image
    import numpy as np
    import os

    # Build tiny dataset on the fly
    data_dir = tmp_path / "data"
    for split, n in [("train", 4), ("val", 2), ("test", 2)]:
        for cls in ["healthy", "defect"]:
            d = data_dir / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(d / f"{cls}_{i}.jpg")

    # loaders
    train_loader, val_loader, _, classes = get_dataloaders(
        str(data_dir), batch_size=2, img_size=64
    )

    # model + 1-epoch train
    model = build_model(num_classes=len(classes), pretrained=False)
    device = torch.device("cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optim.step()
        break  # single batch is enough for a smoke test

    # sanity: run an eval step without crashing
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            _ = model(xb)
            break
