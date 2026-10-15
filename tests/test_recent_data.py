import json
import pickle
import sys
from pathlib import Path

from PIL import Image

from support.json_compile import build_split
from support.recent_data import samples


def test_split_keeps_social_variants_together(tmp_path):
    for index in range(20):
        for social in ("PreSocial", "Facebook"):
            path = tmp_path / social / "Real" / "FFHQ" / "part" / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(path)
    manifest = build_split(tmp_path)
    assert sum(map(len, manifest.values())) == 20
    assert not set(manifest["train"]) & set(manifest["val"])
    assert not set(manifest["train"]) & set(manifest["test"])
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(manifest))
    found = samples(tmp_path, split_file, "train", "realFFHQ:all")
    assert len(found) == 2 * len(manifest["train"])
    assert all(label == 0 for _, label in found)


def test_mdmf_reference_serialization_excludes_val_and_fake(tmp_path):
    import torch
    from types import SimpleNamespace
    method = Path(__file__).resolve().parents[1] / "detectors" / "MDMF"
    sys.path.insert(0, str(method))
    try:
        from run import embedding_bank
    finally:
        sys.path.pop(0)

    class TinyBackbone(torch.nn.Module):
        def forward(self, x):
            return SimpleNamespace(last_hidden_state=torch.ones(x.shape[0], 257, 1024))

    train_real = tmp_path / "train_real.png"
    val_real = tmp_path / "val_real.png"
    train_fake = tmp_path / "train_fake.png"
    for path in (train_real, val_real, train_fake):
        Image.new("RGB", (16, 16)).save(path)
    target = tmp_path / "reference.pkl"
    embedding_bank(TinyBackbone(), torch.device("cpu"), [(train_real, 0)], "train", target)
    with target.open("rb") as handle:
        data = pickle.load(handle)
    assert data["labels"] == [0]
    assert data["image_paths"] == [str(train_real)]
    assert str(val_real) not in data["image_paths"]
    assert str(train_fake) not in data["image_paths"]
