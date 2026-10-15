"""DeepShield adapter for MDMF's DINOv2 embeddings and MMD² score."""

import json
import math
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE / "upstream" / "src")]

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

from support.recent_common import checkpoint, detection_result, device_for, parser, timed_score, validate_inputs, write_results
from support.recent_data import samples
from test_MDMF import MMD_MIL_Detector, pool_tokens_to_patch_size, precompute_ref_stats, score_mmd2_batch
from train_MDMF import MMD_MIL_Detector as TrainableDetector  # same layers, defines the training forward()


def backbone_for(device):
    location = HERE / "checkpoint" / "backbones" / "dinov2-large"
    if not (location / "config.json").is_file():
        raise FileNotFoundError(f"DINOv2 backbone missing: {location}; run support/setup_recent_weights.py MDMF")
    return AutoModel.from_pretrained(location).to(device).eval()


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


@torch.inference_mode()
def embedding_batch(backbone, device, paths, patch_size=32):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(TRANSFORM(image.convert("RGB")))
    tensor = torch.stack(images).to(device)
    tokens = backbone(tensor).last_hidden_state[:, 1:, :].float()
    return pool_tokens_to_patch_size(tokens, patch_size, 224).cpu().numpy()


def embedding(backbone, device, path, patch_size=32):
    return embedding_batch(backbone, device, [path], patch_size)[0]


def embedding_bank(backbone, device, entries, split, target, batch_size=4):
    vectors = []
    for offset in range(0, len(entries), batch_size):
        vectors.extend(embedding_batch(backbone, device,
                       [path for path, _ in entries[offset:offset + batch_size]]))
    data = {"embeddings": vectors, "labels": [label for _, label in entries],
            "image_paths": [str(path) for path, _ in entries], "patch_size": 32,
            "split": split, "embed_dim": 1024}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(data, handle, protocol=4)
    return data


def model_for(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    model = MMD_MIL_Detector(config.get("embed_dim", 1024), config.get("hidden_dim", 256),
                             config.get("dropout", .3), config.get("init_sigma", 1.)).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model.eval(), int(config.get("patch_size", 32))


def reference_for(model, device, path):
    if not path.is_file():
        raise FileNotFoundError(f"MDMF real training reference bank missing: {path}; run detectors/MDMF/prepare.py")
    with path.open("rb") as handle:
        bank = pickle.load(handle)
    if bank.get("split") != "train" or any(bank["labels"]):
        raise ValueError("MDMF reference bank must contain training-split real images only")
    if not bank["embeddings"]:
        raise ValueError("MDMF reference bank is empty")
    with torch.inference_mode():
        logits = torch.stack([model.compute_logits(torch.from_numpy(x).float().unsqueeze(0).to(device)).squeeze(0)
                              for x in bank["embeddings"]])
        stats = precompute_ref_stats(logits, model.get_sigma())
    return logits, stats


def scorer(backbone, model, device, patch_size, reference, calibration):
    ref, (xx_mean, norm2) = reference
    threshold, scale = calibration["threshold"], calibration["scale"]

    @torch.inference_mode()
    def score_batch(paths):
        vectors = torch.from_numpy(embedding_batch(backbone, device, paths, patch_size)).float().to(device)
        logits = model.compute_logits(vectors)
        raws = score_mmd2_batch(ref, norm2, xx_mean, logits, model.get_sigma())
        fakes = torch.sigmoid((raws - threshold) / scale)
        return list(zip(raws.cpu().tolist(), fakes.cpu().tolist()))

    def score(path):
        return score_batch([path])[0]

    score.batch = score_batch
    return score


def calibrate(backbone, model, device, patch_size, reference, entries, path):
    # The released MDMF checkpoint contains no binary decision threshold.
    neutral = {"threshold": 0., "scale": 1.}
    score = scorer(backbone, model, device, patch_size, reference, neutral)
    real = [score(p)[0] for p, label in entries if label == 0]
    fake = [score(p)[0] for p, label in entries if label == 1]
    if not real or not fake:
        raise ValueError("MDMF calibration needs both real and fake validation images")
    import statistics
    a, b = statistics.median(real), statistics.median(fake)
    if b <= a:
        # Expected after very short (smoke) training; a real run should separate real and fake scores.
        print("WARNING: MDMF validation scores do not separate (median fake <= median real); "
              "predictions from this checkpoint are unreliable", file=sys.stderr)
    result = {"threshold": (a + b) / 2, "scale": max(abs(b - a) / 4, 1e-6),
              "source": "validation split", "real_count": len(real), "fake_count": len(fake)}
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(task):
    args = parser("MDMF", task).parse_args()
    validate_inputs(args, task)
    device = device_for(args.device)
    name = args.model or args.name if task == "detect" else args.name
    directory = HERE / "checkpoint" / name / "weights"
    backbone = backbone_for(device)

    if task == "train":
        train = samples(args.data_root, args.split_file, "train", args.data_keys)
        val = samples(args.data_root, args.split_file, "val", args.data_keys)
        real = [(p, y) for p, y in train if y == 0]
        if not real or not any(y == 1 for _, y in train):
            raise ValueError("MDMF training requires real and fake training images")
        directory.mkdir(parents=True, exist_ok=True)
        train_bank = embedding_bank(backbone, device, train, "train", directory / "train.pkl", args.batch_size)
        embedding_bank(backbone, device, real, "train", directory / "reference.pkl", args.batch_size)
        model = TrainableDetector().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        real_vec = [torch.from_numpy(x).float() for x, y in zip(train_bank["embeddings"], train_bank["labels"]) if y == 0]
        fake_vec = [torch.from_numpy(x).float() for x, y in zip(train_bank["embeddings"], train_bank["labels"]) if y == 1]
        for epoch in range(args.epochs):
            model.train()
            half_batch = max(1, args.batch_size // 2)
            steps = math.ceil(max(len(real_vec), len(fake_vec)) / half_batch)
            if args.max_steps:
                steps = min(steps, args.max_steps)
            for step in range(steps):
                a = torch.stack([real_vec[(step * half_batch + i) % len(real_vec)]
                                 for i in range(half_batch)]).to(device)
                b = torch.stack([fake_vec[(step * half_batch + i) % len(fake_vec)]
                                 for i in range(half_batch)]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, mmd, _ = model(a, b)
                loss.backward()
                optimizer.step()
                print(f"epoch={epoch + 1} step={step + 1} loss={loss.item():.5f} mmd2={mmd.item():.5f}")
        torch.save({"model_state_dict": model.state_dict(), "config": {"embed_dim": 1024, "hidden_dim": 256,
                    "dropout": .3, "init_sigma": 1., "patch_size": 32}}, directory / "best.pt")
        model.eval()
        reference = reference_for(model, device, directory / "reference.pkl")
        calibrate(backbone, model, device, 32, reference, val, directory / "calibration.json")
        return

    model, patch_size = model_for(checkpoint("MDMF", name), device)
    reference = reference_for(model, device, directory / "reference.pkl")
    calibration_path = directory / "calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"MDMF calibration missing: {calibration_path}; run detectors/MDMF/prepare.py")
    score = scorer(backbone, model, device, patch_size, reference, json.loads(calibration_path.read_text()))
    if task == "detect":
        image = Path(args.image)
        if not image.is_file():
            raise FileNotFoundError(image)
        raw, fake, elapsed = timed_score(score, image)
        detection_result("MDMF", image, raw, fake, elapsed, args.output)
    else:
        rows = []
        entries = samples(args.data_root, args.split_file, "test", args.data_keys)
        for offset in range(0, len(entries), args.batch_size):
            chunk = entries[offset:offset + args.batch_size]
            start = time.perf_counter()
            scores = score.batch([path for path, _ in chunk])
            elapsed = (time.perf_counter() - start) / len(chunk)
            for (path, label), (raw, fake) in zip(chunk, scores):
                rows.append({"path": str(path), "label": label, "raw_score": raw,
                             "fake_score": fake, "prediction": int(fake >= .5), "elapsed_time": elapsed})
        write_results("MDMF", name, args.data_keys, rows)
