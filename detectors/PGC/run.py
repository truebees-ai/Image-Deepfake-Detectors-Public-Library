"""DeepShield adapter for the official PGC model and preprocessing."""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE / "upstream")]

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from support.recent_common import checkpoint, detection_result, device_for, parser, timed_score, validate_inputs, write_results
from support.recent_data import samples
from data.transforms import MEAN, STD, create_eval_transforms, create_train_transforms
from models.pgc import PGCNetwork


class Images(Dataset):
    def __init__(self, entries, transform):
        self.entries, self.transform = entries, transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, label = self.entries[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def model_for(device):
    backbone = HERE / "checkpoint" / "backbones" / "dinov2-large"
    if not backbone.is_dir():
        raise FileNotFoundError(f"DINOv2 backbone missing: {backbone}; run support/setup_recent_weights.py PGC")
    return PGCNetwork(dino_variant="dinov2-large", pretrained_root=str(backbone)).to(device)


def transform(train=False):
    fn = create_train_transforms if train else create_eval_transforms
    return fn(image_size=224, mean=MEAN["dino"], std=STD["dino"])


def main(task):
    args = parser("PGC", task).parse_args()
    validate_inputs(args, task)
    device = device_for(args.device)
    model = model_for(device)
    if task == "train":
        entries = samples(args.data_root, args.split_file, "train", args.data_keys)
        loader = DataLoader(Images(entries, transform(True)), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_threads)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        model.train()
        for epoch in range(args.epochs):
            for step, (images, labels) in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.to(device)).flatten()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float().to(device))
                loss.backward()
                optimizer.step()
                print(f"epoch={epoch + 1} step={step + 1} loss={loss.item():.5f}")
                if args.max_steps and step + 1 >= args.max_steps:
                    break
        target = HERE / "checkpoint" / args.name / "weights" / "best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, target)
        print(f"Saved {target}")
        return

    state = torch.load(checkpoint("PGC", getattr(args, "model", None) or args.name), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    evaluation = transform()

    @torch.inference_mode()
    def score(path):
        with Image.open(path) as image:
            tensor = evaluation(image.convert("RGB")).unsqueeze(0).to(device)
        raw = float(model(tensor).flatten()[0].item())
        return raw, float(torch.sigmoid(torch.tensor(raw)).item())

    if task == "detect":
        image = Path(args.image)
        if not image.is_file():
            raise FileNotFoundError(image)
        raw, fake, elapsed = timed_score(score, image)
        detection_result("PGC", image, raw, fake, elapsed, args.output)
    else:
        entries = samples(args.data_root, args.split_file, "test", args.data_keys)
        loader = DataLoader(Images(entries, evaluation), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_threads)
        rows = []
        offset = 0
        with torch.inference_mode():
            for images, _ in loader:
                start = time.perf_counter()
                logits = model(images.to(device)).flatten().cpu()
                elapsed = (time.perf_counter() - start) / len(logits)
                probabilities = torch.sigmoid(logits)
                for index, (raw, fake) in enumerate(zip(logits.tolist(), probabilities.tolist())):
                    path, label = entries[offset + index]
                    rows.append({"path": str(path), "label": label, "raw_score": raw,
                                 "fake_score": fake, "prediction": int(fake >= .5), "elapsed_time": elapsed})
                offset += len(logits)
        write_results("PGC", args.name, args.data_keys, rows)
