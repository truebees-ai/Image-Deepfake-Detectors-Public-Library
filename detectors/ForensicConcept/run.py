"""DeepShield adapter for the official ForensicConcept CLIP codebook model."""

import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE / "upstream")]

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from config_loader import ConfigLoader
from data import get_dataset
from networks.trainer import Trainer
from support.recent_common import checkpoint, detection_result, device_for, parser, timed_score, validate_inputs, write_results
from support.recent_data import samples, stage


def configuration(name, device, batch_size, threads, *, train, dataroot=None, checkpoint_path=None):
    config = ConfigLoader(str(HERE / "upstream" / "configs" / "clip_codebook.yaml"))
    backbone = HERE / "checkpoint" / "backbones" / "ViT-L-14.pt"
    codebook = HERE / "upstream" / "codebooks" / "cleandift_codebook.npy"
    if not backbone.is_file():
        raise FileNotFoundError(f"CLIP backbone missing: {backbone}; run support/setup_recent_weights.py ForensicConcept")
    if not codebook.is_file():
        raise FileNotFoundError(f"Codebook missing: {codebook}")
    config.set("models.pretrained.clip_model", str(backbone))
    config.set("models.checkpoints_dir", str(HERE / "checkpoint"))
    config.set("training.clip_codebook_path", str(codebook))
    config.set("training.name", name)
    config.set("training.gpu_ids", device.index if device.type == "cuda" else -1)
    config.set("training.batch_size", batch_size)
    config.set("testing.batch_size", batch_size)
    config.set("system.num_threads", threads)
    config.set("training.continue_train", not train)
    config.set("training.epoch", str(checkpoint_path) if checkpoint_path else "best")
    config.set("logging.tensorboard", False)
    if dataroot:
        config.set("data.train_dataroot", str(dataroot))
        config.set("data.train_split", "train")
    return config


def load(name, device, batch_size, threads):
    path = checkpoint("ForensicConcept", name)
    config = configuration(name, device, batch_size, threads, train=False, checkpoint_path=path)
    opt = config.to_namespace(is_train=False)
    opt.local_rank = -1
    return Trainer(opt).model.eval().to(device)


def eval_transform():
    # The released CLIP codebook configuration uses legacy_preprocess=true.
    # Its eval path is resize -> center crop -> ImageNet normalization.
    return transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def main(task):
    args = parser("ForensicConcept", task).parse_args()
    validate_inputs(args, task)
    device = device_for(args.device)
    name = args.model or args.name if task == "detect" else args.name
    if task == "train":
        entries = samples(args.data_root, args.split_file, "train", args.data_keys)
        if {label for _, label in entries} != {0, 1}:
            raise ValueError("ForensicConcept training needs real and fake images")
        with tempfile.TemporaryDirectory(prefix="forensicconcept-") as temporary:
            stage(temporary, "train", entries)
            config = configuration(name, device, args.batch_size, args.num_threads,
                                   train=True, dataroot=temporary)
            opt = config.to_namespace(is_train=True)
            opt.dataroot = str(Path(temporary) / "train")
            opt.local_rank = -1
            data = get_dataset(opt)
            loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_threads)
            trainer = Trainer(opt)
            for epoch in range(args.epochs):
                trainer.train()
                for step, batch in enumerate(loader):
                    trainer.set_input(batch)
                    trainer.optimize_parameters()
                    print(f"epoch={epoch + 1} step={step + 1} loss={trainer.loss.item():.5f}")
                    if args.max_steps and step + 1 >= args.max_steps:
                        break
            trainer.save_networks("best")
            source = Path(trainer.save_dir) / "model_epoch_best.pth"
            target = HERE / "checkpoint" / name / "weights" / "best.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"Saved {target}")
        return

    model = load(name, device, args.batch_size, args.num_threads)
    transform = eval_transform()

    @torch.inference_mode()
    def score(path):
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        output = model(tensor)
        raw = float(output["logits"].flatten()[0].item())
        return raw, float(torch.sigmoid(torch.tensor(raw)).item())

    if task == "detect":
        image = Path(args.image)
        if not image.is_file():
            raise FileNotFoundError(image)
        raw, fake, elapsed = timed_score(score, image)
        detection_result("ForensicConcept", image, raw, fake, elapsed, args.output)
    else:
        rows = []
        entries = samples(args.data_root, args.split_file, "test", args.data_keys)
        with torch.inference_mode():
            for offset in range(0, len(entries), args.batch_size):
                chunk = entries[offset:offset + args.batch_size]
                tensors = []
                for path, _ in chunk:
                    with Image.open(path) as image:
                        tensors.append(transform(image.convert("RGB")))
                start = time.perf_counter()
                logits = model(torch.stack(tensors).to(device))["logits"].flatten().cpu()
                elapsed = (time.perf_counter() - start) / len(chunk)
                for (path, label), raw, fake in zip(chunk, logits.tolist(), torch.sigmoid(logits).tolist()):
                    rows.append({"path": str(path), "label": label, "raw_score": raw,
                                 "fake_score": fake, "prediction": int(fake >= .5), "elapsed_time": elapsed})
        write_results("ForensicConcept", name, args.data_keys, rows)
