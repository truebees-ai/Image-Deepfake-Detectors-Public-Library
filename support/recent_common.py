"""Shared CLI and reporting helpers for the 2026 detector adapters."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser(method, task):
    p = argparse.ArgumentParser(description=f"{method} {task}")
    p.add_argument("--name", default="pretrained")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_threads", type=int, default=0)
    if task == "detect":
        p.add_argument("--image", required=True)
        p.add_argument("--model", default=None)
        p.add_argument("--output")
    else:
        p.add_argument("--task", default=task)
        p.add_argument("--data_root", required=True)
        p.add_argument("--split_file", required=True)
        p.add_argument("--data_keys", required=True)
        p.add_argument("--epochs", type=int, default=1)
        p.add_argument("--max_steps", type=int, default=0)
    return p


def device_for(spec):
    import torch
    if spec in (None, "auto"):
        spec = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {spec}, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS, but it is unavailable")
    return device


def validate_inputs(args, task):
    if task == "detect":
        if not Path(args.image).is_file():
            raise FileNotFoundError(f"Image not found: {args.image}")
    else:
        if not Path(args.data_root).is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {args.data_root}")
        if not Path(args.split_file).is_file():
            raise FileNotFoundError(f"Split file not found: {args.split_file}")
        if args.batch_size < 1:
            raise ValueError("--batch_size must be at least 1")


def checkpoint(method, name):
    path = ROOT / "detectors" / method / "checkpoint" / name / "weights" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}. Run support/setup_recent_weights.py or train first.")
    return path


def detection_result(method, image, raw_score, fake_score, elapsed, output=None):
    result = {"detector": method, "image": str(Path(image).resolve()),
              "prediction": "fake" if fake_score >= 0.5 else "real",
              "fake_score": float(fake_score), "confidence": float(max(fake_score, 1 - fake_score)),
              "raw_score": float(raw_score), "elapsed_time": float(elapsed)}
    path = Path(output) if output else ROOT / "detection_results" / method / "detect" / (Path(image).stem + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def write_results(method, name, keys, rows):
    directory = ROOT / "detectors" / method / "results" / name / keys.replace(":", "-").replace("&", "_")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "raw_score", "fake_score", "prediction", "elapsed_time"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {directory / 'results.csv'}")


def timed_score(score_fn, path):
    start = time.perf_counter()
    raw, fake = score_fn(path)
    return raw, fake, time.perf_counter() - start
