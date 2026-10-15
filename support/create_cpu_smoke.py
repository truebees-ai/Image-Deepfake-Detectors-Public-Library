"""Create a small, balanced DeepShield smoke dataset using image symlinks."""

import argparse
import json
import random
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GROUPS = (
    ("Fake", "StyleGAN2", "conf-f-psi-0.5"),
    ("Fake", "StableDiffusionXL", "faces"),
    ("Real", "FFHQ", None),
    ("Real", "FORLAB", None),
)
METHODS = ("R50_TF", "R50_nodown", "CLIP-D", "P2G", "NPR", "PGC", "ForensicConcept", "MDMF")
SMOKE_ARGS = {
    "R50_TF": ["--num_epoches", "2", "--batch_size", "2"],
    "R50_nodown": ["--num_epoches", "1", "--batch_size", "2"],
    "CLIP-D": ["--num_epoches", "1", "--batch_size", "2"],
    "NPR": ["--niter", "2", "--batch_size", "2"],
    "PGC": ["--epochs", "1", "--max_steps", "1"],
    "ForensicConcept": ["--epochs", "1", "--max_steps", "1"],
    "MDMF": ["--epochs", "1", "--max_steps", "1"],
}


def create(source: Path, output: Path, seed: int, train_count: int, val_count: int, test_count: int):
    source = source.expanduser().resolve()
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new directory.")

    total = train_count + val_count + test_count
    if min(train_count, val_count, test_count) < 1:
        raise ValueError("Each split needs at least one image per group")

    chosen = []
    for index, (label, generator, subdir) in enumerate(GROUPS):
        directory = source / "PreSocial" / label / generator
        if subdir:
            directory /= subdir
        if not directory.is_dir():
            raise FileNotFoundError(f"DeepShield group missing: {directory}")
        candidates = sorted(path for path in directory.iterdir()
                            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if len(candidates) < total:
            raise ValueError(f"Need {total} images in {directory}; found {len(candidates)}")
        chosen.append((label, random.Random(seed + index).sample(candidates, total)))

    splits = {"train": [], "val": [], "test": []}
    selected = []
    for label, paths in chosen:
        offsets = (("train", 0, train_count),
                   ("val", train_count, train_count + val_count),
                   ("test", train_count + val_count, total))
        for split, start, end in offsets:
            for source_image in paths[start:end]:
                relative = source_image.relative_to(source)
                link = output / "dataset" / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(source_image)
                source_id = relative.relative_to(Path("PreSocial") / label).with_suffix("").as_posix()
                splits[split].append(source_id)
                selected.append({"split": split, "label": label.lower(),
                                 "source": str(source_image), "link": str(link)})

    for ids in splits.values():
        ids.sort()
    (output / "split.json").write_text(json.dumps(splits, indent=2) + "\n")
    (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")

    configs = output / "configs"
    configs.mkdir()
    for method in METHODS:
        config = yaml.safe_load((ROOT / "configs" / f"{method}.yaml").read_text())
        config["global"].update(dataset_path=str(output / "dataset"),
                                split_file=str(output / "split.json"),
                                device_override="cpu", num_threads=0)
        config["testing"] = ["gan2:pre&realFFHQ:pre"]
        config["detector_args"] = config.get("detector_args", []) + SMOKE_ARGS.get(method, [])
        (configs / f"{method}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"Created {output}: {len(selected)} image links; "
          + ", ".join(f"{name}={len(ids)}" for name, ids in splits.items()))
    print("P2G still requires CUDA; see README.md for the CPU limitation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Root of the full DeepShield dataset")
    parser.add_argument("output", type=Path, help="New directory for symlinks and smoke configs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-group", type=int, default=8)
    parser.add_argument("--val-per-group", type=int, default=8)
    parser.add_argument("--test-per-group", type=int, default=4)
    args = parser.parse_args()
    create(args.source, args.output, args.seed,
           args.train_per_group, args.val_per_group, args.test_per_group)
