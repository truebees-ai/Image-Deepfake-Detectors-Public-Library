"""Resolve DeepShield split IDs into labeled image paths for recent detectors."""

import json
import os
from pathlib import Path


GENERATOR_KEYS = {
    "gan1": "StyleGAN", "gan2": "StyleGAN2", "gan3": "StyleGAN3",
    "sd15": "StableDiffusion1.5", "sd2": "StableDiffusion2",
    "sd3": "StableDiffusion3", "sdXL": "StableDiffusionXL",
    "flux": "FLUX.1", "realFFHQ": "FFHQ", "realFORLAB": "FORLAB",
}
SOCIAL_KEYS = {"pre": "PreSocial", "fb": "Facebook", "tl": "Telegram", "tw": "X"}
EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _selected(keys):
    pairs = []
    for item in keys.split("&"):
        generator, social = item.split(":", 1)
        generators = GENERATOR_KEYS.values() if generator == "all" else (
            ["FFHQ", "FORLAB"] if generator == "real" else [GENERATOR_KEYS[generator]]
        )
        socials = SOCIAL_KEYS.values() if social == "all" else (
            [SOCIAL_KEYS[k] for k in ("fb", "tl", "tw")] if social == "shr" else [SOCIAL_KEYS[social]]
        )
        pairs.extend((g, s) for g in generators for s in socials)
    return set(pairs)


def samples(data_root, split_file, split, keys):
    """Return (absolute path, 0-real/1-fake) without crossing split boundaries."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    with open(split_file, encoding="utf-8") as handle:
        ids = set(json.load(handle)[split])
    selected = _selected(keys)
    found = []
    for social_dir in root.iterdir():
        if not social_dir.is_dir():
            continue
        for label_dir in social_dir.iterdir():
            if label_dir.name not in ("Real", "Fake") or not label_dir.is_dir():
                continue
            for current, _, filenames in os.walk(label_dir, followlinks=True):
                for filename in filenames:
                    path = Path(current) / filename
                    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                        continue
                    rel = path.relative_to(label_dir)
                    if (rel.parts[0], social_dir.name) not in selected:
                        continue
                    if rel.with_suffix("").as_posix() in ids:
                        found.append((path.resolve(), int(label_dir.name == "Fake")))
    if not found:
        raise ValueError(f"No {split} images matched {keys!r} under {root}")
    return sorted(found, key=lambda item: str(item[0]))


def stage(root, split, entries):
    """Create an ImageFolder-compatible view with links to the original images."""
    root = Path(root)
    counts = {0: 0, 1: 0}
    for path, label in entries:
        target = root / split / ("1_fake" if label else "0_real") / f"{counts[label]:08d}{path.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() == path:
            counts[label] += 1
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(path)
        counts[label] += 1
    return root / split
