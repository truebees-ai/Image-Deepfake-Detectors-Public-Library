"""Build a deterministic DeepShield split without source overlap."""
import argparse
import json
import random
from pathlib import Path


def build_split(dataset_root, seed=42):
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    ids = set()
    for social in ('PreSocial', 'Facebook', 'Telegram', 'X'):
        for label in ('Real', 'Fake'):
            base = root / social / label
            if base.exists():
                ids.update(p.relative_to(base).with_suffix('').as_posix()
                           for p in base.rglob('*')
                           if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not ids:
        raise ValueError(f'No images found below {root}')
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    return {'train': sorted(ordered[:int(n * .70)]),
            'val': sorted(ordered[int(n * .70):int(n * .85)]),
            'test': sorted(ordered[int(n * .85):])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root')
    parser.add_argument('--output', default='split.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(build_split(args.dataset_root, args.seed), indent=2) + '\n')
