"""Build an MDMF reference from real training images and calibrate on validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import HERE, backbone_for, calibrate, model_for, reference_for
from support.recent_common import checkpoint, device_for, parser, validate_inputs
from support.recent_data import samples
from run import embedding_bank


if __name__ == "__main__":
    args = parser("MDMF", "prepare").parse_args()
    validate_inputs(args, "prepare")
    device = device_for(args.device)
    directory = HERE / "checkpoint" / args.name / "weights"
    model, patch_size = model_for(checkpoint("MDMF", args.name), device)
    backbone = backbone_for(device)
    real = [(p, y) for p, y in samples(args.data_root, args.split_file, "train", args.data_keys) if y == 0]
    if not real:
        raise ValueError("No real training images for MDMF reference")
    embedding_bank(backbone, device, real, "train", directory / "reference.pkl", args.batch_size)
    reference = reference_for(model, device, directory / "reference.pkl")
    val = samples(args.data_root, args.split_file, "val", args.data_keys)
    calibrate(backbone, model, device, patch_size, reference, val, directory / "calibration.json")
    print(f"Prepared {len(real)} training-real reference images")
