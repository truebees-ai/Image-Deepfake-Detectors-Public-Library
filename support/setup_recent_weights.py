"""Fetch pinned published detector checkpoints and their original backbones."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
PGC_REV = "86424d6502f143c84a9df91f8336d2da07d0e235"
FC_REV = "ee3ed528de85ed44b4fcb4657e1df44955ed463c"
DINO_REV = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
MDMF_REV = "22d03f49d877b8a01d730284b5f73df555f5656f"
MDMF_SHA256 = "b9ae92006dcc837066eec8c8a9a26133f64281d6f6713e9eb6e0ccdc213f76dc"
CODEBOOK_SHA256 = "0a7cd9555750c5f08b1db0a5a2dc0d866ef195fcd05210bffe53b2c82cdc87ab"
CLIP_URL = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
UPSTREAM_REVISIONS = {
    "PGC": "0c9b10f3b89964b804ad6097e61709f74e827fbf",
    "ForensicConcept": "1a57fbcd5212a113341b7377f307897509e13f0d",
    "MDMF": MDMF_REV,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_verified(source, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or (expected and digest(target) != expected and Path(source) != target):
        shutil.copyfile(source, target)
    if target.stat().st_size == 0:
        raise ValueError(f"Empty weight file: {target}")
    checksum = digest(target)
    if expected and checksum != expected:
        raise ValueError(f"Checksum mismatch for {target}: {checksum}")
    return checksum


def hf_file(repo, revision, file, target):
    source = hf_hub_download(repo_id=repo, revision=revision, filename=file)
    return copy_verified(Path(source), target, digest(source))


def download(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or (expected and digest(target) != expected):
        temporary = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    return copy_verified(target, target, expected)


def dino_backbone(method):
    primary = ROOT / "detectors" / "PGC" / "checkpoint" / "backbones" / "dinov2-large"
    if not (primary / "config.json").is_file() or not (primary / "model.safetensors").is_file():
        source = Path(snapshot_download(repo_id="facebook/dinov2-large", revision=DINO_REV,
                                        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]))
        primary.mkdir(parents=True, exist_ok=True)
        for file in source.iterdir():
            if file.is_file():
                copy_verified(file, primary / file.name, digest(file))
    if not (primary / "model.safetensors").is_file():
        raise FileNotFoundError(f"DINOv2 model weights missing under {primary}")
    if method == "MDMF":
        link = ROOT / "detectors" / "MDMF" / "checkpoint" / "backbones" / "dinov2-large"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(primary, target_is_directory=True)
    return {str(p.relative_to(ROOT)): digest(p) for p in primary.iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["PGC", "ForensicConcept", "MDMF", "all"])
    args = parser.parse_args()
    methods = ["PGC", "ForensicConcept", "MDMF"] if args.method == "all" else [args.method]
    for method in methods:
        base = ROOT / "detectors" / method / "checkpoint"
        files = {}
        if method in ("PGC", "MDMF"):
            files.update(dino_backbone(method))
        if method == "PGC":
            path = base / "pretrained" / "weights" / "best.pt"
            files[str(path.relative_to(ROOT))] = hf_file("xiaoyuzhou68/PGC_ckpt", PGC_REV,
                "PGC_train_progan_sdv1_4_ckpt.pth", path)
            source = f"https://huggingface.co/xiaoyuzhou68/PGC_ckpt/tree/{PGC_REV}"
            license_name = "Apache-2.0"
        elif method == "ForensicConcept":
            path = base / "pretrained" / "weights" / "best.pt"
            files[str(path.relative_to(ROOT))] = hf_file("ethan225/ForensicConcept", FC_REV,
                "weights/detectors/clip_vitl14_codebook_stage2.pth", path)
            clip = base / "backbones" / "ViT-L-14.pt"
            files[str(clip.relative_to(ROOT))] = download(CLIP_URL, clip,
                "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836")
            codebook = ROOT / "detectors" / method / "upstream" / "codebooks" / "cleandift_codebook.npy"
            codebook_url = (f"https://raw.githubusercontent.com/EthanAdamm/FORENSICCONCEPT/"
                            f"{UPSTREAM_REVISIONS[method]}/assets/codebooks/cleandift_codebook.npy")
            files[str(codebook.relative_to(ROOT))] = download(codebook_url, codebook, CODEBOOK_SHA256)
            source = f"https://huggingface.co/ethan225/ForensicConcept/tree/{FC_REV}"
            license_name = "Apache-2.0 detector; MIT OpenAI CLIP backbone"
        else:
            path = base / "pretrained" / "weights" / "best.pt"
            url = f"https://raw.githubusercontent.com/ZBox1005/MDMF/{MDMF_REV}/checkpoints/model.pth"
            files[str(path.relative_to(ROOT))] = download(url, path, MDMF_SHA256)
            source = f"https://github.com/ZBox1005/MDMF/tree/{MDMF_REV}"
            license_name = "MIT"
        record = {"method": method, "source": source, "upstream_code_revision": UPSTREAM_REVISIONS[method],
                  "license": license_name,
                  "backbone": f"facebook/dinov2-large@{DINO_REV}" if method != "ForensicConcept" else CLIP_URL,
                  "sha256": files}
        (base / "weights-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Verified {method}: {len(files)} files; manifest: {base / 'weights-manifest.json'}")


if __name__ == "__main__":
    main()
