
# Image Deepfake Detectors Public Library

## Overview

This repository provides a unified framework for training, testing, and benchmarking multiple deepfake detection models. It supports automated experiments, demo runs, and single-image detection, with modular configuration and extensible detector support.

### Main Features

- **Multiple Detectors:** Supports CLIP-D, NPR, P2G, R50_TF, R50_nodown, and more.
- **Automated Training & Testing:** Use `launcher.py` to run experiments across detectors and datasets.
- **Demo Mode:** Easily test all detectors on sample images in `demo_images/`.
- **Single Image Detection:** Run detection on individual images via command-line.
- **Flexible Configuration:** All experiment parameters are set via YAML files in `configs/`.
- **Logging & Results:** Logs and results are saved per detector and scenario for easy analysis.

---

## Data Organization

### Dataset Structure

- **Main Dataset:** Organize your dataset as specified in the YAML config (`dataset_path`). Example:
	```
	/path/to/dataset/
			Real/
			Fake/
	```
- **Demo Images:** Place sample images for quick testing in `demo_images/`, organized by platform and label:
	```
	demo_images/
			Facebook/
					Fake/
					Real/
			PreSocial/
					Fake/
					Real/
			Telegram/
					Fake/
					Real/
			X/
					Fake/
					Real/
	```

### Split Files

- **split.json:** Main split file for experiments. Format: JSON with train/test keys and lists of sample IDs.
- **split_demo.json:** Auto-generated for demo mode, covering all images in `demo_images/`.

---

## Weights Organization

- **Pretrained Weights:** Place weights in each detector's folder:
	```
	detectors/<DETECTOR>/checkpoint/pretrained/weights/best.pt
	```
- **Demo Mode:** The launcher will copy weights to a temporary location for demo runs.
- **Custom Weights:** Specify custom weights via command-line (`--weights_name`).

---

## Configuration

- **YAML Files:** All detectors have a config file in `configs/` (e.g., `CLIP-D.yaml`, `NPR.yaml`).
- **Config Options:**
	- `global`: Dataset path, device, split file, threads, etc.
	- `detector_args`: Model-specific arguments.
	- `training`: List of training scenarios.
	- `testing`: List of test scenarios.

---

## Usage

### 1. Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```
Or use conda:
```bash
conda env create -f environment.yml
```

### 2. Running Experiments

**Automated (Recommended):**
```bash
python launcher.py --detector <DETECTOR> --phases both
```
- `<DETECTOR>`: One of `CLIP-D`, `NPR`, `P2G`, `R50_TF`, `R50_nodown`
- `--phases`: `train`, `test`, or `both`

**Manual:**
```bash
cd detectors/<DETECTOR>
python train.py --name <EXPERIMENT_NAME> --split_file <SPLIT_FILE> --task train --data_keys <DATA_KEYS> --data_root <DATASET_ROOT> --device <DEVICE>
python test.py --name <EXPERIMENT_NAME> --split_file <SPLIT_FILE> --task test --data_keys <DATA_KEYS> --data_root <DATASET_ROOT> --device <DEVICE>
```

### 3. Demo Mode

Test all detectors on sample images:
```bash
python launcher.py --demo
```

### 4. Single Image Detection

```bash
python launcher.py --detect --detector <DETECTOR> --image <IMAGE_PATH> --weights <WEIGHTS_PATH>
```

---

## Results & Logs

- **Results:** Saved in `detectors/<DETECTOR>/results/`
- **Logs:** Saved in `logs/` per run and scenario

---

## Supported Detectors

- CLIP-D
- NPR
- P2G
- R50_TF
- R50_nodown

Each detector has its own folder with scripts for training, testing, and detection.

---

## Advanced

- **Custom Configs:** Add new YAML files in `configs/` for new detectors or scenarios.
- **Extending:** Add new detectors by creating a folder in `detectors/` and updating `launcher.py`.

---

## Citation & Dataset

- DeepShield Dataset: [Zenodo link](https://zenodo.org/records/15648378)

---


