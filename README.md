# Image Deepfake Detectors Public Library

**Version 2.0** · Apache-2.0 (project code) · see [License](#license)

One command-line interface to **train, test, and run** eight AI-generated-image detectors on the same data, with the same split and the same configuration format.

| Detector | Type | Backbone | Ref. | CPU? | Published weights |
|---|---|---|---|---|---|
| `CLIP-D` | CLIP linear probe | CLIP ViT-L/14 | [1] | yes | manual download |
| `NPR` | CNN, up-sampling traces | ResNet-50 | [2] | yes | manual download |
| `P2G` | prompt tuning, continual | CLIP | [3] | **GPU only** | manual download |
| `R50_TF` | ResNet-50 + prototypes | ResNet-50 | [4] | yes | manual download |
| `R50_nodown` | ResNet-50 without down-sampling | ResNet-50 | [5] | yes | manual download |
| `PGC` | peak-guided calibration | DINOv2-L | [8] | yes | `setup_recent_weights.py` |
| `ForensicConcept` | CLIP codebook | CLIP ViT-L/14 | [9] | yes | `setup_recent_weights.py` |
| `MDMF` | local distribution shift (MMD) | DINOv2-L | [10] | yes | `setup_recent_weights.py` + `prepare.py` |

Detector names are case-sensitive. All commands are run **from the repository root**.

## Contents

1. [Install](#1-install)
2. [Get the data and split](#2-get-the-data-and-split)
3. [Test the published (pretrained) weights](#3-test-the-published-pretrained-weights)
4. [Train your own model](#4-train-your-own-model)
5. [Smoke test on a tiny CPU dataset](#5-smoke-test-on-a-tiny-cpu-dataset)
6. [Configuration](#6-configuration)
7. [Custom generators and new datasets](#7-custom-generators-and-new-datasets)
8. [Outputs](#8-outputs)
9. [Troubleshooting](#9-troubleshooting)
10. [License](#license) · [References](#references)

---

## 1. Install

Python 3.10+ is required. CUDA is optional (except P2G).

```bash
git clone <this-repo-url> && cd Image-Deepfake-Detectors-Public-Library
python -m venv .venv
source .venv/bin/activate          # re-run in every new terminal
python -m pip install -r requirements.txt
```

(`environment.yml` is provided for conda users.) The first run may download backbone files into your local model cache.

## 2. Get the data and split

The library uses the **DeepShield** dataset ([Zenodo](https://zenodo.org/records/15648378)). Layout:

```text
DeepShield/
  PreSocial/{Real,Fake}/<generator>/...
  Facebook/{Real,Fake}/<generator>/...
  Telegram/{Real,Fake}/<generator>/...
  X/{Real,Fake}/<generator>/...
```

> "TrueFake" is the historical name of this dataset ([4]); you will still see it in R50_TF's code and paper title.

1. Point every config to the dataset. In **each** `configs/<DETECTOR>.yaml` set:

   ```yaml
   global:
     dataset_path: /absolute/path/to/DeepShield
     split_file: ./split.json
   ```
   One-liner for all configs: `sed -i 's|<PATH_TO_DATASET>|/absolute/path/to/DeepShield|' configs/*.yaml`

2. **Use the `split.json` shipped in the repo** (70 040 train / 14 980 val / 14 980 test source IDs). Do **not** regenerate it if you want results comparable with other users or if you test pretrained weights. It lists source IDs, so the same file works for every detector.

   Only for a *new* dataset, build a deterministic 70/15/15 split with
   `python support/json_compile.py /path/to/dataset --output split.json` (seed 42, no overlap between sets).

## 3. Test the published (pretrained) weights

Testing a model = **install its weights as the run called `pretrained`**, then evaluate that run. No training is needed.

### 3.1 Install the weights

| Detector | How |
|---|---|
| `PGC`, `ForensicConcept`, `MDMF` | `python support/setup_recent_weights.py all` (or one name). Downloads checkpoints and backbones, verifies checksums, writes `weights-manifest.json`. No datasets are downloaded. |
| `R50_TF`, `R50_nodown`, `CLIP-D`, `P2G`, `NPR` | Download the [checkpoint archive](https://drive.google.com/file/d/1F60FN2B9skRcb3YrZwhFTZQihbj3ipJQ/view?usp=sharing) and unzip it **at the repository root**: `unzip <archive>.zip -d .` (it already contains the `detectors/<MODEL>/checkpoint/pretrained/weights/best.pt` paths) |

Check that it worked (8 files expected):

```bash
ls detectors/*/checkpoint/pretrained/weights/best.pt
```

**MDMF extra step.** Its `best.pt` does not contain the real-image reference bank or the calibrated threshold, so build them once from the *training-split real images* of the dataset you will test on:

```bash
export DATASET=/absolute/path/to/DeepShield
python detectors/MDMF/prepare.py --name pretrained --device cpu \
  --data_root "$DATASET" --split_file split.json \
  --data_keys 'gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre'
```

(For the smoke set of step 5 use `--data_root "$SMOKE/dataset" --split_file "$SMOKE/split.json"`.)

### 3.2 Evaluate on the labeled test set

```bash
export DATASET=/absolute/path/to/DeepShield
export MODEL=NPR            # R50_TF | R50_nodown | CLIP-D | P2G | NPR | PGC | ForensicConcept | MDMF
export DEVICE=auto          # auto | cpu | cuda:0   (P2G: cuda:0)

python launcher.py --detector "$MODEL" --phases test \
  --weights_name pretrained --device "$DEVICE"
```

This runs **every scenario listed under `testing:`** in `configs/$MODEL.yaml` (all generators × pre-social / Facebook / Telegram / X) on the `test` part of `split.json`. List the result files with:

```bash
find detectors/$MODEL/results/pretrained -name results.csv
```

Evaluate all detectors in a loop:

```bash
for m in R50_TF R50_nodown CLIP-D NPR PGC ForensicConcept MDMF; do
  python launcher.py --detector "$m" --phases test --weights_name pretrained --device auto || break
done
```

Add `--dry-run` to print the underlying commands without running them; `--batch-size N` lowers memory use for PGC / ForensicConcept / MDMF.

### 3.3 Classify a single image

```bash
python launcher.py --detect --detector "$MODEL" --image /path/to/image.jpg \
  --weights pretrained --device "$DEVICE" --output detection_results/$MODEL/out.json
```

The JSON contains `prediction`, `fake_score` (near 1 = likely fake) and `elapsed_time`. MDMF additionally reports `raw_score` (MMD², larger = more likely fake).

### 3.4 Demo on sample images

Download the [demo images](https://drive.google.com/file/d/134Bw8l9tEC7oZJpTAeMO80QRqgdJfJS9/view?usp=sharing) and unzip them at the repository root (`unzip <archive>.zip -d .`, which fills `demo_images/`), then:

```bash
python launcher.py --demo --demo-detector all     # or a single detector name
python launcher.py --demo --demo-detector all --device cpu   # force CPU
```

Detectors without installed pretrained weights (and P2G when not on CUDA) are skipped in `all` mode. Results: `detectors/<MODEL>/results/*/all-all/results.csv` or `.../data/all:all/` (`find detectors/*/results -path '*all*' -name results.csv`). Demo images are full resolution and are processed one at a time by the older detectors, so on CPU R50_TF and R50_nodown take about 10–30 minutes each (the others take about a minute); a GPU is much faster.

## 4. Train your own model

Pick a **new run name** (`--weights_name`); use the same name for testing and single-image detection. Training uses the `training:` entry of the config (default: StyleGAN2 + StableDiffusionXL fakes, FFHQ + FORLAB reals, pre-social).

```bash
export MODEL=PGC RUN=my_PGC DEVICE=auto

python launcher.py --detector "$MODEL" --phases train --weights_name "$RUN" --device "$DEVICE"
test -s detectors/$MODEL/checkpoint/$RUN/weights/best.pt && echo "checkpoint OK"

python launcher.py --detector "$MODEL" --phases test  --weights_name "$RUN" --device "$DEVICE"
python launcher.py --detect --detector "$MODEL" --image /path/to/image.jpg \
  --weights "$RUN" --device "$DEVICE" --output detection_results/$MODEL/my.json
```

- `--phases both` (default) trains and then tests.
- PGC, ForensicConcept and MDMF need their backbones (`setup_recent_weights.py`) even for training; MDMF builds its reference bank and threshold automatically. Training these on CPU can take many hours.
- P2G requires a CUDA GPU (`--device cuda:0`).
- For R50_TF, R50_nodown, CLIP-D, NPR and P2G the batch size and epochs come from the YAML `detector_args`.

## 5. Smoke test on a tiny CPU dataset

A functional check (not an accuracy benchmark) using an 80-image symlinked sample (32 train / 32 val / 16 test). It writes its **own** split and configs, so your main `configs/` and `split.json` stay untouched.

```bash
export SMOKE=/absolute/path/to/new-empty-dir      # must not exist yet
python support/create_cpu_smoke.py "$DATASET" "$SMOKE"   # skip if you already have a smoke folder
export IMAGE="$(find "$SMOKE/dataset/PreSocial/Fake/StableDiffusionXL" -type l -print -quit)"

for model in R50_TF R50_nodown CLIP-D NPR PGC ForensicConcept MDMF; do
  run="smoke_$model"
  python launcher.py --detector "$model" --phases train --config-dir "$SMOKE/configs" \
    --weights_name "$run" --device cpu --batch-size 2 || break
  test -s "detectors/$model/checkpoint/$run/weights/best.pt" || { echo "missing ckpt: $model"; break; }
  python launcher.py --detector "$model" --phases test --config-dir "$SMOKE/configs" \
    --weights_name "$run" --device cpu --batch-size 2 || break
  python launcher.py --detect --detector "$model" --image "$IMAGE" --weights "$run" \
    --device cpu --output "detection_results/$model/smoke.json" || break
done
```

For P2G run the same three commands on a GPU with `--device cuda:0`. The original DeepShield files must stay in place because the sample uses symlinks.

To test the **pretrained** weights on the smoke set instead, use the commands of step 3 with `--config-dir "$SMOKE/configs" --weights_name pretrained` (MDMF: run `prepare.py` with the smoke paths first, see 3.1).

> P2G rewrites `detectors/P2G/configs/test.json` / `train.json` on every run. Discard those changes with `git checkout -- detectors/P2G/configs/` before committing.

## 6. Configuration

Each detector has `configs/<DETECTOR>.yaml`:

| Section | Meaning |
|---|---|
| `global` | `dataset_path`, `split_file`, `device_override`, `num_threads`, `dry_run`, `only_list` |
| `detector_args` | Model-specific arguments passed to `train.py` / `test.py` |
| `training` | List of training scenarios (`data: <key>:<social>&...`) |
| `testing` | List of test scenarios, all evaluated by `--phases test` |

Launcher flags override the YAML: `--device`, `--batch-size`, `--config-dir`, `--weights_name`.

## 7. Custom generators and new datasets

**Different training generators.** Edit `training:` in the config:

```yaml
training:
  - data: gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre
```

Keys are `<generator>:<social>` joined with `&`.

Generator keys: `gan1` (StyleGAN), `gan2` (StyleGAN2), `gan3` (StyleGAN3), `sd15` (StableDiffusion1.5), `sd2` (StableDiffusion2), `sd3` (StableDiffusion3), `sdXL` (StableDiffusionXL), `flux` (FLUX.1), `realFFHQ` (FFHQ [6], real), `realFORLAB` (FORLAB [7], real).

Social keys: `pre` (not shared on a social network), `fb` (Facebook), `tl` (Telegram), `tw` (X).

The split does not need to be regenerated: it holds source IDs, not generator choices.

**New dataset.** Use the same folder layout as DeepShield (`<Social>/{Real,Fake}/<generator>/...`, with `Social` in `PreSocial`, `Facebook`, `Telegram`, `X`), generate a split with `support/json_compile.py`, and set `dataset_path` / `split_file` in the configs.

## 8. Outputs

| What | Where |
|---|---|
| Checkpoints | `detectors/<MODEL>/checkpoint/<RUN>/weights/best.pt` |
| Labeled test results | `results.csv` below `detectors/<MODEL>/results/<RUN>/` (one per scenario) |
| Single-image results | the path given to `--output` |
| Logs | `logs/` |

Checkpoints, results, logs, and downloaded weights are git-ignored.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Dataset directory not found` | `global.dataset_path` still `<PATH_TO_DATASET>` in that config |
| `Model weights not found` | Wrong run name, or weights not at `checkpoint/<RUN>/weights/best.pt` (see 3.1) |
| MDMF fails for `pretrained` | Run `detectors/MDMF/prepare.py` (3.1) |
| P2G fails on CPU | P2G needs CUDA |
| Out of memory | Lower `--batch-size` (PGC / ForensicConcept / MDMF) or `detector_args` batch size |
| GPU 0 busy (shared machine) | Pick another GPU: `CUDA_VISIBLE_DEVICES=1 python launcher.py ... --device cuda:0` (P2G always uses `cuda:0`) |
| NumPy / SciPy binary error | Keep NumPy < 2, as pinned in `requirements.txt` |
| PGC / MDMF: `Missing key(s) in state_dict ... q_proj` | `transformers` ≥ 5 is installed; re-run `pip install -r requirements.txt` (pins `transformers<5`) |
| No images found for `:tw` scenarios | Your copy names the X folder `Twitter/`; rename it (or `ln -s Twitter X`) inside every dataset root |

## License

Project code and documentation: [Apache-2.0](LICENSE); version in [VERSION](VERSION). Each detector keeps its own license (see `detectors/<MODEL>/LICENSE*` and `upstream/`). Downloaded weights and datasets are **not** covered by the root license and remain under their original terms.

2026 detectors: PGC code Apache-2.0 ([code](https://github.com/xiaoyu6868/PGC), [weights](https://huggingface.co/xiaoyuzhou68/PGC_ckpt)); ForensicConcept code Apache-2.0 ([code](https://github.com/EthanAdamm/FORENSICCONCEPT), [weights](https://huggingface.co/ethan225/ForensicConcept)); MDMF MIT ([code and checkpoint](https://github.com/ZBox1005/MDMF)). PGC and ForensicConcept weights are in separately licensed repositories; exact revisions and hashes are recorded in the ignored `weights-manifest.json` files.

Other methods considered: B-Free and QuAD (GRIP-UNINA) were reviewed but excluded because their licenses restrict use to informational and non-profit purposes; QuAD also calibrates across near-duplicate images instead of classifying one image.

## References

**Detectors (2023–2025)**

[1] D. Cozzolino, G. Poggi, R. Corvi, M. Nießner, L. Verdoliva. "Raising the Bar of AI-generated Image Detection with CLIP." *CVPR Workshops (CVPRW)*, 2024, pp. 4356–4366. [Code](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)

[2] C. Tan, H. Liu, Y. Zhao, S. Wei, G. Gu, P. Liu, Y. Wei. "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection." *CVPR*, 2024, pp. 28130–28139. [arXiv:2312.10461](https://arxiv.org/abs/2312.10461)

[3] F. Laiti, B. Liberatori, T. De Min, E. Ricci. "Conditioned Prompt-Optimization for Continual Deepfake Detection." *ICPR 2024*, Lecture Notes in Computer Science, Springer, 2025, pp. 64–79. [arXiv:2407.21554](https://arxiv.org/abs/2407.21554)

[4] S. Dell'Anna, A. Montibeller, G. Boato. "TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks." *IJCNN*, 2025. [arXiv:2504.20658](https://arxiv.org/abs/2504.20658)

[5] R. Corvi, D. Cozzolino, G. Zingarini, G. Poggi, K. Nagano, L. Verdoliva. "On the Detection of Synthetic Images Generated by Diffusion Models." *ICASSP*, 2023. [Code](https://github.com/grip-unina/DMimageDetection)

**Data**

[6] NVlabs. "Flickr-Faces-HQ Dataset (FFHQ)." <https://github.com/NVlabs/ffhq-dataset>

[7] M. Iuliani, M. Fontani, A. Piva. "A Leak in PRNU Based Source Identification — Questioning Fingerprint Uniqueness." *IEEE Access*, vol. 9, pp. 52455–52463, 2021.

**Detectors (2026)**

[8] X. Zhou, J. Fei, P. Yu, J. Xie, C. Cheng, Z. Xia. "PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection." arXiv:2605.21207, 2026. [Paper](https://arxiv.org/abs/2605.21207)

[9] M. Zhou, Z. Zhou, K. Sun, Y. Luo, J. Ji, X. Sun, R. Ji. "ForensicConcept: Transferable Forensic Concepts for AIGI Detection." arXiv:2606.07034, 2026. [Paper](https://arxiv.org/abs/2606.07034)

[10] B. Zhang, J. Zhu, Q. Wang, J. Liu, R. Tang. "Micro-Defects Expose Macro-Fakes: Detecting AI-Generated Images via Local Distributional Shifts." arXiv:2605.09296, 2026. [Paper](https://arxiv.org/abs/2605.09296)
