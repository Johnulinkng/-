# WiDAN: Auditable Cross-Sensor and Cross-Condition Fault Diagnosis

[中文说明](README_CN.md)

This repository contains a reproducible research framework for vibration-based fault diagnosis under sensor, operating-condition, channel-count, and noise shifts. It evaluates strict unsupervised domain adaptation (UDA): source-domain fault labels are available during training, while target-training fault labels are hidden.

> **Research status:** integrity audit **PASS**; registered CRA-v2A performance gate **FAILED**. The neural candidate is provided for research evaluation and negative-transfer analysis, not as a recommended production model.

## Scope

- 3-to-3 same-condition cross-sensor transfer
- 3-to-3 cross-condition and dual-shift transfer
- 3-to-1 transfer with target channels 0, 1, and 2 trained and evaluated independently
- Clean and AWGN 20/10/0 dB evaluation
- Linear Ridge and RBF kernel-ridge baselines
- CRA-v2A channel-robust neural candidate with auditable training, inference, and evaluation receipts

## Key results and limitations

The source-validation-selected linear Ridge baseline reached the following clean development-set Accuracy / Macro-F1 averages:

| Dataset | Main development matrix | Cross-condition / dual-shift 3-to-1 |
| --- | ---: | ---: |
| Water pump | 68.400 / 67.003% | 57.622 / 56.918% |
| Laboratory gearbox | 76.500 / 75.818% | 73.500 / 72.345% |
| SEU gearbox | 63.917 / 62.465% | 32.861 / 31.592% |

Against the common-anchor control, CRA-v2A improved both Accuracy and Macro-F1 in 7 of 16 paired comparisons, but its mean changes were -0.34 and -0.37 percentage points. Dual-shift Macro-F1 changed by -1.98 points, with three additional zero-recall cases. Therefore no model is registered as the recommended neural model.

The corrected 44-condition fixed-model estimates are 59.675 / 58.338% for linear Ridge and 55.893 / 53.666% for RBF-KRR. Because earlier invalid evaluation exposed final-set metadata, these are **post-exposure fixed-model estimates**, not fresh blinded test results.

## Repository layout

```text
.
|-- cra_v2/                 # Frozen CRA-v2A code, checkpoints, reports and verifier
|-- baselines/              # Ridge/RBF inference entry points and usage notes
|-- docs/                   # Acceptance matrix, evidence and reporting boundaries
|-- requirements-cpu.txt    # Reproduced CPU environment
|-- requirements-gpu.txt    # Validated WSL/CUDA core environment
|-- README.md
`-- README_CN.md
```

The large, immutable Ridge/RBF research snapshot is distributed as a GitHub Release asset instead of being committed to Git history.

## Installation

Create a clean environment. For CPU-only inspection:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-cpu.txt
```

The GPU lock records the environment used for the audited WSL run. Install the matching PyTorch build from the official PyTorch index before installing the remaining packages when necessary.

## Verify the frozen CRA-v2A package

```bash
python cra_v2/tools/verify_channel_robust_v2_delivery.py cra_v2
```

The verifier checks the frozen manifest, file hashes, checkpoint registry, experiment receipts, and the registered comparison structure. It does not require the original datasets.

## Inference

Select a checkpoint whose task, target channel, variant, sampling rate, and SHA-256 match `cra_v2/model_registry.json`:

```bash
python cra_v2/inference/predict_channel_robust_v2.py \
  --checkpoint cra_v2/models/candidates/<arm-id>/best_model.pth \
  --checkpoint-sha256 <sha256-from-registry> \
  --task-id PG-S1 \
  --variant candidate \
  --target-slots 1 \
  --sampling-rate-hz 24000 \
  --input /path/to/raw_windows.npy \
  --output /path/to/predictions.npz
```

Input must be finite `float32` NPY data shaped `[N, C, 2048]`. The tool does not infer sensor identity, reorder channels, or resample signals.

For the stronger Ridge/RBF baseline, download and unpack the `research_snapshot_v3` release asset, then follow `KERNEL_INFERENCE.md` in that package.

## Data policy

Raw signals, labels, prepared windows, private evaluation roots, and target-development/final label stores are intentionally excluded. Obtain each dataset from its rights holder and maintain local paths outside the repository. Target-training labels must remain hidden for strict UDA evaluation.

## Reproducibility boundary

- Model selection uses source validation only.
- Each 3-to-1 target channel has its own independently trained checkpoint.
- Target-final results must not be mixed with development results.
- Audit success proves package integrity, not model superiority.
- The available recordings do not establish independent-device or clinical-style generalization.

## Rights and third-party material

No project-level open-source license has been granted. Public visibility does not grant permission to copy, modify, redistribute, or sublicense this repository. See [RIGHTS_AND_LICENSE.md](RIGHTS_AND_LICENSE.md) and [the detailed third-party boundary](cra_v2/docs/V2_THIRD_PARTY_AND_RIGHTS.md).

