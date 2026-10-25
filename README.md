# Disclosure

**AI tools were used to create the base for code and text documents for this project**

# VisualQC

**Prototype camera-based visual quality inspection with OpenCV and PyTorch**

VisualQC examines a light paper test piece on a dark work surface. Black printed circles simulate required manufactured features. The classical OpenCV pipeline counts candidate circular marks; a separate ResNet18 classifier predicts whether the captured piece is `good` or `defective`. The live application shows both judgments, reports their disagreement, and skips AI classification when the paper detector finds no component.

This is a proof of concept, **not a production-qualified industrial inspection system**.

## What is implemented

- Logitech C922 webcam capture (`capture.py`) with image saving and camera focus controls.
- Geometric circle inspection, result counters, and human labeling (`defect_detection.py`).
- Light-paper localization, rotated rectangle fitting, perspective warp, and 224 × 224 cropping (`paper_detection.py`).
- Offline dataset cropping (`prepare_crops.py`).
- ResNet18 transfer learning with class-weighted cross-entropy, augmentation, and last-block fine-tuning (`train.py`).
- Live comparison of neural-network and rule-based predictions, focus slider, confidence display, and approximate FPS (`live_inspection.py`).

## Hardware and environment

Developed on Windows with Python and VS Code. Demonstrated hardware: Logitech C922 camera, AMD Ryzen 7 7800X3D, NVIDIA RTX 4080 SUPER (16 GB VRAM), and 64 GB RAM. CUDA is optional; CPU inference is also supported by the application.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch/torchvision pair using the command from the [official PyTorch selector](https://pytorch.org/get-started/locally/) appropriate for your NVIDIA driver. Then:

```powershell
python -m pip install -r requirements.txt
```

The basic capture program is started with:

```powershell
python capture.py
```

For dataset preparation, organize **human-verified** photographs under `data/good/` and `data/defective/` and run:

```powershell
python prepare_crops.py
python audit_dataset.py
python train.py
```

Training reads `data_cropped/` and saves `models/visualqc_resnet18_cropped.pth`. Launch live inspection after a model checkpoint exists:

```powershell
python live_inspection.py
```

The pretrained ResNet18 weights may be downloaded on the first training run.

## Dataset integrity warning

The development archive reviewed on 25 October 2026 contained **240 labeled image files but only 157 distinct SHA-256 file contents**. There were 83 duplicate files in the labeled dataset. Its cropped directory contained 218 images (151 good and 67 defective). Do not interpret a random image-level split as an independent evaluation: exact duplicates and near-duplicate camera captures may occur in both training and test partitions.

Before publishing performance claims:

1. Deduplicate without discarding the original backup.
2. Record physical-sheet IDs and capture-session IDs.
3. Split *groups* (not individual frames) into train, validation, and a held-out test set.
4. Report a confusion matrix, defective recall, false acceptance rate, false rejection rate, and per-frame latency.

Run `python audit_dataset.py` to inspect exact file duplication. It does **not** delete or rename files.

## Experimental results

An earlier training run with 82 eligible images reported 71.93% training accuracy, 66.67% best validation accuracy, and 76.92% test accuracy (10/13). That result belongs to the initial model and a very small image-level holdout. It should not be presented as the accuracy of the newer cropped model or as industrial validation.

Live demonstrations show both successful classifications and disagreement cases, including a false PASS on a three-circle paper before cropping. The displayed softmax score is a model score, **not a calibrated certainty measure**. Requested camera FPS and observed processed FPS are different quantities.

## Repository contents

| File | Purpose |
| --- | --- |
| `capture.py` | Webcam capture |
| `defect_detection.py` | Rule-based inspection and labeling |
| `paper_detection.py` | Paper localization and crop |
| `prepare_crops.py` | Prepare cropped dataset |
| `train.py` | ResNet18 training and basic evaluation |
| `live_inspection.py` | Live dual-method inspection |
| `audit_dataset.py` | Read-only exact-duplicate check |
| `test_gpu.py` | CUDA smoke test |

## Limitations and planned work

- Paper localization assumes a light sheet on a darker background and can fail under other lighting or with other components.
- Circle counting alone does not verify the expected spatial positions, dimensions, or hole quality.
- Three classes of live state (`NO COMPONENT`, `PASS`, `FAIL`) are not equivalent to a three-class trained model; ResNet18 has two learned classes.
- The current training program randomly splits image indices and does not group near-duplicates or capture sessions.
- The dataset is small, uneven across defect types, and contains exact duplicates.
- Fixed 224 × 224 input can lose small defect details; the display FPS is lower than the camera's requested acquisition rate.

The next priority is **dataset integrity and independent validation**, followed by positional defect checks, better paper localization, confusion-matrix reporting, and controlled comparisons of OpenCV and ResNet18.

## Project status

Version 0.1 — educational proof of concept. Source scripts were retained from the author's supplied project archive; publication materials and a read-only dataset audit were added for this repository package.

Author: Antti Koljonen · Intelligent Systems Engineering student, Kajaani University of Applied Sciences.
