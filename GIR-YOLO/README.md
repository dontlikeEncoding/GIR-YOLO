# GIR-YOLO

GIR-YOLO is a lightweight frequency-aware residual enhancement for pedestrian detection in low-light visible images. The release contains the modified Ultralytics components, model configurations, training and evaluation scripts, and dataset conversion utilities used in the paper.

## Repository scope

This repository is an overlay for the Ultralytics codebase used in the experiments. It is not a redistribution of the LLVIP or CityPersons datasets, and it does not include training runs or model checkpoints. The files under `ultralytics/nn/` replace the corresponding files in the pinned Ultralytics source tree.

## Environment

The experiments used Python 3.8, PyTorch with CUDA support, and Ultralytics 8.0.221. Install the matching Ultralytics release first:

```bash
python -m pip install ultralytics==8.0.221
```

Then copy the contents of this repository's `ultralytics/nn/` directory into the corresponding `ultralytics/nn/` directory of that installation or of an Ultralytics 8.0.221 source checkout. The custom classes are registered in `ultralytics/nn/tasks.py` and `ultralytics/nn/modules/dgf_frb.py`.

The original experiments used PyTorch 2.x and CUDA. Exact GPU performance depends on the installed CUDA, PyTorch, and driver versions.

## Dataset preparation

Download LLVIP and CityPersons from their official release channels and convert annotations to YOLO format where necessary. Edit `configs/llvip.yaml` and `configs/citypersons.yaml` so that `path` points to the local dataset directory. The repository contains no image or annotation files.

Expected layout:

```text
LLVIP/
  images/train/  images/val/
  labels/train/  labels/val/
CityPersons_VOC/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

## Training

Run commands from the repository root. Place the base `yolov8n.pt` checkpoint in the working directory or pass an equivalent local checkpoint in the training scripts.

```bash
python scripts/train_input_residual_dgf.py
python scripts/train_v3_ablation.py configs/yolov8n_input_residual_dgf_s005.yaml strength_005
python scripts/train_input_residual_dgf_city.py configs/yolov8n_input_residual_dgf.yaml citypersons_gir
```

The scripts use 1024-pixel inputs, 120 epochs, AdamW, and GPU device 0 by default. Adjust batch size, workers, and device for the available hardware.

Generate the supplied ablation configurations with:

```bash
python scripts/make_v3_ablation_yamls.py
```

## Validation and benchmarking

Validation scripts expect checkpoints under `runs/` and the benchmark scripts expect baseline/custom weights under `weights/` or `runs/`. These files are intentionally not included:

```bash
python scripts/val_input_residual_dgf_best.py
python scripts/benchmark_yolo_fps.py
```

Use `tools/export_yolo_predictions.py` to export predictions and ground-truth boxes for a configured split. The XML conversion utility is intended for preparing CityPersons annotations.

## License and data

Ultralytics remains subject to its original license. Dataset files remain subject to the licenses and terms of their respective providers. This repository does not redistribute either dataset or third-party checkpoints.
