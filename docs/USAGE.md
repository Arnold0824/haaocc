# Usage Guide

This document focuses on the command flow used to reproduce the paper experiments, evaluate checkpoints, and regenerate the qualitative visualizations shown in the README.

## 1. Prepare the Dataset Metadata

Generate the nuScenes info files used by the configs:

```bash
python tools/create_data_bevdet.py
```

If an existing dataset needs coordinate updates, run:

```bash
python tools/update_data_coords.py
```

Expected layout:

```text
data/
  nuscenes/
    samples/
    sweeps/
    v1.0-trainval/
    gts/
    bevdetv2-nuscenes_infos_train.pkl
    bevdetv2-nuscenes_infos_val.pkl
    occ3d_panoptic/   # optional, for panoptic or ray-based evaluation
```

## 2. Train the Main Models

### HAA-enhanced paper model

```bash
bash tools/dist_train.sh \
  projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py \
  4 \
  --work-dir work_dirs/proposed_resnext50_dcn_haa
```

### Alternative HAA config

```bash
bash tools/dist_train.sh \
  projects/configs/proposed_method/proposed-nuscenes-resnext101-dcn-haa-256x704-cb.py \
  4 \
  --work-dir work_dirs/proposed_resnext101_dcn_haa
```

### FlashOcc-style baseline for controlled comparison

```bash
bash tools/dist_train.sh \
  projects/configs/comparison/compare-baseline-nuscenes-bevdet-occ-r50-256x704.py \
  4 \
  --work-dir work_dirs/baseline_bevdet_occ_r50
```

If you only need single-GPU debugging, replace the launcher with:

```bash
python tools/train.py <config> --work-dir <work_dir>
```

## 3. Evaluate Checkpoints

### mIoU evaluation

```bash
bash tools/dist_test.sh \
  projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py \
  ckpts/proposed_resnext50_dcn_haa.pth \
  4 \
  --eval miou
```

### Ray-IoU evaluation

```bash
bash tools/dist_test.sh \
  projects/configs/proposed_method/proposed-nuscenes-resnext101-dcn-haa-256x704-cb.py \
  ckpts/proposed_resnext101_dcn_haa.pth \
  4 \
  --eval ray-iou
```

### Save predictions for later visualization

```bash
bash tools/dist_test.sh \
  projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py \
  ckpts/proposed_resnext50_dcn_haa.pth \
  4 \
  --eval miou \
  --eval-options show_dir=work_dirs/proposed_resnext50_dcn_haa/results
```

Saved predictions are written under:

```text
work_dirs/proposed_resnext50_dcn_haa/results/<scene-name>/<sample-token>/pred.npz
```

## 4. Reproduce the Paper Visualizations

### Standalone occupancy rendering

```bash
python tools/analysis_tools/vis_occ.py \
  work_dirs/proposed_resnext50_dcn_haa/results \
  --root_path ./data/nuscenes \
  --save_path ./vis/proposed_resnext50_dcn_haa \
  --format image
```

### BEV comparison panels

```bash
python tools/analysis_tools/vis_occ_bev.py \
  --results-dir work_dirs/proposed_resnext50_dcn_haa/results \
  --compare-results-dir work_dirs/baseline_bevdet_occ_r50/results \
  --output-dir ./bev_vis/proposed_vs_baseline \
  --max-samples 50
```

### Full 3D comparison panels

```bash
python tools/analysis_tools/vis_occ_full_compare.py \
  --bev-vis-dir ./bev_vis/proposed_vs_baseline \
  --baseline-results-dir work_dirs/baseline_bevdet_occ_r50/results \
  --ours-results-dir work_dirs/proposed_resnext50_dcn_haa/results \
  --output-dir ./full_compare/proposed_vs_baseline
```

These scripts are the ones used to assemble the qualitative examples under `figs/`.

## 5. Benchmark the Model

### Throughput benchmark

```bash
python tools/analysis_tools/benchmark.py \
  projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py \
  ckpts/proposed_resnext50_dcn_haa.pth \
  --samples 200 \
  --log-interval 20
```

### Sequential benchmark

```bash
python tools/analysis_tools/benchmark_sequential.py \
  projects/configs/comparison/compare-temporal-nuscenes-r50-4d-stereo-256x704-ce.py \
  ckpts/compare_temporal_r50_4d_stereo.pth \
  --samples 200 \
  --log-interval 20
```

### FLOPs inspection

```bash
python tools/analysis_tools/get_flops.py \
  projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py
```

## 6. Deployment-Facing Configs

TensorRT-oriented wrapper configs are provided with the `-trt.py` suffix:

- `projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb-trt.py`
- `projects/configs/proposed_method/proposed-nuscenes-resnext101-dcn-haa-256x704-cb-trt.py`

The broader config archive under `projects/configs/occ_study/` also includes deployment wrappers used during larger experiment sweeps.
