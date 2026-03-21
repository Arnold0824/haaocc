# Plug-and-Play Height-Aware Decoder Refinement for Efficient Camera-Only 3D Semantic Occupancy Prediction

<p align="center">
  Code release for a deployment-oriented decoder refinement on top of FlashOcc-style camera-only BEV occupancy pipelines.
</p>

<p align="center">
  <a href="docs/INSTALL.md">Installation</a> |
  <a href="docs/USAGE.md">Usage</a> |
  <a href="docs/CONFIGS.md">Configs</a> |
  <a href="docs/THIRD_PARTY.md">Third-Party Notes</a>
</p>

<p align="center">
  <img src="figs/architecture.png" width="94%" alt="Overall pipeline of the proposed method">
</p>

## Overview

This repository accompanies the paper "Plug-and-Play Height-Aware Decoder Refinement for Efficient Camera-Only 3D Semantic Occupancy Prediction". The project revisits the decoder stage of efficient BEV occupancy pipelines and keeps the deployment-friendly FlashOcc design philosophy: most computation stays in 2D BEV space, while voxel semantics are produced only at the final stage.

Our key observation is that channel-to-height decoding is efficient but spatially invariant along the height axis. This makes vertical semantic-geometric alignment harder for thin, small, and height-sensitive categories. To address this, we introduce a lightweight Height-Aware Attention (HAA) branch after the BEV encoder and fuse it with the standard decoder through zero-initialized gated residual modulation. The added branch uses only standard 2D convolutions and element-wise operators, so the baseline decoding path is preserved at initialization and the deployment profile stays lightweight.

## Highlights

- Plug-and-play HAA refinement for FlashOcc-style channel-to-height occupancy heads.
- Zero-initialized gated residual fusion that starts from the original decoder behavior and learns height-aware correction progressively.
- 33.41 mIoU vs. 32.83 on the strengthened ResNeXt50 + DCN baseline with only 0.30M extra parameters and 1.21 ms extra FP16 TensorRT latency.
- 33.79 mIoU vs. 32.08 for the full cumulative system on Occ3D-nuScenes.
- Largest gains on thin and height-sensitive categories such as barrier, bicycle, motorcycle, pedestrian, and traffic cone.

## Main Results

All latency numbers below are deployment-side FP16 TensorRT measurements with batch size 1.

| Variant | Params (M) | FLOPs (G) | FP16 latency (ms) | mIoU |
| --- | ---: | ---: | ---: | ---: |
| FlashOcc baseline | 44.74 | 248.57 | 14.79 | 32.08 |
| + ResNeXt50 | 44.22 | 251.83 | 15.35 | 32.46 |
| + DCN | 45.21 | 251.66 | 16.28 | 32.83 |
| + HAA | 45.51 | 263.55 | 17.49 | 33.41 |
| + Multi-loss (full) | 45.51 | 263.55 | 17.49 | 33.79 |

| Category group | Mean IoU gain over FlashOcc |
| --- | ---: |
| Thin / small / height-sensitive objects | +5.73 |
| Large vehicles | +0.28 |
| Ground-surface classes | -1.74 |
| Static background classes | +2.00 |

## HAA Occupancy Head

<p align="center">
  <img src="figs/cth_haa_head_detail.png" width="88%" alt="Detailed structure of the HAA-enhanced occupancy head">
</p>

<p align="center">
  The HAA branch predicts a spatially varying height response from the BEV feature map and refines decoded voxel logits through gated residual fusion.
</p>

## Visualization Gallery

<p align="center">
  <img src="figs/legend.png" width="72%" alt="Semantic legend used in the qualitative visualizations">
</p>

### BEV View Comparison

<table>
  <tr>
    <th>View</th>
    <th>Scene <code>2a271f85</code></th>
    <th>Scene <code>0490bd92</code></th>
    <th>Scene <code>1232e460</code></th>
    <th>Scene <code>2493d6ff</code></th>
  </tr>
  <tr>
    <td align="center"><strong>GT</strong></td>
    <td><img src="figs/bev_vis_compare/2a271f85c90a45a9a6f33d9cc281943e_gt.png" alt="GT BEV scene 2a271f85" width="100%"></td>
    <td><img src="figs/bev_vis_compare/0490bd92372a4a2d98c7136ba6ebcfce_gt.png" alt="GT BEV scene 0490bd92" width="100%"></td>
    <td><img src="figs/bev_vis_compare/1232e4600cb4400db443ae7e6a710c1c_gt.png" alt="GT BEV scene 1232e460" width="100%"></td>
    <td><img src="figs/bev_vis_compare/2493d6ff221e4dfda32f3f46dfd02fa3_gt.png" alt="GT BEV scene 2493d6ff" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>FlashOcc</strong></td>
    <td><img src="figs/bev_vis_compare/2a271f85c90a45a9a6f33d9cc281943e_flashocc.png" alt="FlashOcc BEV scene 2a271f85" width="100%"></td>
    <td><img src="figs/bev_vis_compare/0490bd92372a4a2d98c7136ba6ebcfce_flashocc.png" alt="FlashOcc BEV scene 0490bd92" width="100%"></td>
    <td><img src="figs/bev_vis_compare/1232e4600cb4400db443ae7e6a710c1c_flashocc.png" alt="FlashOcc BEV scene 1232e460" width="100%"></td>
    <td><img src="figs/bev_vis_compare/2493d6ff221e4dfda32f3f46dfd02fa3_flashocc.png" alt="FlashOcc BEV scene 2493d6ff" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Ours</strong></td>
    <td><img src="figs/bev_vis_compare/2a271f85c90a45a9a6f33d9cc281943e_ours.png" alt="Our BEV result scene 2a271f85" width="100%"></td>
    <td><img src="figs/bev_vis_compare/0490bd92372a4a2d98c7136ba6ebcfce_ours.png" alt="Our BEV result scene 0490bd92" width="100%"></td>
    <td><img src="figs/bev_vis_compare/1232e4600cb4400db443ae7e6a710c1c_ours.png" alt="Our BEV result scene 1232e460" width="100%"></td>
    <td><img src="figs/bev_vis_compare/2493d6ff221e4dfda32f3f46dfd02fa3_ours.png" alt="Our BEV result scene 2493d6ff" width="100%"></td>
  </tr>
</table>

The proposed decoder yields cleaner semantic boundaries and fewer false-positive occupied regions, especially around road edges, vegetation-manmade transitions, and small-object regions.

### 3D Occupancy Comparison

<table>
  <tr>
    <th>Scene</th>
    <th>FlashOcc Baseline</th>
    <th>Ours</th>
  </tr>
  <tr>
    <td align="center"><code>06cc9133</code></td>
    <td><img src="figs/full_compare/06cc9133c0684cf2b676cf5f7c49e88b_baseline.png" alt="FlashOcc 3D scene 06cc9133" width="100%"></td>
    <td><img src="figs/full_compare/06cc9133c0684cf2b676cf5f7c49e88b_ours.png" alt="Our 3D result scene 06cc9133" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><code>0aa136a1</code></td>
    <td><img src="figs/full_compare/0aa136a1b6f54e8faed7e1d08ebfaa82_baseline.png" alt="FlashOcc 3D scene 0aa136a1" width="100%"></td>
    <td><img src="figs/full_compare/0aa136a1b6f54e8faed7e1d08ebfaa82_ours.png" alt="Our 3D result scene 0aa136a1" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><code>1dd5d04e</code></td>
    <td><img src="figs/full_compare/1dd5d04e621f4fe791bff994708af69b_baseline.png" alt="FlashOcc 3D scene 1dd5d04e" width="100%"></td>
    <td><img src="figs/full_compare/1dd5d04e621f4fe791bff994708af69b_ours.png" alt="Our 3D result scene 1dd5d04e" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><code>2a271f85</code></td>
    <td><img src="figs/full_compare/2a271f85c90a45a9a6f33d9cc281943e_baseline.png" alt="FlashOcc 3D scene 2a271f85" width="100%"></td>
    <td><img src="figs/full_compare/2a271f85c90a45a9a6f33d9cc281943e_ours.png" alt="Our 3D result scene 2a271f85" width="100%"></td>
  </tr>
</table>

Compared with the baseline, our model produces more coherent upright structures and more plausible vertical occupancy geometry, including better trunk-canopy continuity and clearer small-object volumes.

## Repository Layout

- `projects/mmdet3d_plugin/`: occupancy datasets, models, losses, evaluation code, and custom CUDA operators.
- `projects/configs/proposed_method/`: main HAA-enhanced paper configs.
- `projects/configs/comparison/`: baseline, backbone, HAA, loss, temporal, and resolution comparison configs.
- `projects/configs/occ_study/`: consolidated archive of main, ablation, and deployment variants.
- `tools/`: training, evaluation, benchmarking, and visualization entry points.
- `docs/`: installation notes, usage recipes, config map, and third-party notes.
- `mmdetection3d/`, `mmdeploy/`, `ppl.cv/`: bundled upstream code used for training and deployment workflows.

## Quick Start

### Installation

See `docs/INSTALL.md` for the full environment setup. The paper experiments were built around:

- Python 3.8
- PyTorch 1.10.0
- MMCV 1.5.3
- MMDetection 2.25.1
- MMDetection3D 1.0.0rc4
- MMDeploy 0.9.0 for deployment-side latency measurements

### Data Preparation

Generate the nuScenes metadata used by the configs:

```bash
python tools/create_data_bevdet.py
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
```

For panoptic or ray-based evaluation, keep `occ3d_panoptic/` under `data/nuscenes/`.

### Training

Train the main HAA-enhanced model:

```bash
bash tools/dist_train.sh \
  projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py \
  4 \
  --work-dir work_dirs/proposed_resnext50_dcn_haa
```

Train the FlashOcc-style baseline used in comparison experiments:

```bash
bash tools/dist_train.sh \
  projects/configs/comparison/compare-baseline-nuscenes-bevdet-occ-r50-256x704.py \
  4 \
  --work-dir work_dirs/baseline_bevdet_occ_r50
```

### Evaluation

Evaluate mIoU:

```bash
bash tools/dist_test.sh \
  projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py \
  ckpts/proposed_resnext50_dcn_haa.pth \
  4 \
  --eval miou
```

Save predictions for visualization:

```bash
bash tools/dist_test.sh \
  projects/configs/proposed_method/proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py \
  ckpts/proposed_resnext50_dcn_haa.pth \
  4 \
  --eval miou \
  --eval-options show_dir=work_dirs/proposed_resnext50_dcn_haa/results
```

### Visualization

Render BEV comparisons:

```bash
python tools/analysis_tools/vis_occ_bev.py \
  --results-dir work_dirs/proposed_resnext50_dcn_haa/results \
  --compare-results-dir work_dirs/baseline_bevdet_occ_r50/results \
  --output-dir ./bev_vis/proposed_vs_baseline \
  --max-samples 50
```

Render full 3D comparison panels:

```bash
python tools/analysis_tools/vis_occ_full_compare.py \
  --bev-vis-dir ./bev_vis/proposed_vs_baseline \
  --baseline-results-dir work_dirs/baseline_bevdet_occ_r50/results \
  --ours-results-dir work_dirs/proposed_resnext50_dcn_haa/results \
  --output-dir ./full_compare/proposed_vs_baseline
```

Additional training, evaluation, benchmarking, and visualization recipes are collected in `docs/USAGE.md`.

## Documentation

- `docs/INSTALL.md`
- `docs/USAGE.md`
- `docs/CONFIGS.md`
- `docs/THIRD_PARTY.md`
