# Config Map

This repository organizes the paper experiments into three complementary config directories:

- `projects/configs/proposed_method/` for the main HAA-enhanced models
- `projects/configs/comparison/` for controlled comparisons and ablations
- `projects/configs/occ_study/` for the consolidated archive used during larger experiment sweeps

## 1. Main Paper Configs

The configs below correspond to the primary HAA-enhanced models discussed in the paper.

| Config | Purpose |
| --- | --- |
| `proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py` | Main HAA model with ResNeXt50, DCN, and the full multi-loss setup |
| `proposed-nuscenes-resnext101-dcn-haa-256x704-cb.py` | Alternative HAA model with a different backbone and resolution budget |
| `proposed-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb-trt.py` | TensorRT-facing wrapper for deployment experiments |
| `proposed-nuscenes-resnext101-dcn-haa-256x704-cb-trt.py` | TensorRT-facing wrapper for the ResNeXt101 variant |

## 2. Controlled Comparison Configs

The `comparison/` directory is grouped by the question each experiment answers.

| Theme | Pattern | What it is used for |
| --- | --- | --- |
| Baselines | `compare-baseline-*` | Reproducing the FlashOcc-style and stereo baseline families |
| Backbone study | `compare-backbone-*` | Measuring the effect of ResNet, ResNeXt, and DCN changes |
| HAA study | `compare-haa-*` | Isolating the proposed decoder refinement |
| Loss study | `compare-loss-*` | Testing focal, CE, and combined loss settings |
| Main setting references | `compare-main-*` | Keeping the main reference setup used in broader comparisons |
| Temporal study | `compare-temporal-*` | Comparing single-frame and temporal BEV variants |
| Resolution study | `compare-resolution-*` | Exploring larger-resolution stereo settings |

### Comparison config inventory

#### Baselines

- `compare-baseline-nuscenes-bevdet-occ-r50-256x704.py`
- `compare-baseline-nuscenes-bevstereo4d-occ-r50-256x704.py`
- `compare-baseline-nuscenes-bevstereo4d-occ-swinb-512x1408.py`

#### Backbone study

- `compare-backbone-nuscenes-r50-900x1600-focal-ce.py`
- `compare-backbone-nuscenes-resnext50-900x1600-focal-ce.py`
- `compare-backbone-nuscenes-resnext50-dcn-900x1600-focal-ce.py`

#### HAA study

- `compare-haa-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py`
- `compare-haa-nuscenes-resnext101-dcn-haa-256x704-cb.py`

#### Loss study

- `compare-loss-nuscenes-r50-900x1600-ce-only.py`
- `compare-loss-nuscenes-resnext101-dcn-256x704-ce-only.py`
- `compare-loss-nuscenes-resnext101-dcn-256x704-focal.py`
- `compare-loss-nuscenes-resnext101-dcn-256x704-focal-ce-cb.py`

#### Temporal and resolution references

- `compare-main-nuscenes-r50-256x704-m0-ce.py`
- `compare-temporal-nuscenes-r50-4d-stereo-256x704-ce.py`
- `compare-resolution-nuscenes-swinb-4d-stereo-512x1408-lr1e-2.py`
- `compare-resolution-nuscenes-swinb-4d-stereo-512x1408-lr2e-4.py`

## 3. Consolidated Archive in `occ_study/`

The `occ_study/` directory mirrors the broader experiment space in a flatter naming scheme. It is useful when you want one place that contains:

- `main-*` configs for the primary training runs
- `baseline-*` configs for comparison methods
- `ablation-*` configs for loss or architecture studies
- `deployment-*` configs for TensorRT and export experiments

Representative files include:

- `main-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py`
- `main-nuscenes-resnext101-dcn-haa-256x704-cb.py`
- `baseline-nuscenes-bevdet-occ-r50-256x704.py`
- `ablation-nuscenes-resnext101-dcn-256x704-focal-ce-cb.py`
- `deployment-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb-trt.py`

## 4. How to Choose a Config

- Use `proposed_method/` if you want the paper model directly.
- Use `comparison/` if you want a controlled baseline, ablation, or reference experiment.
- Use `occ_study/` if you want the full archive in one directory for scripting, sweeps, or deployment export.

For command examples, see `docs/USAGE.md`. For environment setup, see `docs/INSTALL.md`.
