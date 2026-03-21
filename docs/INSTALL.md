# Installation Guide

This repository accompanies the paper "Plug-and-Play Height-Aware Decoder Refinement for Efficient Camera-Only 3D Semantic Occupancy Prediction". The codebase extends MMDetection3D with custom occupancy heads, losses, datasets, visualization tools, and CUDA operators for FlashOcc-style camera-only occupancy prediction.

## 1. Reference Environments

### Core training stack

| Package | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 1.10.0 |
| torchvision | 0.11.0 |
| MMCV-Full | 1.5.3 |
| MMDetection | 2.25.1 |
| MMSegmentation | 0.25.0 |
| MMDetection3D | 1.0.0rc4 |
| NumPy | 1.23.5 |
| Numba | 0.53.0 |
| NetworkX | 2.2 |
| trimesh | 2.35.39 |
| setuptools | 59.5.0 |
| yapf | 0.40.1 |

### Deployment stack used for paper latency numbers

| Item | Configuration |
| --- | --- |
| Operating system | Ubuntu 20.04.6 LTS |
| CPU | Intel Core i9-12900K |
| System memory | 128 GiB |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| NVIDIA driver | 570.133.20 |
| CUDA Toolkit | 11.7 |
| cuDNN | 8.4.1.50 |
| MMDeploy | 0.9.0 |
| ONNX | 1.12.0 |
| TensorRT | 8.6.1.6 |
| Inference setting | Single GPU, batch size = 1 |

## 2. Create the Python Environment

```bash
conda create -n haa_occ python=3.8.20 -y
conda activate haa_occ
```

Install the PyTorch build that matches your CUDA toolchain. The examples below follow the `cu111` setup used by the original training stack:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

## 3. Install OpenMMLab Dependencies

```bash
pip install mmcv-full==1.5.3
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0
```

If you need a CUDA-specific MMCV wheel, install the matching prebuilt package for your PyTorch and CUDA version before continuing.

## 4. Install Runtime and Utility Packages

```bash
pip install -r requirements/runtime.txt
pip install numpy==1.23.5
pip install setuptools==59.5.0
pip install yapf==0.40.1
pip install pyquaternion
pip install open3d
pip install pycuda
```

The repository also expects the standard nuScenes and visualization dependencies already listed in `requirements/runtime.txt`, including `nuscenes-devkit`, `lyft_dataset_sdk`, `plyfile`, `scikit-image`, and `tensorboard`.

## 5. Build MMDetection3D and Project Extensions

Install the bundled MMDetection3D tree and the project plugin package in editable mode:

```bash
pip install -v -e ./mmdetection3d
pip install -v -e ./projects
```

This step compiles the custom CUDA operators under `projects/mmdet3d_plugin/ops/`. Make sure `CUDA_HOME`, a working C++ compiler, and the matching PyTorch CUDA headers are available.

## 6. Optional Deployment Environment

The paper reports FP16 TensorRT latency through MMDeploy. If you want to reproduce deployment-side benchmarking, install the deployment dependencies as well:

```bash
pip install Cython==0.29.24
pip install onnx==1.12.0
pip install onnxruntime-gpu==1.8.1
pip install spconv==2.3.6

cd mmdeploy
pip install -e .
cd ..
```

If TensorRT export is required, configure the relevant environment variables before building or running MMDeploy:

- `TENSORRT_DIR`
- `ONNXRUNTIME_DIR`
- `CUDA_HOME`
- any matching `LD_LIBRARY_PATH` entries for CUDA, cuDNN, and TensorRT

`ppl.cv/` is bundled in the repository for deployment workflows that require it, but it is not needed for standard training and evaluation.

## 7. Dataset Layout

The training and evaluation scripts expect the following directory structure:

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

For panoptic or ray-based evaluation, keep the additional annotation directory here as well:

```text
data/
  nuscenes/
    occ3d_panoptic/
```

The metadata files can be generated with:

```bash
python tools/create_data_bevdet.py
```

## 8. Next Steps

After installation, use `docs/USAGE.md` for training, evaluation, visualization, and benchmarking commands, and `docs/CONFIGS.md` to choose the config that matches the experiment you want to reproduce.
