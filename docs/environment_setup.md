# StructuralGLIP Environment Setup

This document explains how to configure the StructuralGLIP runtime environment, compile the Mask R-CNN C++/CUDA extension, prepare model and data files, and troubleshoot common setup errors.

## 1. Source Code Layout

StructuralGLIP is based on two upstream projects:

- MIU-VL: `https://github.com/MembrAI/MIU-VL`
- Mask R-CNN Benchmark: `https://github.com/facebookresearch/maskrcnn-benchmark`

This repository already contains the integrated Mask R-CNN Benchmark style module:

```text
maskrcnn_benchmark/
```

The C++/CUDA source files that need local compilation are under:

```text
maskrcnn_benchmark/csrc/
├── vision.cpp
├── cpu/
└── cuda/
```

## 2. Recommended Environment

The recommended base environment is:

| Component | Version |
|---|---|
| OS | Linux |
| Python | 3.8 |
| CUDA / NVCC | 11.3 |
| PyTorch | 1.10.1+cu113 |
| torchvision | 0.11.2+cu113 |
| torchaudio | 0.10.1 |

Create and activate a conda environment:

```bash
conda create -n structuralglip python=3.8 -y
conda activate structuralglip
```

If the server provides multiple CUDA versions, switch to CUDA 11.3 first. Some servers provide a helper command similar to:

```bash
source /home/omnisky/switch-cuda.sh 11.3
```

On other servers, use the local CUDA module or environment management command instead. Verify the active CUDA compiler:

```bash
nvcc --version
```

## 3. Install PyTorch

Install the CUDA 11.3 PyTorch build:

```bash
pip install \
  torch==1.10.1+cu113 \
  torchvision==0.11.2+cu113 \
  torchaudio==0.10.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Check whether PyTorch can see CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

## 4. Install Project Dependencies

Install the recommended core dependencies first:

```bash
pip install -r requirements-core.txt
```

The full `requirements.txt` is a reference dump from the broader experiment environment and contains many task-specific packages. Use it only when you need to reproduce that full environment.

These package versions are often useful when debugging legacy Mask R-CNN compilation issues:

```bash
pip install numpy==1.20.3 cython==0.29.36 ninja yacs==0.1.8
pip install Pillow==9.5.0
pip install pycocotools scipy
```

The dependency files in this repository may use newer compatible versions. If a legacy Mask R-CNN compile error appears, try the versions above inside a fresh environment.

For OpenCV system library errors:

```bash
pip install opencv-python
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libglib2.0-dev libgl1
```

## 5. Compile Mask R-CNN C++/CUDA Extension

### Why It Must Be Recompiled

The Python package imports a local extension named:

```text
maskrcnn_benchmark._C
```

It is built from `setup.py` and `maskrcnn_benchmark/csrc/`. The extension binds operators such as:

- NMS, multi-label NMS, and soft NMS
- ROIAlign and ROIPool forward/backward functions
- sigmoid focal loss forward/backward functions
- deformable convolution and deformable pooling functions

This compiled `.so` file depends on the exact Python version, PyTorch version, CUDA version, compiler, and GPU architecture. A `_C*.so` copied from another machine usually fails with import errors, undefined symbols, or CUDA architecture errors. The reliable solution is to rebuild it locally in the active conda environment.

### Clean Old Build Files

Run from the repository root:

```bash
unset PYTHONPATH
rm -rf ~/.cache/torch_extensions
rm -rf build **/*.so maskrcnn_benchmark/_C*.so *.egg-info
```

If the shell does not expand `**/*.so`, use:

```bash
find . -name "*.so" -delete
rm -rf build maskrcnn_benchmark/_C*.so *.egg-info
```

### Set GPU Architecture

Set `TORCH_CUDA_ARCH_LIST` according to the GPU. A broad default list is:

```bash
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
```

Common examples:

```text
V100: 7.0
T4 / RTX 20 series: 7.5
A100: 8.0
RTX 30 series: 8.6
```

Check the GPU model with:

```bash
nvidia-smi
```

### Build

```bash
python setup.py clean
python setup.py build develop -v
```

In `setup.py`, CUDA sources are included when `torch.cuda.is_available()` is true. If PyTorch cannot see CUDA, the extension may fall back to CPU-only compilation and will not match the expected GPU workflow.

### Verify

```bash
python -c "from maskrcnn_benchmark import _C; print('maskrcnn_benchmark extension loaded')"
```

If this command succeeds, the extension was compiled and imported correctly.

## 6. Data And Model Files

Prepare model files under:

```text
MODEL/
├── bert-base-uncased/
└── glip_tiny_model_o365_goldg_cc_sbu.pth
```

Prepare CVC-300 data under:

```text
DATA/
└── POLYP/
    ├── annotations/
    │   └── CVC-300_val.json
    └── val/
        └── CVC-300/
            ├── images/
            └── masks/
```

Make sure `configs/pretrain/glip_Swin_T_O365_GoldG_polyp_cvc300.yaml` points to the same paths.

## 7. Run Zero-shot Inference

The simplest command is:

```bash
bash reference.sh
```

The script checks the config, model checkpoint, prompt JSON, and task config before calling `test.py`.

Equivalent direct command:

```bash
python test.py \
  --json blip_json/cvc300_val_noloc.json \
  --config-file configs/pretrain/glip_Swin_T_O365_GoldG_polyp_cvc300.yaml \
  --weight MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth \
  --task_config configs/pretrain/glip_Swin_T_O365_GoldG_polyp_cvc300.yaml \
  OUTPUT_DIR output/polyp-test \
  TEST.IMS_PER_BATCH 2 \
  SOLVER.IMS_PER_BATCH 2 \
  TEST.EVAL_TASK detection \
  DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
  DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
  DATASETS.USE_OVERRIDE_CATEGORY True \
  DATASETS.USE_CAPTION_PROMPT True
```

## 8. Common Problems

### `ImportError: cannot import name '_C'`

The Mask R-CNN extension was not built successfully, or the `.so` file came from another environment. Clean old artifacts and rebuild with `python setup.py build develop -v`.

### `undefined symbol` or PyTorch ABI errors

The compiled extension does not match the active PyTorch or compiler environment. Rebuild in the current conda environment. Do not reuse another server's `_C*.so`.

### CUDA architecture errors

Set `TORCH_CUDA_ARCH_LIST` to the target GPU architecture and rebuild.

### `module 'numpy' has no attribute 'float'`

Some legacy code may still use `np.float`, which was removed in newer NumPy versions. Replace `np.float` with `float` or `np.float32`, or use an older compatible NumPy version in a controlled environment.

### OpenCV `libGL.so` or `libgthread` errors

Install the system libraries:

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libglib2.0-dev libgl1
```

### `ImportError: cannot import name 'OFATokenizer' from 'transformers'`

This belongs to the optional OFA Hybrid workflow. The default StructuralGLIP inference path does not require OFA. If you need to reproduce the hybrid workflow, install OFA's custom transformers branch separately:

```bash
python -m pip uninstall -y transformers
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
python -m pip install ./OFA/transformers
```

Then download `OFA-base`, initialize Git LFS, and pull the real model weights:

```bash
git clone https://huggingface.co/OFA-Sys/OFA-base
mv OFA-base ofa-base
apt-get install git-lfs
cd ofa-base
git lfs install
git lfs pull
ls -lh pytorch_model.bin
cd ..
```

Only use this OFA setup when running the hybrid scripts. Avoid overwriting the verified StructuralGLIP base environment unless the hybrid workflow is required.

## 9. 中文说明

本文说明 StructuralGLIP 的环境安装、Mask R-CNN C++/CUDA 扩展编译、模型和数据准备，以及常见环境问题。

环境配置的关键点是：Mask R-CNN 的 C++/CUDA 扩展必须在目标服务器本地重新编译。该扩展名为 `maskrcnn_benchmark._C`，由 `setup.py` 和 `maskrcnn_benchmark/csrc/` 下的 C++/CUDA 文件生成，包含 NMS、ROIAlign、ROIPool、focal loss、deformable convolution、deformable pooling 等算子。

由于编译产物依赖 Python、PyTorch、CUDA、GCC 和显卡架构，不能直接复制其他机器上的 `_C*.so`。正确做法是在当前 conda 环境中清理旧文件后重新执行：

```bash
unset PYTHONPATH
rm -rf ~/.cache/torch_extensions
rm -rf build **/*.so maskrcnn_benchmark/_C*.so *.egg-info
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
python setup.py clean
python setup.py build develop -v
python -c "from maskrcnn_benchmark import _C; print('maskrcnn_benchmark extension loaded')"
```

如果最后一行可以正常输出，说明扩展已经通过当前环境编译并加载成功。之后可以运行：

```bash
bash reference.sh
```

进行 CVC-300 零样本推理。
