# StructuralGLIP

StructuralGLIP is a medical zero-shot object detection framework based on GLIP and Mask R-CNN Benchmark. Instead of directly concatenating medical prompts with category names, StructuralGLIP encodes prompts as a latent knowledge bank and dynamically selects image-relevant visual and language features. This design builds a more fine-grained structured representation for medical vision-language detection.

## Key Features

- **Dual-branch prompt modeling**: the main branch handles the image and target category, while the auxiliary branch independently encodes medical prompts to reduce semantic interference.
- **Hierarchical mutual selection**: prompt features first select relevant visual regions, and selected visual features further select important prompt tokens.
- **Reusable category-level knowledge**: category attributes such as color, shape, texture, and location can be reused across samples and dynamically matched to each image.

## Documentation

- [Project summary](docs/project_summary.md): a conceptual overview of StructuralGLIP, including motivation, method design, and experimental conclusions.
- [Environment setup](docs/environment_setup.md): a detailed setup guide, including Mask R-CNN C++/CUDA extension compilation and common environment issues.
- [Code structure](docs/code_structure.md): a module-level overview of the repository.
- [Chinese README](README_zh-CN.md): Chinese project overview and quick start.

## Repository Structure

```text
StructuralGLIP/
├── blip_json/               # Text descriptions and prompts for medical images
├── configs/                 # Model, dataset, and experiment configs
├── DATA/                    # Local dataset directory
├── docs/                    # Project summary, environment setup, and code structure notes
├── knowledge/               # ODinW knowledge configs
├── maskrcnn_benchmark/      # Model, data, training, evaluation, and CUDA ops
├── MODEL/                   # BERT and GLIP weight directory
├── tools/                   # Training, testing, and visualization entry points
├── reference.sh             # CVC-300 zero-shot inference example
├── requirements-core.txt    # Recommended core dependencies
├── requirements.txt         # Full environment reference
├── setup.py                 # C++/CUDA extension build entry
└── test.py                  # Medical zero-shot inference entry
```

## Recommended Environment

- Linux
- Python 3.8
- CUDA / NVCC 11.3
- PyTorch 1.10.1+cu113
- torchvision 0.11.2+cu113
- torchaudio 0.10.1

Create the environment:

```bash
conda create -n structuralglip python=3.8 -y
conda activate structuralglip
```

Install PyTorch:

```bash
pip install \
  torch==1.10.1+cu113 \
  torchvision==0.11.2+cu113 \
  torchaudio==0.10.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Install core dependencies:

```bash
pip install -r requirements-core.txt
```

`requirements.txt` records the broader experiment environment. For reproducing this repository, start with `requirements-core.txt` and install extra packages only when a specific task requires them.

## Build Mask R-CNN C++/CUDA Extension

The extension must be rebuilt on the target server. Do not reuse `_C*.so` files generated under another Python, PyTorch, or CUDA environment.

```bash
unset PYTHONPATH
rm -rf ~/.cache/torch_extensions
rm -rf build maskrcnn_benchmark/_C*.so *.egg-info

# Adjust according to the target GPU. This example covers V100, T4/RTX 20, A100, and RTX 30.
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

python setup.py clean
python setup.py build develop -v
```

Verify the extension:

```bash
python -c "from maskrcnn_benchmark import _C; print('extension loaded')"
```

If OpenCV reports missing system libraries:

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libglib2.0-dev libgl1
```

See [docs/environment_setup.md](docs/environment_setup.md) for more details about Mask R-CNN compilation, CUDA architecture, and common errors.

## Prepare Models

```text
MODEL/
├── bert-base-uncased/
│   ├── config.json
│   ├── pytorch_model.bin        # or model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── glip_tiny_model_o365_goldg_cc_sbu.pth
```

- Download Hugging Face `bert-base-uncased` into `MODEL/bert-base-uncased/`.
- Download the GLIP-T O365/GoldG/CC/SBU pretrained checkpoint into `MODEL/`.
- More model and dataset resources can be found in [MIU-VL](https://github.com/MembrAI/MIU-VL).

The code loads BERT from the repository-relative path `MODEL/bert-base-uncased`.

## Prepare Data

CVC-300 example:

```text
DATA/POLYP/
├── annotations/
│   └── CVC-300_val.json
└── val/
    └── CVC-300/
        ├── images/
        └── masks/
```

Related config:

```text
configs/pretrain/glip_Swin_T_O365_GoldG_polyp_cvc300.yaml
```

For other datasets, check `DATASETS.REGISTER` in the corresponding YAML file. Dataset paths should be written relative to the repository root.

## CVC-300 Zero-shot Inference

After preparing data and model weights, run:

```bash
bash reference.sh
```

Default paths:

```text
Config: configs/pretrain/glip_Swin_T_O365_GoldG_polyp_cvc300.yaml
Weight: MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
Prompt JSON: blip_json/cvc300_val_noloc.json
Output: output/polyp-test
```

You can override paths temporarily:

```bash
MODEL_CHECKPOINT=/path/to/model.pth \
OUTPUT_DIR=output/cvc300 \
bash reference.sh
```

Equivalent full command:

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

## Other Entry Points

```bash
# Training
python tools/train_net.py --config-file <config.yaml> OUTPUT_DIR <output_dir>

# General object detection test
python tools/test_net.py --config-file <config.yaml> --weight <checkpoint.pth>

# Grounding test
python tools/test_grounding_net.py --config-file <config.yaml> --weight <checkpoint.pth>

# Visualization
python tools/visualize_grounding_net.py --help
```
