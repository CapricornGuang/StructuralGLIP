# StructuralGLIP 中文说明

StructuralGLIP 是一个基于 GLIP 和 Mask R-CNN Benchmark 的医学图像零样本目标检测框架。针对传统方法将提示词直接拼接到类别名称、导致图像与描述对齐较粗的问题，StructuralGLIP 将医学提示编码为潜在知识库，并根据当前图像动态选择相关视觉与语言特征，形成细粒度的结构化表示。

## 核心特点

- **双分支提示建模**：主分支处理图像与目标名称，辅助分支独立编码医学提示，减少冗余描述对目标语义的干扰。
- **分层双向特征选择**：在多层视觉语言融合过程中，先由提示筛选相关视觉区域，再由视觉特征筛选关键提示 token。
- **类别级提示复用**：颜色、形状、纹理和位置等类别知识可在同类样本间复用，并针对每幅图像进行动态匹配。

## 文档入口

- [项目总结](docs/project_summary.md)：介绍 StructuralGLIP 的研究动机、方法设计和实验结论。
- [环境配置](docs/environment_setup.md)：说明环境安装、Mask R-CNN C++/CUDA 扩展编译和常见问题。
- [代码结构](docs/code_structure.md)：说明仓库模块结构和主要调用关系。

## 目录结构

```text
StructuralGLIP/
├── blip_json/               # 医学图像文本描述和 prompt
├── configs/                 # 模型、数据集及实验配置
├── DATA/                    # 数据集目录
├── docs/                    # 项目总结、环境配置和代码结构文档
├── knowledge/               # ODinW knowledge 配置
├── maskrcnn_benchmark/      # 模型、数据、训练、评估和 CUDA 算子
├── MODEL/                   # BERT 与 GLIP 权重目录
├── tools/                   # 训练、测试和可视化入口
├── reference.sh             # CVC-300 零样本推理示例
├── requirements-core.txt    # 建议安装的核心依赖
├── requirements.txt         # 完整环境依赖参考
├── setup.py                 # C++/CUDA 扩展编译入口
└── test.py                  # 医学数据零样本推理入口
```

## 推荐环境

- Linux
- Python 3.8
- CUDA / NVCC 11.3
- PyTorch 1.10.1+cu113
- torchvision 0.11.2+cu113
- torchaudio 0.10.1

创建环境并安装 PyTorch：

```bash
conda create -n structuralglip python=3.8 -y
conda activate structuralglip

pip install \
  torch==1.10.1+cu113 \
  torchvision==0.11.2+cu113 \
  torchaudio==0.10.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

安装核心依赖：

```bash
pip install -r requirements-core.txt
```

`requirements.txt` 记录了更完整的实验环境。复现当前整理后的项目时，建议优先使用 `requirements-core.txt`，再根据具体任务补充其他依赖。

## 编译 Mask R-CNN C++/CUDA 扩展

该扩展必须在目标服务器上重新编译，不能直接使用其他 Python、PyTorch 或 CUDA 环境生成的 `_C*.so`。

```bash
unset PYTHONPATH
rm -rf ~/.cache/torch_extensions
rm -rf build maskrcnn_benchmark/_C*.so *.egg-info

# 根据实际 GPU 修改。示例覆盖 V100、T4/RTX 20、A100、RTX 30。
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

python setup.py clean
python setup.py build develop -v
```

验证：

```bash
python -c "from maskrcnn_benchmark import _C; print('extension loaded')"
```

如果 OpenCV 报系统动态库错误：

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libglib2.0-dev libgl1
```

更详细的环境配置、Mask R-CNN 编译问题和 OFA Hybrid 说明见 [docs/environment_setup.md](docs/environment_setup.md)。

## 模型准备

```text
MODEL/
├── bert-base-uncased/
│   ├── config.json
│   ├── pytorch_model.bin        # 或 model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── glip_tiny_model_o365_goldg_cc_sbu.pth
```

- 将 Hugging Face `bert-base-uncased` 下载到 `MODEL/bert-base-uncased/`。
- 下载 GLIP-T 的 O365/GoldG/CC/SBU 预训练权重，放到 `MODEL/`。
- 更多模型和数据来源可参考 [MIU-VL](https://github.com/MembrAI/MIU-VL)。

代码通过仓库相对路径 `MODEL/bert-base-uncased` 加载 BERT。

## 数据准备

CVC-300 示例：

```text
DATA/POLYP/
├── annotations/
│   └── CVC-300_val.json
└── val/
    └── CVC-300/
        ├── images/
        └── masks/
```

对应配置：

```text
configs/pretrain/glip_Swin_T_O365_GoldG_polyp_cvc300.yaml
```

其他数据集请检查对应 YAML 中的 `DATASETS.REGISTER`。数据路径应相对于仓库根目录填写。

## CVC-300 零样本推理

准备好数据和权重后运行：

```bash
bash reference.sh
```

默认使用：

```text
配置：configs/pretrain/glip_Swin_T_O365_GoldG_polyp_cvc300.yaml
权重：MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
文本：blip_json/cvc300_val_noloc.json
输出：output/polyp-test
```

也可以临时覆盖路径：

```bash
MODEL_CHECKPOINT=/path/to/model.pth \
OUTPUT_DIR=output/cvc300 \
bash reference.sh
```

完整等价命令：

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

## 其他运行入口

```bash
# 训练
python tools/train_net.py --config-file <config.yaml> OUTPUT_DIR <output_dir>

# 通用目标检测测试
python tools/test_net.py --config-file <config.yaml> --weight <checkpoint.pth>

# Grounding 测试
python tools/test_grounding_net.py --config-file <config.yaml> --weight <checkpoint.pth>

# 可视化
python tools/visualize_grounding_net.py --help
```
