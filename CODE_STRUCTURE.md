# StructuralGLIP 代码结构说明

## 1. 方法核心

StructuralGLIP 在 GLIP 的视觉语言深层融合框架上引入提示知识分支。主分支编码输入图像和目标名称，辅助分支将医学属性提示编码为分层知识表示。在每个融合层中，模型先根据提示与多尺度图像特征的相关性选择视觉 token，再利用所选视觉 token 反向筛选关键提示 token，最终将得到的结构化表示注入主分支。

这一设计对应三个主要模块：

| 模块 | 作用 | 主要代码位置 |
|---|---|---|
| 双分支表示 | 分别处理目标名称与属性提示 | `modeling/detector/generalized_vl_rcnn.py`、`modeling/language_backbone/bert_model.py` |
| 双向特征选择 | 在视觉区域与提示 token 之间进行分层筛选 | `utils/fuse_helper.py` |
| 深层视觉语言融合 | 将筛选后的结构化表示用于多层跨模态交互 | `modeling/rpn/vldyhead.py`、`utils/fuse_helper.py` |

类别级提示可组织颜色、形状、纹理和位置等属性，在不同样本间共享；筛选过程则根据当前图像动态提取相关知识。

## 2. 核心调用流程

零样本推理的主调用关系如下：

```text
reference.sh
  └── test.py
      ├── 读取 configs/*.yaml
      ├── build_detection_model()
      │   └── GeneralizedVLRCNN
      │       ├── Swin Transformer 视觉骨干
      │       ├── BERT 主分支：编码目标名称
      │       ├── BERT 辅助分支：编码提示知识
      │       └── VLDyHead
      │           ├── 选择相关多尺度视觉 token
      │           ├── 选择关键提示 token
      │           └── 执行深层视觉语言融合
      ├── make_data_loader()
      ├── DetectronCheckpointer.load()
      └── engine/inference_vqa.py
          ├── 读取 blip_json/*.json
          ├── 构造文本查询
          ├── 执行模型推理
          └── 保存并评估预测结果
```

## 3. 根目录文件

| 文件 | 作用 |
|---|---|
| `test.py` | 医学数据集零样本推理入口。 |
| `reference.sh` | CVC-300 推理命令示例。 |
| `setup.py` | 编译 `maskrcnn_benchmark._C` 扩展。 |
| `requirements-core.txt` | 建议安装的核心 Python 依赖。 |
| `requirements.txt` | 完整实验环境依赖参考。 |

## 4. 配置目录

### `configs/pretrain/`

包含 GLIP 预训练配置和医学数据集配置：

- `glip_Swin_T_O365_GoldG_polyp_*.yaml`：息肉数据集；
- `glip_Swin_T_O365_GoldG_bcdd_*.yaml`：血细胞数据集；
- `glip_Swin_T_O365_GoldG_isic.yaml`：皮肤病变数据；
- `glip_Swin_T_O365_GoldG_tbx11k.yaml`：结核病数据。

配置中的 `DATASETS.REGISTER` 定义 annotation 和图像目录，`MODEL` 定义视觉及语言骨干，`SOLVER` 定义训练参数。

### `configs/odinw_13/` 和 `configs/odinw_35/`

ODinW 跨域目标检测基准配置。

### `configs/flickr/` 和 `configs/lvis/`

Flickr phrase grounding 与 LVIS 检测评估配置。

## 5. `maskrcnn_benchmark/`

### `config/`

- `defaults.py`：全部默认配置项；
- `paths_catalog.py`：模型和数据路径查找。

### `data/`

- `build.py`：构建数据集、采样器、DataLoader 和 tokenizer；
- `datasets/`：COCO、Flickr、LVIS、VOC、grounding 等数据集；
- `transforms/`：缩放、翻转、归一化及数据增强；
- `samplers/`：分布式和迭代采样器。

### `modeling/`

- `detector/generalized_vl_rcnn.py`：StructuralGLIP 顶层模型；
- `backbone/`：Swin、ResNet、FPN 等视觉骨干；
- `language_backbone/bert_model.py`：BERT 文本编码器；
- `rpn/vldyhead.py`：视觉语言动态检测头及分层融合入口；
- `rpn/loss.py`：ATSS、token 和对齐相关损失；
- `roi_heads/`：两阶段检测相关模块。

### `engine/`

- `trainer.py`：训练循环；
- `inference.py`：通用推理；
- `inference_vqa.py`：当前医学零样本推理和文本查询逻辑；
- `predictor_glip.py`：GLIP 交互式预测封装。

### `layers/`

Python 封装的 NMS、ROIAlign、Deformable Convolution、Focal Loss 等算子。

### `csrc/`

C++/CUDA 扩展源码。`setup.py` 会将其编译为 `maskrcnn_benchmark._C`。生成的 `.so` 与本机环境绑定。

### `structures/`

BoxList、mask、keypoint 和 image list 等检测任务数据结构。

### `solver/`

优化器和学习率调度器。

### `utils/`

checkpoint、分布式通信、日志、AMP 和配置保存等辅助功能。其中 `fuse_helper.py` 实现多尺度视觉特征与提示 token 的双向选择及跨模态融合，是结构化表示的核心实现。

## 6. `tools/`

| 文件 | 作用 |
|---|---|
| `train_net.py` | 标准训练入口。 |
| `finetune.py` | 微调入口。 |
| `test_net.py` | 通用目标检测测试。 |
| `test_grounding_net.py` | Grounding 测试。 |
| `visualize_grounding_net.py` | 推理结果可视化。 |
| `eval_all.py` | 批量评估输出目录中的 checkpoint。 |
