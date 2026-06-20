# DATA

该目录用于存放 StructuralGLIP 训练和推理所需的数据集。

CVC-300 示例结构：

```text
DATA/POLYP/
├── annotations/CVC-300_val.json
└── val/CVC-300/
    ├── images/
    └── masks/
```

其他数据集的目录结构和标注路径请查看对应的 `configs/pretrain/*.yaml`。配置文件中的数据路径均相对于项目根目录。
