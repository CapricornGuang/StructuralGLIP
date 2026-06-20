# MODEL

该目录用于存放 StructuralGLIP 所需的语言模型和检测模型权重。

零样本 CVC-300 示例需要：

```text
MODEL/
├── bert-base-uncased/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── glip_tiny_model_o365_goldg_cc_sbu.pth
```

- `bert-base-uncased/`：BERT 配置、分词器和模型权重。
- `glip_tiny_model_o365_goldg_cc_sbu.pth`：GLIP-T 预训练权重。

模型准备方法见根目录 `README.md`。
