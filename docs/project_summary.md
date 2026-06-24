# StructuralGLIP Project Summary

This document summarizes the motivation, method design, and experimental conclusions of StructuralGLIP. The English version is placed first, followed by the Chinese version.

## English Version

StructuralGLIP is a medical zero-shot object detection framework based on GLIP and Mask R-CNN Benchmark. It focuses on a common challenge in vision-language detection: prompts are usually treated as a flat input sentence, so attributes such as color, shape, texture, and location are not explicitly organized. This can introduce redundant or noisy prompt tokens and weaken fine-grained image-text alignment.

The key idea of StructuralGLIP is to treat prompts as a latent knowledge bank instead of a plain input sequence. The model uses a dual-branch design and a dynamic mutual selection mechanism to retrieve the most relevant visual and semantic tokens for each image. In this way, prompt information becomes structured, reusable, and selectively involved in cross-modal fusion.

## Motivation

Zero-shot object detection aims to detect target objects without task-specific annotated training data. Models such as GLIP use text semantics as supervision and align image features with text features through cross-modal attention.

However, the standard GLIP-style pipeline has two important weaknesses:

- Semantic structure is missing. Prompts are directly concatenated into natural-language text, and internal attributes are not explicitly modeled.
- Prompt information is used inefficiently. All prompt tokens participate in matching with similar importance, even when some tokens are irrelevant to the current image.

These issues make the alignment relatively coarse. StructuralGLIP addresses them by changing the role of the prompt from "input text" to "organized knowledge".

## Method Overview

StructuralGLIP uses two branches:

- Main Branch: receives the image and target category name, and performs the main detection task.
- Auxiliary Branch: receives prompts and encodes them into a knowledge bank that provides structured semantic information.

The two branches decouple the target representation from auxiliary medical knowledge. This reduces interference between the detection target and the descriptive prompt.

## Mutual Selection

The core mechanism can be divided into four steps:

1. Prompt Encoding: encode prompts into a latent knowledge bank.
2. Visual Token Selection: select the Top-P visual tokens most related to the knowledge bank.
3. Prompt Token Selection: based on the selected visual tokens, select the Top-Q prompt tokens most related to the image.
4. Cross-modal Fusion: fuse the selected visual and prompt tokens through cross-modal attention.

This mutual selection process filters irrelevant semantic information and strengthens key attributes. Compared with using all prompt tokens equally, it changes the model from static full-token matching to dynamic knowledge retrieval.

## Prompt Design

StructuralGLIP supports two prompt types:

- Instance-level prompts: dynamically generated for each image, often by a VQA model.
- Category-level prompts: predefined category knowledge, such as color, shape, texture, and location.

The main finding is that category-level prompts are especially useful in StructuralGLIP. In standard GLIP, category-level prompts may hurt performance because they are mixed directly with target text. In StructuralGLIP, the prompt is placed in an auxiliary knowledge branch, so category knowledge can be selected and reused more effectively.

## Experiments And Conclusions

The paper evaluates StructuralGLIP on several medical imaging scenarios:

- Endoscopy datasets, such as Kvasir and ColonDB.
- Microscopy datasets, such as BCCD.
- Photography datasets, such as ISIC.
- Radiology datasets, such as TBX11K.

The reported tasks include zero-shot detection, zero-shot enhancement with limited fine-tuning data, and comparisons between instance-level and category-level prompts.

Main conclusions:

- StructuralGLIP performs better than GLIP in zero-shot and fine-tuning settings.
- Structured prompts improve generalization in medical images.
- Category-level prompts become more valuable when they are modeled as a knowledge bank.
- The method is more robust in complex semantic scenes because irrelevant prompt tokens can be filtered.

## Why It Works

From a modeling perspective, the difference can be summarized as:

```text
GLIP:
prompt = input sequence

StructuralGLIP:
prompt = retrievable knowledge bank
```

The improvement mainly comes from three design choices:

- Semantic structuring: prompt attributes are represented more clearly.
- Dynamic selection: irrelevant prompt and visual tokens are filtered through Top-K selection.
- Decoupled modeling: target text and auxiliary prompt knowledge are processed separately, reducing semantic conflict and distribution shift during fine-tuning.

## 中文版本

本文总结 StructuralGLIP 的研究动机、方法设计和实验结论。

StructuralGLIP 是一个基于 GLIP 和 Mask R-CNN Benchmark 的医学图像零样本目标检测框架。它关注现有视觉语言检测方法中的一个关键问题：prompt 往往被当作一段普通输入文本直接拼接到类别名称后面，颜色、形状、纹理、位置等内部属性没有被显式组织，容易引入冗余或噪声 token，导致图像与文本之间只能进行较粗粒度的对齐。

StructuralGLIP 的核心思想是把 prompt 从“输入文本”转化为“潜在知识库”。模型通过双分支结构和动态 Mutual Selection 机制，根据当前图像自适应选择最相关的视觉 token 和语义 token，使 prompt 信息变成结构化、可复用、可筛选的辅助知识。

## 研究动机

零样本目标检测希望在缺少特定任务标注数据的情况下，通过视觉语言模型完成目标检测。GLIP 这类方法使用文本语义作为监督信号，并通过跨模态注意力完成图像特征和文本特征的融合。

但传统 GLIP 范式存在两个问题：

- 语义结构缺失：prompt 通常只是自然语言拼接，内部属性没有被显式建模。
- 信息利用低效：所有 prompt token 在匹配中被近似等权使用，缺少针对当前图像内容的选择机制。

因此，传统方法更像是粗粒度对齐。StructuralGLIP 则把 prompt 视为可组织、可检索的知识结构，从而支持更细粒度的跨模态对齐。

## 方法概览

StructuralGLIP 采用双分支结构：

- 主分支：输入图像和目标类别名称，负责主要检测任务。
- 辅助分支：输入 prompt，并将其编码为知识库，为模型提供结构化语义信息。

这种设计将目标信息和辅助知识解耦，避免把大量描述性 prompt 直接混入目标类别文本，从而减少语义干扰。

## Mutual Selection 机制

核心流程可以拆成四步：

1. Prompt Encoding：把 prompt 编码为 latent knowledge bank。
2. Visual Token Selection：根据图像特征与知识库的相似度，选择 Top-P 个相关视觉 token。
3. Prompt Token Selection：基于已选视觉 token，从知识库中选择 Top-Q 个相关 prompt token。
4. Cross-modal Fusion：只使用筛选后的 token 做跨模态融合，实现更细的图文对齐。

Mutual Selection 的本质是用图像内容引导 prompt 信息选择，过滤无关语义，强化关键属性表达。相比传统方法中所有 prompt token 全量参与，它让模型从静态文本匹配转向动态知识检索。

## Prompt 设计

StructuralGLIP 支持两类 prompt：

- Instance-level prompt：针对单张图像动态生成，通常可由 VQA 模型生成。
- Category-level prompt：基于类别属性预定义，例如颜色、形状、纹理和位置等。

核心结论是：category-level prompt 在 StructuralGLIP 中表现更好。原因是它不再直接干扰 target 表示，而是作为辅助知识库参与动态选择，因此类别级知识可以被更稳定地复用。

## 实验结论

论文在多个医学影像任务上验证方法，包括：

- Endoscopy：如 Kvasir、ColonDB。
- Microscopy：如 BCCD。
- Photography：如 ISIC。
- Radiology：如 TBX11K。

实验包括 zero-shot detection、少量监督微调下的 zero-shot enhancement，以及不同 prompt 类型的对比。

主要结论：

- StructuralGLIP 在 zero-shot 和微调场景下均优于 GLIP。
- 结构化 prompt 可以提升医学图像零样本泛化能力。
- category-level prompt 在知识库建模方式下更有价值。
- 模型在复杂语义场景中更鲁棒，因为无关 prompt token 可以被过滤。

## 方法本质

从建模角度看，两者差异可以概括为：

```text
GLIP:
prompt = 输入文本

StructuralGLIP:
prompt = 可检索知识库
```

StructuralGLIP 有效的原因主要包括：

- 语义结构化：将颜色、形状等属性拆解并组织，避免语义混杂。
- 动态选择：通过 Top-K 筛选去除无关 prompt，强化关键语义。
- 解耦建模：prompt 作为辅助知识，target 作为检测目标，减少语义冲突和微调时的分布偏移。
