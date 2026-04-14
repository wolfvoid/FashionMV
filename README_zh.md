[English](README.md) | **中文**

# FashionMV: 基于多视角时尚数据的商品级组合图像检索

**FashionMV** 官方代码。

[[论文 (arXiv)]](https://arxiv.org/abs/2604.10297) | [[数据集 (Hugging Face)]](https://huggingface.co/datasets/yuandaxia/FashionMV) | [[模型 (Hugging Face)]](https://huggingface.co/yuandaxia/ProCIR)

## 数据构造流程

<p align="center">
  <img src="assets/data_pipeline.png" width="100%"/>
</p>

FashionMV 通过三阶段流水线构建：
1. **描述生成** — 将多视角商品图像输入多模态大语言模型，生成逐图和商品级描述（长/短两种）。
2. **幻觉过滤** — 使用另一个多模态大语言模型交叉验证每条描述与图像的一致性，检测并去除幻觉描述。
3. **CIR 三元组构造** — 通过多路候选检索（视觉相似度、长描述相似度、短描述相似度）筛选目标商品，再由多模态大语言模型生成描述差异的修改文本。

## 模型架构

<p align="center">
  <img src="assets/model_framework.png" width="100%"/>
</p>

**ProCIR** 基于 Qwen3.5-0.8B，采用感知-推理解耦的对话架构：
- **第一轮（感知）**：编码多视角源商品图像 → 源嵌入 `s`
- **第二轮（推理）**：基于完整对话上下文处理修改文本 → 查询嵌入 `q`

训练结合了 CIR 检索损失与基于描述的对齐（源端 & 文档端），以注入商品知识。知识注入支持两条路径：SFT 预训练或思维链（CoT）。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集与模型

从 [HuggingFace](https://huggingface.co/yuandaxia/ProCIR) 下载 **ProCIR 模型**，从 [HuggingFace Datasets](https://huggingface.co/datasets/yuandaxia/FashionMV) 下载 **FashionMV 标注数据**：

```
FashionMV/
├── model/                          # ProCIR 模型权重 (0.8B)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── data/
    ├── val_triplets.jsonl          # 32,718 验证集三元组
    ├── val_captions.jsonl          # 18,803 验证集描述
    ├── train_triplets.jsonl        # 188,015 训练集三元组
    └── train_captions.jsonl        # 108,428 训练集描述
```

### 3. 准备图像数据

FashionMV 使用三个公开数据集的图像，需要用户自行下载并按以下结构组织：

```
images/
├── deepfashion/
│   ├── WOMEN/Dresses/id_00004544/02/
│   │   ├── 02_1_front.jpg
│   │   ├── 02_2_side.jpg
│   │   └── ...
│   └── ...
├── f200k/
│   ├── 90456812/
│   │   ├── 90456812_0.jpeg
│   │   ├── 90456812_1.jpeg
│   │   └── ...
│   └── ...
├── fashiongen_val/
│   ├── 95636/
│   │   ├── 974.jpg
│   │   ├── 975.jpg
│   │   └── ...
│   └── ...
└── fashiongen_train/   （仅训练时需要）
    └── ...
```

#### DeepFashion

1. 下载 [In-shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)
2. 按原始目录结构，以商品 ID 组织图像

#### Fashion200K

1. 从 [fashion-200k](https://github.com/xthan/fashion-200k) 下载
2. 按商品 ID 分组到文件夹

#### FashionGen

> **注意**：FashionGen 官方网站目前已不可用。研究者可能需要通过 [Kaggle](https://www.kaggle.com/) 等其他渠道获取 `fashiongen_256_256_validation.h5`（以及可选的 `fashiongen_256_256_train.h5`）。我们不提供该数据集的下载链接。

1. h5 文件按行存储图像，每行有一个 `input_productID` 字段（int32）标识所属商品，同一商品的行是**连续存储**的。我们三元组/描述中的 `product_id` 与该字段直接对应。
2. 使用我们提供的脚本提取图像（无需映射文件，直接从 h5 读取 `input_productID`）：

```bash
# 验证集
python tools/prepare_fashiongen.py \
  --h5_path /path/to/fashiongen_256_256_validation.h5 \
  --output_dir /path/to/images/fashiongen_val

# 训练集（可选）
python tools/prepare_fashiongen.py \
  --h5_path /path/to/fashiongen_256_256_train.h5 \
  --output_dir /path/to/images/fashiongen_train
```

### 4. 运行评测

每个数据集使用**独立的 gallery** 分别评测。可以一次评测全部三个数据集，也可以通过 `--datasets` 参数选择特定数据集。

**参数说明：**

| 参数 | 是否必需 | 说明 |
|------|---------|------|
| `--model_path` | 是 | ProCIR 模型路径 |
| `--image_root` | 是 | 图像根目录，包含 `deepfashion/`、`f200k/`、`fashiongen_val/` |
| `--data_dir` | 是 | 数据目录，包含 `val_triplets.jsonl` |
| `--datasets` | 否 | 要评测的数据集，可选：`deepfashion`、`f200k`、`fashiongen_val`。默认评测全部 |
| `--output_dir` | 否 | 结果保存路径（默认 `./results`） |
| `--batch_size` | 否 | 批大小（默认 10） |

#### 评测全部数据集

```bash
python evaluate.py \
  --model_path /path/to/model \
  --image_root /path/to/images \
  --data_dir /path/to/data
```

#### 评测单个数据集（如仅 DeepFashion）

```bash
python evaluate.py \
  --model_path /path/to/model \
  --image_root /path/to/images \
  --data_dir /path/to/data \
  --datasets deepfashion
```

#### 评测指定数据集

```bash
python evaluate.py \
  --model_path /path/to/model \
  --image_root /path/to/images \
  --data_dir /path/to/data \
  --datasets deepfashion f200k
```

#### 多卡评测 (DDP)

```bash
torchrun --nproc_per_node=4 evaluate.py \
  --model_path /path/to/model \
  --image_root /path/to/images \
  --data_dir /path/to/data \
  --datasets deepfashion
```

> **提示**：你只需要准备你要评测的数据集的图像。例如，如果只评测 DeepFashion，则不需要下载 Fashion200K 和 FashionGen。

### 预期结果

ProCIR (0.8B)：

| 数据集 | R@5 | R@10 |
|--------|-----|------|
| DeepFashion | 89.2 | 95.1 |
| Fashion200K | 77.6 | 86.6 |
| FashionGen-val | 75.0 | 85.3 |
| **平均** | **80.6** | **89.0** |

## 项目结构

```
├── evaluate.py                  # 评测主入口
├── requirements.txt
├── README.md                    # English
├── README_zh.md                 # 中文
├── assets/                      # 图片资源
│   ├── data_pipeline.png        # 数据构造流程图
│   └── model_framework.png      # ProCIR 模型架构图
├── procir/                      # 核心模型与数据模块
│   ├── __init__.py
│   ├── model.py                 # FashionEmbeddingModel (Qwen3.5 封装)
│   ├── chat_utils.py            # Chat template 工具
│   ├── datasets.py              # CIRValDataset, ProductValDataset
│   └── collators.py             # DocCollator, CIRQueryCollator
└── tools/
    └── prepare_fashiongen.py    # FashionGen h5 图像提取
```

## 引用

```bibtex
@article{yuan2026fashionmv,
  title={FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data},
  author={Yuan, Peng and Mei, Bingyin and Zhang, Hui},
  year={2026}
}
```

## 许可证

代码：MIT 许可证。模型权重：遵循 Qwen3.5 许可证。数据标注：CC BY-NC 4.0。图像需从原始数据源获取，遵循其各自的许可协议。
