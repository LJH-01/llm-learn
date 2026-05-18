# Neural Networks: Zero to Hero — GPT 学习路线图

> ⭐ 基于 Andrej Karpathy 的 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) 构建的中文学习指南
>
> 本仓库包含视频的详细中文注释、代码解读和学习笔记。

---

## 🎯 学习路线图总览

```
阶段一: 神经网络基础 (Lectures 1-2)
    ↓
阶段二: MLP 与现代训练技术 (Lectures 3-6)
    ↓
阶段三: Transformer 核心 (Lectures 7-8) ⭐
    ↓
阶段四: GPT 训练与微调 (Lecture 9+)
    ↓
阶段五: 底层实现与进阶 (可选)
```

---

## 📚 阶段一: 神经网络基础

### Lecture 1: 反向传播 (Backpropagation)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| **代码仓库** | [micrograd](https://github.com/karpathy/micrograd) |
| **中文注释版** | [micrograd-comment](./micrograd-comment/) |
| **核心内容** | 反向传播算法、梯度下降、PyTorch 基础 |

**学习目标**:
- 理解神经网络是如何学习的
- 掌握梯度和反向传播的核心概念
- 理解损失函数与优化的关系

**课程大纲**:
1. 从标量运算开始理解梯度
2. 构建计算图 (Computational Graph)
3. 反向传播的数学推导
4. 用 PyTorch 实现 micrograd

---

### Lecture 2: 语言建模入门 (Language Modeling)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-part1-comment](./makemore-part1-comment/) |
| **核心内容** | 字符级语言模型、Bigram 模型、PyTorch Tensor |

**学习目标**:
- 理解什么是语言建模
- 掌握 PyTorch Tensor 的使用技巧
- 理解训练/采样/评估的完整流程

**课程大纲**:
1. Bigram 字符级语言模型
2. torch.Tensor 的使用技巧
3. 模型训练与损失函数
4. 采样生成文本

---

## 📚 阶段二: MLP 与现代训练技术

### Lecture 3: 多层感知机 (MLP)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/TCH_1BHY58I) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-part2-comment](./makemore-part2-comment/) |
| **核心内容** | MLP 架构、过拟合/欠拟合、学习率调优 |

**核心概念**:
- 多层感知机结构
- 训练/验证/测试集划分
- 学习率调度
- 超参数调优

---

### Lecture 4: 激活函数与 BatchNorm

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/P6sfmUTpUmc) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-part3-comment](./makemore-part3-comment/) |
| **核心内容** | 激活函数统计、BatchNorm 原理、梯度诊断 |

**核心概念**:
- 前向传播激活统计
- 梯度流与初始化
- BatchNorm 原理与作用
- 深度网络训练稳定性

---

### Lecture 5: 手写反向传播 (Backprop Ninja)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/q8SA3rM6ckI) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-part4-comment](./makemore-part4-comment/) |
| **核心内容** | 手动实现反向传播、Cross Entropy、LayerNorm |

**核心概念**:
- 不使用 autograd，手写 backward
- 理解梯度如何流过计算图
- Cross Entropy 损失反向传播
- BatchNorm 反向传播推导

---

### Lecture 6: WaveNet 与 CNN

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/t3YJ5hKiMQ0) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [makemore-part5-comment](./makemore-part5-comment/) |
| **核心内容** | 卷积神经网络、膨胀卷积、层次化结构 |

**核心概念**:
- 树状结构 MLP → CNN
- 因果膨胀卷积 (Causal Dilated Convolutions)
- torch.nn 深入理解
- 典型开发流程演示

---

## 📚 阶段三: Transformer 核心 ⭐

### Lecture 7: Let's build GPT

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| **代码仓库** | [makemore](https://github.com/karpathy/makemore) |
| **中文注释版** | [gpt-build-comment](./gpt-build-comment/) |
| **核心内容** | Transformer 架构、Self-Attention、GPT 实现 |

**⭐ 这是课程的核心视频！建议优先观看。**

**核心概念**:
1. **Self-Attention 机制**
   - Query, Key, Value 投影
   - 注意力分数计算: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
   - Multi-Head Attention (多头注意力)

2. **Transformer 架构**
   - 编码器/解码器结构
   - 残差连接 (Residual Connection)
   - Layer Normalization

3. **GPT 模型**
   - GPT-2 架构详解
   - 因果遮蔽 (Causal Mask)
   - 文本生成采样

---

### Lecture 8: GPT Tokenizer

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://www.youtube.com/watch?v=zduSFxRajkE) |
| **代码仓库** | [minbpe](https://github.com/karpathy/minbpe) |
| **中文注释版** | [tokenizer-comment](./tokenizer-comment/) |
| **核心内容** | BPE 算法、Tokenization 问题、GPT Tokenizer |

**核心概念**:
- Byte Pair Encoding (BPE) 算法
- encode() / decode() 实现
- Tokenization 带来的问题
- 为什么理想情况下应删除此阶段

---

## 📚 阶段四: GPT 训练与微调

### Lecture 9: 复现 GPT-2 (124M)

| 项目 | 内容 |
|------|------|
| **视频** | [🎬 YouTube](https://youtu.be/l8pRSuU81PU) |
| **代码仓库** | [build-nanogpt](https://github.com/karpathy/build-nanogpt) |
| **中文注释版** | [nanogpt-build-comment](./nanogpt-build-comment/) |
| **核心内容** | 从零训练 GPT-2、使用 FineWeb 数据集 |

**核心概念**:
- GPT-2 模型配置 (124M 参数)
- FineWeb 数据集处理
- 训练监控与优化
- 生成文本质量评估

---

## 📚 阶段五: 底层实现与进阶 (可选)

### 底层训练

| 项目 | 内容 |
|------|------|
| **代码仓库** | [llm.c](https://github.com/karpathy/llm.c) |
| **中文注释版** | [llm-c-comment](./llm-c-comment/) |
| **核心内容** | 纯 C/CUDA 实现 LLM 训练，零依赖 |

### Llama2 推理

| 项目 | 内容 |
|------|------|
| **代码仓库** | [llama2.c](https://github.com/karpathy/llama2.c) |
| **中文注释版** | [llama2-c-comment](./llama2-c-comment/) |
| **核心内容** | 单文件纯 C 推理 Llama 2 |

### BPE 算法实现

| 项目 | 内容 |
|------|------|
| **代码仓库** | [minbpe](https://github.com/karpathy/minbpe) |
| **中文注释版** | [minbpe-comment](./minbpe-comment/) |
| **核心内容** | 最简 BPE 算法实现 |

---

## 🗂️ 仓库结构

```
karpathy-gpt-roadmap/
├── README.md                          # 本文件
│
├── 阶段一/                           # 神经网络基础
│   ├── micrograd-comment/            # Lecture 1 中文注释
│   └── makemore-part1-comment/       # Lecture 2 中文注释
│
├── 阶段二/                           # MLP 与训练技术
│   ├── makemore-part2-comment/       # Lecture 3 中文注释
│   ├── makemore-part3-comment/       # Lecture 4 中文注释
│   ├── makemore-part4-comment/       # Lecture 5 中文注释
│   └── makemore-part5-comment/       # Lecture 6 中文注释
│
├── 阶段三/                           # Transformer 核心 ⭐
│   ├── gpt-build-comment/            # Lecture 7 中文注释
│   └── tokenizer-comment/            # Lecture 8 中文注释
│
├── 阶段四/                           # 训练与微调
│   └── nanogpt-build-comment/         # Lecture 9 中文注释
│
└── 阶段五/                           # 底层实现 (可选)
    ├── llm-c-comment/
    ├── llama2-c-comment/
    └── minbpe-comment/
```

---

## 📖 学习建议

### 推荐学习顺序

```
第1-2周 → Lecture 1-2 (基础)
第3-4周 → Lecture 3-4 (MLP + BatchNorm)
第5-6周 → Lecture 5-6 (反向传播 + CNN)
第7周   → Lecture 7 ⭐ (GPT 核心 - 必看!)
第8周   → Lecture 8-9 (Tokenizer + 训练)
第9周+  → 实战项目
```

### 学习技巧

1. **边看边敲**: 每个视频都配套 Jupyter Notebook，先跟着敲一遍
2. **做笔记**: 每个 lecture 都有对应的注释文件夹，用于记录学习心得
3. **完成练习**: 视频描述中都有练习题，试着自己做
4. **实践项目**: 完成后尝试用自己的数据集微调模型

---

## 🔗 相关资源链接

### 原始仓库

| 仓库 | 描述 |
|------|------|
| [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | 主课程仓库 |
| [micrograd](https://github.com/karpathy/micrograd) | 反向传播实现 |
| [makemore](https://github.com/karpathy/makemore) | 字符级语言模型 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | GPT 训练库 |
| [build-nanogpt](https://github.com/karpathy/build-nanogpt) | GPT 从零构建 |
| [minbpe](https://github.com/karpathy/minbpe) | BPE Tokenizer |
| [llm.c](https://github.com/karpathy/llm.c) | C/CUDA 训练 |
| [llama2.c](https://github.com/karpathy/llama2.c) | 单文件 Llama2 |

### 视频播放列表

- **YouTube**: [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

---

## 📝 贡献指南

如果你想为仓库添中文注释或学习笔记:

1. Fork 本仓库
2. 在对应的 `-comment` 文件夹中添加你的笔记
3. 提交 Pull Request

---

## 📄 License

MIT License - 参见 [LICENSE](./LICENSE)

---

> 本项目仅供学习使用，内容版权归 Andrej Karpathy 所有。
> 视频链接来自 [YouTube](https://www.youtube.com/@AndrejKarpathy)