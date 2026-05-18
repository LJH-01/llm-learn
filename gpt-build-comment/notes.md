# GPT 构建 — Transformer 详解

> 本课程讲解 Transformer 架构的实现，构建完整的 GPT 模型。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| **源代码** | [karpathy/makemore](https://github.com/karpathy/makemore) |
| **核心主题** | Self-Attention、Transformer、GPT-2 |

---

## 🎯 学习目标

1. 理解 Self-Attention 机制
2. 掌握 Multi-Head Attention
3. 理解 Transformer 架构
4. 实现完整的 GPT 模型

---

## 🧠 Self-Attention 机制

### 核心公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- Q (Query): 要查询的内容
- K (Key): 被查询的内容
- V (Value): 实际的内容值

### 直观理解

```
Q: "我想要查询什么？"
K: "我包含什么信息？"
V: "我的实际内容是什么？"

相似度 = Q · K^T
归一化 = softmax(相似度 / √d)
输出 = Σ(归一化权重 × V)
```

---

## 🎯 Multi-Head Attention

### 原理

将 Q, K, V 分成多个头并行计算，然后拼接

```
多头数 = n_head
每头维度 = d_k = d_model / n_head

head_i = Attention(Q_i, K_i, V_i)
output = concat(head_1, ..., head_n) * W_o
```

### 维度变化

```
输入 x: (B, T, d_model)
  ↓
QKV 投影: (B, T, 3*d_model)
  ↓
分割: Q, K, V 各 (B, T, d_model)
  ↓
重塑: (B, T, n_head, d_k) → (B, n_head, T, d_k)
  ↓
注意力: (B, n_head, T, d_k) → (B, n_head, T, d_k)
  ↓
拼接: (B, n_head, T, d_k) → (B, T, d_model)
  ↓
输出投影: (B, T, d_model)
```

---

## 🔧 Transformer 组件

### 1. LayerNorm

```
LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
```

### 2. 残差连接

```
x = x + sublayer(x)
```

### 3. 位置编码

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

## 🏗️ GPT 架构

### 完整结构

```
Token Embeddings + Position Embeddings
  ↓
Transformer Blocks (n_layer 次)
  ↓
LayerNorm
  ↓
LM Head (Linear)
  ↓
Output (vocab_size)
```

### GPT-2 参数

| 组件 | 参数 |
|------|------|
| vocab_size | 50257 |
| block_size | 1024 |
| n_layer | 12 |
| n_head | 12 |
| n_embd | 768 |
| 参数量 | 124M |

---

## 📊 实现要点

### 因果遮蔽

确保 attention 只能看到左边的位置：

```python
# 创建 causal mask
mask = torch.tril(torch.ones(T, T))
attn = attn.masked_fill(mask == 0, float('-inf'))
```

### 权重共享

token embedding 和 lm_head 共享权重：

```python
self.transformer.wte.weight = self.lm_head.weight
```

---

## 🔄 训练 vs 推理

### 训练

- 输入: (B, T) 的 token 序列
- 目标: (B, T) 的下一个 token
- 损失: CrossEntropy(logits, targets)

### 推理

- 逐步生成: 每次预测一个 token
- 自回归: 使用之前预测的结果作为下一步输入

---

> 📚 视频: [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
> 📦 代码: [karpathy/makemore](https://github.com/karpathy/makemore)