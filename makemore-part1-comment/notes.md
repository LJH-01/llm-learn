# Makemore Part 1 — 字符级语言模型详解

> 本课程讲解字符级语言模型的基本概念：Bigram、MLP、数据处理、训练与采样。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| **源代码** | [karpathy/makemore](https://github.com/karpathy/makemore) |
| **核心主题** | 字符级语言模型、Bigram 模型、MLP 语言模型 |

---

## 🎯 学习目标

1. 理解什么是语言建模
2. 掌握 PyTorch Tensor 的使用技巧
3. 理解训练/采样/评估的完整流程
4. 实现字符级名字生成器

---

## 📖 课程大纲

### 1. 语言建模基础

**什么是语言模型？**
- 给定前文，预测下一个词的概率分布
- P(下一个词 | 前文)
- 衡量一句话出现的概率

**字符级 vs 词级：**
- 字符级：最小单位是字符（a-z, 中文字符等）
- 词级：最小单位是词
- 字符级优点：词汇表小，不需要处理未知词

### 2. Bigram 模型 — 最简单的 baseline

**原理：**
- 只根据前一个字符预测下一个字符
- P(w_i | w_{i-1})
- 无需训练嵌入，直接查表

**数学表达：**
```
P("hello") = P("h"|"<START>") × P("e"|"h") × P("l"|"e") × P("l"|"e") × P("o"|"l")
```

**实现：**
```python
class Bigram(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        # 查找表：logits[i][j] = 给定字符 i，下一个字符是 j 的分数
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def forward(self, idx):
        # idx: (B, T) 字符索引
        # 输出: (B, T, vocab_size) 每个位置的 logits
        logits = self.logits[idx]
        return logits
```

**维度：**
```
输入: idx (B, T) — 批量大小 B，序列长度 T
输出: logits (B, T, vocab_size)

参数: (vocab_size, vocab_size)
```

### 3. MLP 语言模型

**原理（Bengio 2003）：**
1. 将前 block_size 个字符的嵌入拼接
2. 通过 MLP 计算下一个字符的 logits

**架构图：**
```
输入: idx = [0, 5, 12, 3]  (<START>, a, b, c)

嵌入层:
  wte(idx[0]) → e0
  wte(idx[1]) → e1
  wte(idx[2]) → e2
  wte(idx[3]) → e3

拼接: [e0, e1, e2, e3] → (n_embd * 4)

MLP:
  Linear(n_embd*4, n_embd2) → Tanh → Linear(n_embd2, vocab_size) → logits
```

**维度变化：**
```
idx: (B, T)
  ↓
wte(idx): (B, T, n_embd)
  ↓
拼接: (B, T, n_embd * block_size)
  ↓
MLP: (B, T, vocab_size)
```

### 4. CharDataset — 数据处理

**编码格式：**
```
单词 "abc"
  ↓
编码: [1, 2, 3]  (a→1, b→2, c→3)
  ↓
输入 x: [0, 1, 2, 3]  (<START>, a, b, c)
目标 y: [1, 2, 3, 0]  (a, b, c, <STOP>)
```

**特殊令牌：**
- 0: <START> / <STOP>
- 1~vocab_size: 实际字符

**数据集划分：**
```
总数据: 约 32000 个名字
训练集: ~90% (约 29000)
测试集: ~10% (约 3000)
```

### 5. 训练流程

```python
# 1. 创建模型
model = Bigram(config)  # 或 MLP(config)

# 2. 前向传播
logits, loss = model(X, Y)  # X: 输入, Y: 目标

# 3. 计算损失
# cross_entropy(logits, targets) 自动处理
# logits: (B*T, vocab_size)
# targets: (B*T,)

# 4. 反向传播
loss.backward()

# 5. 更新参数
optimizer.step()
```

### 6. 采样生成

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    """
    自回归生成下一个字符

    Args:
        idx: (B, T) 条件序列
        temperature: 温度越高越随机
    """
    for _ in range(max_new_tokens):
        # 只取最后 block_size 个字符
        idx_cond = idx[:, -model.block_size:]

        # 前向传播
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # 只取最后一个位置

        # 采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 追加
        idx = torch.cat([idx, idx_next], dim=1)

    return idx
```

**Temperature 采样：**
- temperature = 1.0: 原始概率分布
- temperature → 0: 趋近于贪心选择（最高概率）
- temperature → ∞: 趋近于均匀分布

### 7. 评估指标

**困惑度（Perplexity）：**
```
PP = exp(loss)
```

- PP 越低，模型越好
- PP = 1 表示完美预测
- PP = vocab_size 表示随机猜测

---

## 🔧 关键代码解读

### 数据集类

```python
class CharDataset(Dataset):
    def __init__(self, words, chars, max_word_length):
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}  # char → index
        self.itos = {i: s for s, i in self.stoi.items()}    # index → char
```

### 损失计算

```python
def forward(self, idx, targets=None):
    logits = self.logits[idx]  # (B, T, vocab_size)

    if targets is not None:
        # 展平为 (B*T, vocab_size) 和 (B*T,)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1  # 忽略填充位置
        )
    return logits, loss
```

### 无限数据加载器

```python
class InfiniteDataLoader:
    # 使用有放回采样器实现无限迭代
    train_sampler = RandomSampler(dataset, replacement=True, num_samples=1e10)
```

---

## 📊 模型对比

| 模型 | 参数量 | 复杂度 | 效果 |
|------|--------|--------|------|
| Bigram | ~1K | 最简单 | baseline |
| MLP | ~100K | 中等 | 明显更好 |
| RNN/GRU | ~200K | 较高 | 更好 |
| Transformer | ~300K | 高 | 最佳 |

---

## 🚀 实验建议

1. **先用 Bigram 测试流程**：确保数据处理、训练循环、采样都正确
2. **实现 MLP**：理解嵌入层和全连接层
3. **调节超参数**：学习率、batch_size、embed_dim
4. **观察采样结果**：随着训练损失下降，生成的名字应该越来越"像名字"

---

## 📝 练习题

1. **实现平滑 Bigram**：在损失中加 Laplace 平滑
2. **实现词级语言模型**：将最小单位从字符改为词
3. **添加 dropout**：防止过拟合
4. **实现 beam search**：在采样时考虑多条路径

---

> 📚 视频链接: [Language Modeling](https://www.youtube.com/watch?v=PaCmpygFfXo)
> 📦 代码仓库: [karpathy/makemore](https://github.com/karpathy/makemore)