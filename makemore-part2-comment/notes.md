# Makemore Part 2 — MLP 语言模型

> 本课程讲解多层感知机（MLP）字符级语言模型的实现。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=TCH_1BHY58I) |
| **源代码** | [karpathy/makemore](https://github.com/karpathy/makemore) |
| **核心主题** | MLP 架构、过拟合/欠拟合、学习率调优 |

---

## 🎯 学习目标

1. 理解 MLP 语言模型的架构
2. 掌握嵌入层的使用
3. 理解训练/验证/测试集划分
4. 学会调节超参数

---

## 🧠 MLP 原理

### 架构图

```
输入: idx = [0, 5, 12, 3]  (<START>, a, b, c)

嵌入层:
  wte(idx[0]) → e0
  wte(idx[1]) → e1
  wte(idx[2]) → e2
  wte(idx[3]) → e3

滚动拼接:
  第1步: [e0]           → flatten → Linear → Tanh → Linear → logits
  第2步: [e0, e1]       → flatten → Linear → Tanh → Linear → logits
  第3步: [e0, e1, e2]   → flatten → Linear → Tanh → Linear → logits
  第4步: [e0, e1, e2, e3] → flatten → Linear → Tanh → Linear → logits

输出: logits (4, vocab_size) — 每个位置的下一个字符预测
```

### 维度变化

| 层 | 输入 | 输出 |
|---|------|------|
| Embedding | (B, T) | (B, T, n_embd) |
| Flatten | (B, T, n_embd) | (B, T, n_embd*block_size) |
| Linear + Tanh | (B, T, n*n) | (B, T, n_embd2) |
| Linear | (B, T, n2) | (B, T, vocab_size) |

---

## ⚠️ 过拟合与欠拟合

### 诊断方法

1. **观察训练损失 vs 测试损失**
   - 训练损失 << 测试损失 → 过拟合
   - 两者都高 → 欠拟合

2. **解决方法**

| 问题 | 解决方案 |
|------|----------|
| 过拟合 | 增加正则化、减小模型、增加数据 |
| 欠拟合 | 增大模型、训练更久、调整学习率 |

---

## 📊 学习率调优

### 建议范围

| 优化器 | 学习率范围 |
|--------|-----------|
| AdamW | 1e-4 ~ 1e-3 |
| SGD | 0.1 ~ 1.0 |

### 学习率调度

```python
# 简单的方法：保持恒定学习率
# 复杂的方法：使用学习率调度器

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10000, eta_min=1e-5
)
```

---

## 🔬 超参数一览

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| n_embd | 64 | 嵌入维度 |
| n_embd2 | 64 | MLP 隐藏层维度 |
| block_size | 8 | 上下文长度 |
| batch_size | 32 | 批次大小 |
| learning_rate | 5e-4 | 学习率 |
| weight_decay | 0.01 | 权重衰减 |

---

## 🚀 运行示例

```bash
# 训练 MLP 模型
python makemore_part2.py --input-file names.txt --n-embd 128 --n-embd2 256 --max-steps 5000

# 采样生成
python makemore_part2.py --sample-only --checkpoint out/model.pt
```

---

> 📚 视频: [MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
> 📦 代码: [karpathy/makemore](https://github.com/karpathy/makemore)