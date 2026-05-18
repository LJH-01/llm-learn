# Makemore Part 3 — 激活函数与 BatchNorm

> 本课程讲解激活函数统计、BatchNorm 原理和梯度诊断。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=P6sfmUTpUmc) |
| **源代码** | [karpathy/makemore](https://github.com/karpathy/makemore) |
| **核心主题** | 激活函数统计、BatchNorm 原理、梯度诊断 |

---

## 🎯 学习目标

1. 理解前向传播激活统计
2. 掌握 BatchNorm 原理与作用
3. 理解梯度流与初始化
4. 学会诊断深度网络训练问题

---

## 📊 激活函数统计

### 为什么要统计激活值？

神经网络的训练问题往往可以通过观察激活值的分布来诊断：
- 激活值方差过大 → 梯度爆炸
- 激活值方差过小 → 梯度消失
- 激活值集中在某一区域 → 表示死亡

### 常用统计量

```python
# 统计激活值的均值和标准差
mean = activations.mean()
std = activations.std()
min_val = activations.min()
max_val = activations.max()

# 打印每层的统计
print(f"Layer {i}: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")
```

---

## 🔄 BatchNorm 原理

### 核心思想

对每一层的输出进行标准化：
```
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

其中：
- mean, var 是 batch 的统计量（或移动平均）
- gamma, beta 是可学习参数
- eps 防止除零（通常 1e-5）

### 为什么有效？

1. **缓解内部协变量偏移**：每层输入分布稳定
2. **允许更高的学习率**：梯度更稳定
3. **有一定的正则化效果**：batch 噪声

### 维度变化

```
输入: (B, T, n_embd)
  ↓
计算 batch 均值和方差
  ↓
标准化
  ↓
缩放和平移 gamma, beta
  ↓
输出: (B, T, n_embd)
```

---

## 🔍 梯度诊断

### 观察梯度分布

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm = {grad_norm:.6f}")
```

### 常见问题

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| 梯度消失 | 层太深、激活函数饱和 | 残差连接、合适初始化 |
| 梯度爆炸 | 学习率太高、权重太大 | 梯度裁剪、减小学习率 |
| NaN 损失 | 除零、log(0) | 添加 eps、检查数据 |

---

## ⚙️ 权重初始化

### Xavier/Glorot 初始化

```python
# 适用于 tanh/sigmoid
nn.init.xavier_uniform_(layer.weight)
```

### Kaiming/He 初始化

```python
# 适用于 ReLU
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
```

### GPT-2 使用的初始化

```python
std = 0.02
if hasattr(module, 'NANOGPT_SCALE_INIT'):
    std *= (2 * n_layer) ** -0.5
nn.init.normal_(module.weight, mean=0.0, std=std)
```

---

## 📈 训练稳定性检查清单

- [ ] 损失在下降吗？
- [ ] 梯度范数是否稳定（不超过 100）？
- [ ] 激活值分布是否合理？
- [ ] 学习率是否合适？

---

> 📚 视频: [BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
> 📦 代码: [karpathy/makemore](https://github.com/karpathy/makemore)