# Makemore Part 4 — 手写反向传播

> 本课程讲解手动实现反向传播，不使用 autograd。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=q8SA3rM6ckI) |
| **源代码** | [karpathy/makemore](https://github.com/karpathy/makemore) |
| **核心主题** | 手动反向传播、Cross Entropy、LayerNorm |

---

## 🎯 学习目标

1. 理解反向传播的数学原理
2. 手动实现梯度计算
3. 理解链式法则的应用
4. 掌握 Cross Entropy 的反向传播

---

## 🔑 链式法则

### 基本公式

```
∂L/∂x = ∂L/∂y * ∂y/∂x
```

### 常见操作的梯度

| 操作 | forward | backward |
|------|---------|----------|
| 加法 | c = a + b | da = dc, db = dc |
| 乘法 | c = a * b | da = b * dc, db = a * dc |
| ReLU | c = max(0, a) | da = dc if a > 0 else 0 |
| Softmax | p = exp(a)/Σexp(a) | da = p - y |

---

## 📊 Cross Entropy 反向传播

### 公式

```
CE = -Σ y_i * log(p_i)

∂CE/∂p_i = -y_i / p_i
∂CE/∂a_j = p_j - y_j  (当 i == j 时)
```

### 简化形式

当使用 one-hot 目标时：
```
∂CE/∂a_k = p_k - 1  (如果目标类别是 k)
∂CE/∂a_j = p_j      (如果目标类别不是 j)
```

---

## 🔧 LayerNorm 实现

### 公式

```
μ = mean(x)
σ² = var(x)
x_norm = (x - μ) / sqrt(σ² + eps)
y = gamma * x_norm + beta
```

### 反向传播

需要计算：
- ∂L/∂gamma
- ∂L/∂beta
- ∂L/∂x

---

## 🚀 训练步骤

1. **前向传播**：计算 loss
2. **反向传播**：计算所有参数的梯度
3. **参数更新**：使用梯度更新权重

```python
# 伪代码
for epoch in range(num_epochs):
    # 前向
    output = model(input)
    loss = cross_entropy(output, target)

    # 反向（手动）
    grad_output = cross_entropy_backward(target)
    for layer in reversed(layers):
        grad_input, grad_weights = layer.backward(grad_output)
        weights -= learning_rate * grad_weights
```

---

> 📚 视频: [Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
> 📦 代码: [karpathy/makemore](https://github.com/karpathy/makemore)