# Makemore Part 5 — WaveNet 与 CNN

> 本课程讲解卷积神经网络、膨胀卷积和层次化结构。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) |
| **源代码** | [karpathy/makemore](https://github.com/karpathy/makemore) |
| **核心主题** | CNN、膨胀卷积、WaveNet |

---

## 🎯 学习目标

1. 理解卷积神经网络原理
2. 掌握膨胀卷积（Dilated Convolution）
3. 理解因果膨胀卷积
4. 学会构建层次化特征提取

---

## 🧠 CNN 基础

### 卷积运算

```
输出[i] = Σ w[k] * 输入[i + k]  (对所有 k)
```

### 维度变化

| 参数 | 说明 |
|------|------|
| 输入通道 | C_in |
| 输出通道 | C_out |
| 卷积核大小 | k |
| 步长 | stride |
| 填充 | padding |

```
输入: (B, T, C_in)
输出: (B, T, C_out)
参数量: C_in * C_out * k
```

---

## 🔄 膨胀卷积 (Dilated Convolution)

### 原理

普通卷积：膨胀率 = 1，感受野 = k
膨胀卷积：膨胀率 = d，感受野 = k + (k-1)*(d-1)

### 示例

```
膨胀率 1: [x, x, x, x, x]  (3x3 卷积)
膨胀率 2: [x, _, x, _, x]  (3x3 卷积，感受野 5)
膨胀率 4: [x, _, _, _, x]  (3x3 卷积，感受野 9)
```

### WaveNet 中的膨胀卷积

```
层 1: 膨胀率 1
层 2: 膨胀率 2
层 4: 膨胀率 8
...
每层感受野 = 1 + 2 + 4 + ... + 2^(n-1) = 2^n - 1
```

---

## 🎯 因果卷积 (Causal Convolution)

### 约束

t 时刻的输出只能依赖 t 时刻及之前的输入

```
输出[t] = f(输入[0], 输入[1], ..., 输入[t])
```

### 实现方法

1. 填充到左边
2. 使用 mask（但在 PyTorch 中更常用 causal padding）

---

## 🏗️ WaveNet 架构

### 核心模块

```
门控激活: tanh(W_f * x) * sigmoid(W_g * x)

残差连接: y = x + F(x)

跳跃连接: skip = Σ LayerNorm(G(x))
```

### 完整结构

```
Input
  ↓
输入嵌入
  ↓
初始卷积
  ↓
膨胀卷积堆叠 (dilated 1, 2, 4, 8, ...)
  ↓
1x1 卷积
  ↓
输出投影
  ↓
Softmax
  ↓
Output
```

---

## 📊 CNN vs RNN vs Transformer

| 特性 | CNN | RNN | Transformer |
|------|-----|-----|-------------|
| 并行性 | 高 | 低 | 高 |
| 感受野 | 有限 | 全局 | 全局 |
| 长依赖 | 难 | 容易（梯度问题） | 容易 |
| 计算量 | O(n) | O(n) | O(n²) |

---

> 📚 视频: [WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
> 📦 代码: [karpathy/makemore](https://github.com/karpathy/makemore)