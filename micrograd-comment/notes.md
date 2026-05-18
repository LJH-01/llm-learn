# micrograd 源代码中文详解

> micrograd 是 Andrej Karpathy 实现的一个精简自动微分引擎，是理解 PyTorch 反向传播机制的绝佳入门教材。

---

## 📁 文件结构

```
micrograd/
├── engine.py    # 核心：Value 类，实现自动求导
├── nn.py        # 神经网络层：Neuron, Layer, MLP
└── __init__.py  # 包导出（空文件）
```

---

## 🔥 engine.py — 自动求导引擎核心

### Value 类：一切的基础

```python
class Value:
    """ 存储单个标量值及其梯度 """
```

Value 是 micrograd 的核心，每个 Value 对象包含：

| 属性 | 类型 | 说明 |
|------|------|------|
| `data` | float | 实际存储的数值 |
| `grad` | float | 梯度值，初始化为 0 |
| `_backward` | function | 反向传播时执行的梯度计算函数 |
| `_prev` | set | 前驱节点集合，构建计算图 |
| `_op` | str | 操作类型，用于调试可视化 |

### 数据维度追踪

```
输入: Value(data=2.0)
       ↓
  ┌─────────────────┐
  │     Value       │
  │  data: 2.0      │  ← 标量值
  │  grad: 0        │  ← 梯度（反向传播时填充）
  │  _prev: {}      │  ← 前驱节点
  │  _op: ''        │  ← 操作类型
  └─────────────────┘
```

### 加法运算 `__add__`

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)  # 类型标准化
    out = Value(self.data + other.data, (self, other), '+')       # 构建输出节点
```

**计算图构建：**
```
    Value(a) ──┐
               ├──➕──> Value(out)
    Value(b) ──┘
```

**反向传播梯度计算：**
```python
def _backward():
    self.grad += out.grad      # ∂L/∂a = ∂L/∂out × ∂out/∂a = 1 × out.grad
    other.grad += out.grad     # ∂L/∂b = ∂L/∂out × ∂out/∂b = 1 × out.grad
```
加法的梯度是 **1**，所以上游梯度直接传递给所有前驱节点。

### 乘法运算 `__mul__`

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
```

**计算图构建：**
```
    Value(a) ──┐
               ├──✕──> Value(out)
    Value(b) ──┘
```

**反向传播梯度计算：**
```python
def _backward():
    self.grad += other.data * out.grad   # ∂L/∂a = b × out.grad
    other.grad += self.data * out.grad   # ∂L/∂b = a × out.grad
```

乘法的梯度：**∂(a×b)/∂a = b**

### ReLU 激活函数

```python
def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
```

**梯度规则：**
```python
def _backward():
    self.grad += (out.data > 0) * out.grad  # 正区间梯度为1，负区间梯度为0
```

**几何解释：**
```
ReLU(x)
    │
  1 │────────
    │         ╲
  0 │──────────╲─────────
    └───────────┼────────→
              0 │    x
```

### 反向传播 `backward()`

```python
def backward(self):
    # 1. 拓扑排序：保证从叶到根的顺序
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # 2. 链式法则：从输出向输入传播梯度
    self.grad = 1  # 损失函数对自身的梯度为1
    for v in reversed(topo):
        v._backward()
```

**执行流程图：**
```
原始计算: z = a * b + ReLU(c)
           │
           ▼
计算图拓扑:
    c ──→ ReLU ──┐
                 ├──➕──> z
    a ──✕── b ───┘
           │
           ▼
反向传播顺序: [z, ReLU, b, a, c]
```

**链式法则实例：**
- 假设 `z = a * b`，求 `∂z/∂a`
- 链式法则：`∂z/∂a = ∂z/∂out × ∂out/∂a = 1 × b = b`

---

## 🧠 nn.py — 神经网络模块

### Module 基类

```python
class Module:
    """ 神经网络模块的抽象基类 """
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
```

所有网络层都继承自 Module，提供统一接口：
- `parameters()`: 返回所有可学习参数
- `zero_grad()`: 训练时清除梯度

### Neuron（神经元）

```python
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # 权重向量
        self.b = Value(0)                                            # 偏置
        self.nonlin = nonlin                                        # 是否应用激活
```

**参数维度：**
```
输入: x = [x₁, x₂, ..., xₙᵢₙ]  (nin维向量)
权重: w = [w₁, w₂, ..., wₙᵢₙ]  (nin维向量)
偏置: b (标量)

输出计算:
  act = Σ(wᵢ × xᵢ) + b = w · x + b
  out = ReLU(act) if nonlin else act
```

**维度图示：**
```
       w₁
       w₂
  x ── ✕ ── + ── ReLU ──→ out
       w₃
       ...
       wₙᵢₙ

向量形式: out = ReLU(w·x + b)
```

### Layer（层）

```python
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
```

**网络结构：**
```
输入 x (nin维)
    │
    ▼
┌─────────────────────────────────────┐
│  Layer (nin → nout)                 │
│                                     │
│  Neuron₁: x → out₁                 │
│  Neuron₂: x → out₂                 │
│  ...                                │
│  Neuronₙᵒᵤₜ: x → outₙᵒᵤₜ           │
└─────────────────────────────────────┘
    │
    ▼
输出 [out₁, out₂, ..., outₙᵒᵤₜ] (nout维向量)
```

### MLP（多层感知机）

```python
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) 
                       for i in range(len(nouts))]
```

**MLP 架构：**
```
MLP(nin=2, nouts=[8, 4, 1])

输入 (2维)
    │
    ▼
Layer₁: 2 → 8 + ReLU
    │
    ▼
Layer₂: 8 → 4 + ReLU
    │
    ▼
Layer₃: 4 → 1 (无激活，用于回归/分类)
    │
    ▼
输出 (1维标量)
```

---

## 📊 训练流程示例

### 1. 构建网络

```python
from micrograd.nn import MLP

model = MLP(2, [8, 4, 1])  # 2输入 → 8神经元 → 4神经元 → 1输出
print(model)
# MLP of [Layer of [ReLUNeuron(2), ...×8], Layer of [ReLUNeuron(8), ...×4], Layer of [LinearNeuron(4)]]
```

### 2. 前向传播

```python
x = [1.0, 2.0]           # 输入数据
y_pred = model(x)         # 前向传播
# 返回 Value 对象，data 属性包含预测值
```

### 3. 损失计算

```python
y_true = Value(1.0)                    # 真实标签
loss = (y_pred - y_true) ** 2         # MSE 损失
# 等价于: loss = (y_pred - y_true) * (y_pred - y_true)
```

### 4. 反向传播

```python
loss.backward()  # 计算所有参数的梯度
```

**梯度传播过程：**
```
loss = (y_pred - y_true)²
    │
    ▼
∂loss/∂y_pred = 2(y_pred - y_true)
    │
    ▼
... (通过整个网络反向传播)
    │
    ▼
更新权重: w = w - learning_rate * w.grad
```

### 5. 参数更新

```python
learning_rate = 0.1
for p in model.parameters():
    p.data -= learning_rate * p.grad
model.zero_grad()  # 清除梯度，准备下一轮
```

---

## 🔄 计算图可视化

以 `loss = (2x₁ + 3x₂ - 1)²` 为例：

**前向传播计算图：**
```
x₁ ──✕─→ 2x₁ ──┐
                 ├──➕──> act ──ReLU──> out
x₂ ──✕─→ 3x₂ ──┘        │
                         │
              constant ←─┘

out = ReLU(2x₁ + 3x₂ - 1)
```

**反向传播计算图：**
```
∂loss/∂out = 2(out - 1)    ← 从损失函数来

ReLU 反向:
  ∂out/∂act = 1 if act > 0 else 0

加法反推:
  ∂act/∂(2x₁) = 1
  ∂act/∂(3x₂) = 1

乘法反推:
  ∂(2x₁)/∂x₁ = 2
  ∂(3x₂)/∂x₂ = 3
```

---

## 📐 维度变换总览

| 操作 | 输入维度 | 输出维度 | 说明 |
|------|---------|---------|------|
| Neuron (nin→1) | `[nin]` | `[]` 标量 | 权重点积 + 偏置 |
| Layer (nin→nout) | `[nin]` | `[nout]` | 多个神经元并行 |
| MLP 完整前向 | `[nin]` | `[nout[-1]]` | 逐层变换 |

---

## 🧩 关键设计思想

### 1. 表达式求导而非数值求导
```python
# 不用数值差分: (f(x+ε) - f(x)) / ε
# 而是用链式法则精确求导
```

### 2. 动态计算图
- 每次前向传播实时构建计算图
- 支持任意 Python 控制流（if, for, while）

### 3. 操作符重载
- `+`, `-`, `*`, `**`, `ReLU` 都返回新的 Value 对象
- 自动维护 `_prev` 和 `_op` 构建计算图

---

> 📚 课程视频: [Backpropagation micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
> 📦 源代码: [karpathy/micrograd](https://github.com/karpathy/micrograd)