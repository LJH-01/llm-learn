# micrograd — 反向传播与自动微分

> micrograd 是 Andrej Karpathy 实现的一个精简自动微分引擎，是理解 PyTorch 反向传播机制的绝佳入门教材。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| **源代码** | [karpathy/micrograd](https://github.com/karpathy/micrograd) |
| **核心主题** | 自动求导、反向传播、神经网络 |

---

## 🎯 学习目标

1. 理解反向传播的数学原理
2. 掌握计算图的构建方法
3. 实现自动微分引擎
4. 理解链式法则在神经网络中的应用

---

## 📁 文件结构

```
micrograd/
├── __init__.py         # 包导出
├── engine.py          # 核心：Value 类，自动求导
├── nn.py              # 神经网络层：Neuron, Layer, MLP
└── test/
    └── test_engine.py # 单元测试
```

---

## 🧠 核心概念

### Value 类 — 自动求导的基础

```python
class Value:
    """ 存储单个标量值及其梯度 """
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data        # 标量数值
        self.grad = 0          # 梯度值
        self._backward = lambda: None  # 反向传播函数
        self._prev = set(_children)    # 前驱节点
        self._op = _op                  # 操作类型
```

### 运算符与梯度

| 运算 | 梯度公式 |
|------|---------|
| `a + b` | ∂L/∂a = out.grad |
| `a * b` | ∂L/∂a = b.data × out.grad |
| `a ** n` | ∂L/∂a = n × a^(n-1) × out.grad |
| `ReLU(a)` | ∂L/∂a = (a > 0) × out.grad |

---

## 🔄 反向传播算法

### 两步流程

1. **拓扑排序**：构建从叶到根的执行顺序
2. **链式法则**：逆序应用梯度计算

```python
def backward(self):
    topo = []
    visited = set()
    
    # 1. 拓扑排序
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    
    # 2. 链式法则
    self.grad = 1
    for v in reversed(topo):
        v._backward()
```

---

## 🏗️ 神经网络模块

### Module 基类

所有网络层继承自 Module，提供统一接口：

```python
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
```

### 层级结构

```
Module
├── Neuron(nin)     # 单神经元: y = ReLU(w·x + b)
├── Layer(nin, nout) # 全连接层: 多个 Neuron 并行
└── MLP(nin, nouts)  # 多层感知机: 多个 Layer 堆叠
```

---

## 📊 训练流程

```python
# 1. 构建模型
model = MLP(2, [8, 4, 1])

# 2. 前向传播
y_pred = model([1.0, 2.0])

# 3. 计算损失
loss = (y_pred - y_true) ** 2

# 4. 反向传播
loss.backward()

# 5. 更新参数
for p in model.parameters():
    p.data -= lr * p.grad

# 6. 清除梯度
model.zero_grad()
```

---

## 🧪 测试验证

```bash
cd micrograd
python test/test_engine.py
```

测试用例：
- `test_sanity_check()`: 基础前向/反向传播
- `test_more_ops()`: 扩展操作符测试

---

## 📐 维度变换总览

| 模块 | 输入维度 | 输出维度 | 参数数量 |
|------|---------|---------|---------|
| Neuron (nin→1) | `[nin]` | `[]` 标量 | nin + 1 |
| Layer (nin→nout) | `[nin]` | `[nout]` | nin×nout + nout |
| MLP 完整前向 | `[nin]` | `[nout[-1]]` | Σ(ninᵢ×ninᵢ₊₁ + ninᵢ₊₁) |

---

## 🔑 关键设计思想

1. **表达式求导而非数值求导**：使用链式法则精确计算梯度
2. **动态计算图**：每次前向传播实时构建计算图
3. **运算符重载**：`+`, `-`, `*`, `**`, `ReLU` 自动维护计算图

---

> 📚 视频: [Backpropagation micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
> 📦 代码: [karpathy/micrograd](https://github.com/karpathy/micrograd)