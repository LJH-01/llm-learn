"""
神经网络模块 - 基于 micrograd 引擎构建

本模块实现了三层核心组件：
- Module: 基类，定义参数管理接口
- Neuron: 单个神经元
- Layer: 神经元层
- MLP: 多层感知机

维度说明:
- 输入 x: 向量 [x₁, x₂, ..., xₙ]
- 权重 w: 向量 [w₁, w₂, ..., wₙ]
- 输出: 标量或向量
"""

import random
from micrograd.engine import Value


class Module:
    """
    神经网络模块基类

    所有网络层都继承此类，获得统一接口：
    - parameters(): 返回所有可学习参数
    - zero_grad(): 清除所有参数梯度
    """

    def zero_grad(self):
        """清除所有参数的梯度（训练时每步必调）"""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """返回所有可学习参数（权重和偏置）"""
        return []


class Neuron(Module):
    """
    单神经元节点

    结构: y = ReLU(w·x + b) 或 y = w·x + b

    参数维度:
    ┌──────────────────────────────────────────────┐
    │  输入 x = [x₁, x₂, ..., xₙᵢₙ]              │
    │       ↓                                      │
    │  权重 w = [w₁, w₂, ..., wₙᵢₙ]  (nin个)      │
    │       ↓                                      │
    │  点积 w·x = Σ(wᵢ × xᵢ)                      │
    │       ↓                                      │
    │  加偏置 w·x + b                             │
    │       ↓                                      │
    │  激活 ReLU(w·x + b)                         │
    │       ↓                                      │
    │  输出 y (标量)                               │
    └──────────────────────────────────────────────┘

    Example:
        nin=3, nonlin=True
        输入: x = [1.0, 2.0, 3.0]
        权重自动初始化: w ≈ [-1, 1] 之间的随机值
        计算: y = ReLU(w·x + b)
    """

    def __init__(self, nin, nonlin=True):
        """
        初始化神经元

        Args:
            nin: 输入维度（输入特征数量）
            nonlin: 是否应用 ReLU 激活（最后一层通常为 False）
        """
        # 权重向量：每个输入特征一个权重，初始化为 [-1, 1] 均匀分布
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # 偏置标量：初始化为 0
        self.b = Value(0)
        # 是否应用激活函数
        self.nonlin = nonlin

    def __call__(self, x):
        """
        前向传播

        Args:
            x: 输入向量，长度为 nin

        Returns:
            标量输出（应用激活后）

        计算流程:
            act = Σ(wᵢ * xᵢ) + b
            out = ReLU(act) if self.nonlin else act
        """
        # 计算加权求和: act = w·x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # 应用激活函数（除最后一层外）
        return act.relu() if self.nonlin else act

    def parameters(self):
        """返回权重和偏置，用于优化器更新"""
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """
    全连接层（多个神经元并行）

    结构: 对输入向量并行应用多个神经元，每个神经元产生一个输出

    维度变换:
    ┌────────────────────────────────────────────────────┐
    │                                                    │
    │   输入 x = [x₁, x₂, ..., xₙᵢₙ]                    │
    │       │                                            │
    │       ▼                                            │
    │   ┌─────────────────────────────────────┐          │
    │   │  Neuron₁: x → y₁ (w₁·x + b₁)       │          │
    │   │  Neuron₂: x → y₂ (w₂·x + b₂)       │          │
    │   │  ...                               │          │
    │   │  Neuronₙᵒᵤₜ: x → yₙᵒᵤₜ (wₙ·x+bₙ) │          │
    │   └─────────────────────────────────────┘          │
    │       │                                            │
    │       ▼                                            │
    │   输出 y = [y₁, y₂, ..., yₙᵒᵤₜ]                  │
    │                                                    │
    └────────────────────────────────────────────────────┘

    Example:
        Layer(nin=4, nout=3)
        输入: [0.1, 0.5, -0.3, 0.8] (4维)
        输出: [y₁, y₂, y₃] (3维)
    """

    def __init__(self, nin, nout, **kwargs):
        """
        初始化层

        Args:
            nin: 输入维度
            nout: 神经元数量（输出维度）
            **kwargs: 传递给每个 Neuron（如 nonlin=True）
        """
        # 创建 nin → nout 个神经元
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        前向传播

        Args:
            x: 输入向量

        Returns:
            如果 nout=1 返回单个标量，否则返回向量
        """
        # 并行执行所有神经元
        out = [n(x) for n in self.neurons]
        # 单输出时返回标量而非列表
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """返回所有神经元的所有参数"""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    多层感知机（多层全连接网络）

    结构: Input → Layer₁ → Layer₂ → ... → Layerₖ → Output

    维度变换:
    ┌────────────────────────────────────────────────────────┐
    │                                                        │
    │   输入向量 (nin维)                                     │
    │       │                                                │
    │       ▼                                                │
    │   Layer₁: nin → h₁ + ReLU                              │
    │       │                                                │
    │       ▼                                                │
    │   Layer₂: h₁ → h₂ + ReLU                              │
    │       │                                                │
    │       ▼                                                │
    │   ...                                                  │
    │       │                                                │
    │       ▼                                                │
    │   Layerₖ: hₖ₋₁ → nout[-1] (无激活，用于分类/回归)      │
    │       │                                                │
    │       ▼                                                │
    │   输出标量/向量                                        │
    │                                                        │
    └────────────────────────────────────────────────────────┘

    Example:
        MLP(nin=2, nouts=[8, 4, 1])
        输入: [x₁, x₂]
        Layer₁: 2 → 8 + ReLU → [h₁₁, h₁₂, ..., h₁₈]
        Layer₂: 8 → 4 + ReLU → [h₂₁, h₂₂, h₂₃, h₂₄]
        Layer₃: 4 → 1 (Linear) → y
    """

    def __init__(self, nin, nouts):
        """
        初始化 MLP

        Args:
            nin: 输入维度
            nouts: 每层神经元数量的列表，如 [8, 4, 1]
                   最后一层通常无激活（nonlin=False）
        """
        # 构建层尺寸: [nin, nouts[0], nouts[1], ..., nouts[-1]]
        sz = [nin] + nouts
        # 创建各层：除最后一层外都使用 ReLU 激活
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """逐层前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """收集所有层的所有参数"""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# ============ 训练流程示意 ============

"""
MLP 训练循环:

1. 构建模型:
   model = MLP(2, [8, 4, 1])

2. 前向传播:
   y_pred = model(x)  # x: [2维输入]

3. 计算损失:
   loss = (y_pred - y_true) ** 2

4. 反向传播:
   loss.backward()

5. 更新参数:
   for p in model.parameters():
       p.data -= learning_rate * p.grad
   model.zero_grad()

维度检查:
   x = [2.0, 1.0]            # 2维输入
   y_pred = model(x)         # 输出: Value(data=?, grad=0)
   y_true = Value(1.0)       # 目标标量
   loss = (y_pred - y_true) ** 2  # MSE 损失
   loss.backward()           # 计算所有梯度
"""