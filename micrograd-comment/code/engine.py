"""
micrograd 引擎 - 自动微分核心实现

本模块实现了 micrograd 的核心类 Value，用于构建计算图并自动计算梯度。
这是理解 PyTorch autograd 机制的绝佳入门。
"""

class Value:
    """
    微分量类 - 存储标量值并跟踪其梯度

    Attributes:
        data: 存储的标量值
        grad: 梯度值（loss 对该值的偏导数）
        _backward: 反向传播函数
        _prev: 前驱节点集合（构建计算图）
        _op: 操作类型（用于调试）
    """

    def __init__(self, data, _children=(), _op=''):
        """
        初始化 Value 对象

        Args:
            data: 标量数值
            _children: 前驱节点元组，用于构建计算图
            _op: 操作类型字符串
        """
        self.data = data                          # 标量值 (e.g., 2.5)
        self.grad = 0                             # 梯度初始化为 0
        self._backward = lambda: None             # 反向传播函数（默认空）
        self._prev = set(_children)               # 前驱节点集合
        self._op = _op                             # 操作类型：'+', '*', 'ReLU' 等

    # ------------------- 二元运算符 -------------------

    def __add__(self, other):
        """
        加法: self + other

        计算图:
            a ──┐
                ├──➕──> (a+b)
            b ──┘

        梯度公式:
            ∂L/∂a = ∂L/∂out × 1
            ∂L/∂b = ∂L/∂out × 1
        """
        other = other if isinstance(other, Value) else Value(other)  # 标量转 Value
        out = Value(self.data + other.data, (self, other), '+')      # 构建输出节点

        def _backward():
            # 加法梯度：上游梯度直接传递给所有前驱
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        乘法: self * other

        计算图:
            a ──┐
                ├──✕──> (a×b)
            b ──┘

        梯度公式（链式法则）:
            ∂L/∂a = ∂L/∂out × b
            ∂L/∂b = ∂L/∂out × a

        Example:
            a=2, b=3, out=6
            若 out.grad = 1（即 ∂L/∂out = 1）
            则 ∂L/∂a = 3, ∂L/∂b = 2
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 乘法梯度：另一因子 × 上游梯度
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        幂运算: self ** other

        梯度公式:
            ∂L/∂x = n × x^(n-1) × ∂L/∂out

        仅支持 int/float 指数
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # 幂函数梯度: n × x^(n-1)
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """
        ReLU 激活函数: max(0, x)

        几何形状:
            │
          1 │────────
            │         ╲
          0 │──────────╲────────
            └───────────┼────────→
                      0   x

        梯度公式:
            ∂L/∂x = 1 if x > 0 else 0
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # ReLU 梯度：正区间为1，负区间为0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    # ------------------- 反向传播 -------------------

    def backward(self):
        """
        执行反向传播，计算所有前驱节点的梯度

        算法步骤:
        1. 拓扑排序：从输出到输入排序所有节点
        2. 初始化：设置 out.grad = 1
        3. 链式传播：逆序执行每个节点的 _backward()

        计算图示例:
            z = a * b + ReLU(c)

            拓扑序: [c, b, a, ReLU, *, +.z]
            反向序: [z, +, *, ReLU, a, b, c]
        """
        # 1. 拓扑排序：确保按从叶到根的顺序处理
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2. 逆序遍历，应用链式法则
        self.grad = 1  # 根节点梯度初始化为1
        for v in reversed(topo):
            v._backward()

    # ------------------- 一元运算符 -------------------

    def __neg__(self): return self * -1                # -self

    def __radd__(self, other): return self + other     # other + self
    def __rmul__(self, other): return self * other     # other * self

    def __sub__(self, other): return self + (-other)    # self - other
    def __rsub__(self, other): return other + (-self)   # other - self

    def __truediv__(self, other): return self * other**-1  # self / other
    def __rtruediv__(self, other): return other * self**-1 # other / self

    # ------------------- 调试输出 -------------------

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# 运算维度对照表
"""
┌─────────────────────────────────────────────────────────────┐
│                    运算维度速查表                            │
├─────────────┬────────────┬────────────┬─────────────────────┤
│   运算      │  输入维度   │  输出维度  │       梯度公式       │
├─────────────┼────────────┼────────────┼─────────────────────┤
│  a + b      │  [], []    │  []        │  da = out.grad      │
│  a * b      │  [], []    │  []        │  da = b * out.grad  │
│  a ** n     │  []        │  []        │  da = n*a^(n-1)*grad │
│  ReLU(a)    │  []        │  []        │  da = (a>0)*out.grad │
└─────────────┴────────────┴────────────┴─────────────────────┘
[] 表示标量（0维张量）
"""