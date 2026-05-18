# LLM.C — 纯 C/CUDA 实现 LLM 训练

> 本项目是纯 C 和 CUDA 实现的大语言模型训练，无需任何依赖。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **源代码** | [karpathy/llm.c](https://github.com/karpathy/llm.c) |
| **核心主题** | 纯 C 训练、CUDA 加速、零依赖 |

---

## 🎯 学习目标

1. 理解底层矩阵运算
2. 掌握 CUDA 并行编程
3. 理解训练优化的细节
4. 学习极致性能优化

---

## 🧠 为什么用 C？

### 优势

1. **零依赖**：不需要 Python、PyTorch 等
2. **极致性能**：避免 Python 开销
3. **易于部署**：直接编译运行
4. **学习底层**：理解 GPU 工作原理

### 限制

1. 开发速度慢
2. 调试困难
3. 功能有限

---

## 📊 核心文件

| 文件 | 说明 |
|------|------|
| train_gpt2.cu | CUDA 训练代码 |
| test_gpt2.cu | CUDA 测试代码 |
| train_gpt2.c | CPU 训练代码 |
| test_gpt2.c | CPU 测试代码 |

---

## 🔧 CUDA 优化

### 矩阵乘法优化

```c
// 朴素矩阵乘法: O(n³)
// GPU 并行: 利用数千个线程同时计算

__global__ void matmul_kernel(float* C, float* A, float* B, int N) {
    // 每个线程计算一个输出元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

### 内存访问优化

1. **合并访问**：相邻线程访问相邻内存
2. **共享内存**：减少全局内存访问
3. **矩阵分块**：提高缓存命中率

---

## 🏗️ 训练流程

```
1. 数据加载 (从二进制文件)
2. 初始化模型 (权重随机)
3. 训练循环:
   a. 前向传播 (矩阵乘法 + 激活)
   b. 计算损失 (Cross Entropy)
   c. 反向传播 (梯度计算)
   d. 参数更新 (AdamW)
4. 保存模型 (二进制格式)
```

---

## 📊 性能对比

| 实现 | 速度 | 说明 |
|------|------|------|
| PyTorch (GPU) | 1x | 基准 |
| LLM.C (CUDA) | ~0.8x | 接近 PyTorch |
| LLM.C (CPU) | ~0.1x | 比 PyTorch 慢 |

---

> 📦 代码: [karpathy/llm.c](https://github.com/karpathy/llm.c)