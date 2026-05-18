# Llama2.C — 单文件纯 C 推理 Llama 2

> 本项目是单文件纯 C 实现的 Llama 2 推理引擎。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **源代码** | [karpathy/llama2.c](https://github.com/karpathy/llama2.c) |
| **核心主题** | 纯 C 推理、Llama 2 架构、单文件实现 |

---

## 🎯 学习目标

1. 理解 Llama 2 架构
2. 掌握纯 C 推理实现
3. 学习模型量化
4. 理解推理优化

---

## 🏗️ Llama 2 架构

### 与 GPT-2 的区别

| 组件 | GPT-2 | Llama 2 |
|------|-------|---------|
| 位置编码 | 固定 | Rotary (RoPE) |
| FFN | GELU | SwiGLU |
| 归一化 | Pre-LN | RMSNorm |
| 注意力 | ChatGPT | Grouped Multi-Query |

---

## 🔧 纯 C 实现

### 单文件结构

```c
// run.c - 完整的推理代码
int main(int argc, char* argv[]) {
    // 1. 加载模型
    Transformer transformer = load_model("model.bin");

    // 2. 采样循环
    int token = 1;  // <START>
    for (int i = 0; i < max_new_tokens; i++) {
        // 前向传播
        float* logits = forward(&transformer, token);

        // 采样
        token = sample(logits, temperature);

        // 输出
        printf("%s", tokenizer.decode(token));
    }
}
```

### 关键函数

```c
// 矩阵向量乘法
void matvec(float* out, float* in, float* weight, int rows, int cols);

// 层归一化
void rmsnorm(float* out, float* in, float* weight, int size);

// 旋转位置编码
void rotary(float* q, float* k, int head_dim, int seq_len);
```

---

## 📊 模型文件格式

```
+------------------+
| header (元数据)   |  vocab_size, dim, n_layers, etc.
+------------------+
| embedding table  |  float[vocab_size][dim]
+------------------+
| layers            |  repeated n_layers times:
|  - attention      |    q, k, v, o projections
|  - feedforward   |    gate, up, down projections
|  - rmsnorm       |    weights for each layer
+------------------+
| final rmsnorm     |
+------------------+
```

---

## 🚀 运行示例

```bash
# 编译
gcc -O3 -o run run.c

# 下载模型（需要单独下载 tokenizer.model 和 model.bin）
# 运行
./run prompt.txt
```

---

## 📈 性能

| 平台 | 速度 |
|------|------|
| CPU (单线程) | ~50 tokens/s |
| CPU (多线程) | ~200 tokens/s |
| GPU (CUDA) | ~1000 tokens/s |

---

> 📦 代码: [karpathy/llama2.c](https://github.com/karpathy/llama2.c)