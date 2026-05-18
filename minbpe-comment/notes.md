# Minbpe — 最简 BPE 算法实现

> 本项目是最简 BPE (Byte Pair Encoding) 算法的实现。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **源代码** | [karpathy/minbpe](https://github.com/karpathy/minbpe) |
| **核心主题** | BPE 算法、Tokenizer 实现 |

---

## 🎯 学习目标

1. 理解 BPE 算法的实现细节
2. 掌握 Tokenizer 的构建过程
3. 学习正则化分词
4. 理解训练和推理

---

## 🧠 BPE 算法

### 基本步骤

```
1. 初始化：将所有字符作为词汇表
2. 统计：计算所有字节对的频率
3. 合并：找到最频繁的对，添加新词到词汇表
4. 重复：直到达到目标词汇表大小
```

### 训练复杂度

```
时间: O(n * k)
其中:
  n = 训练文本长度
  k = 目标词汇表大小

优化: 使用优先队列（堆）加速
```

---

## 🔧 代码结构

### 核心类

| 类 | 说明 |
|---|------|
| Base | 基础类，实现 encode/decode 接口 |
| BasicTokenizer | 最简单的 BPE 实现 |
| RegexTokenizer | 带正则化预处理的 BPE |
| SentencePiece | 更复杂的分词器 |

---

## 📊 分词流程

### RegexTokenizer 流程

```
输入: "Hello, world!"

1. 正则预处理
   - 分离标点符号
   - 分割数字
   - 处理空白

2. 查找合并
   - 遍历文本
   - 贪婪匹配最长词汇
   - 未知词回退到字节级

3. 输出 token IDs
```

---

## 🔄 与 GPT Tokenizer 的区别

| 特性 | Minbpe | GPT Tokenizer |
|------|--------|---------------|
| 词汇表大小 | 可配置 | 50,257 |
| 正则化 | 可选 | 必需 |
| 训练速度 | 快 | 慢 |
| 用途 | 教学 | 生产 |

---

## 📝 使用示例

```python
from minbpe import RegexTokenizer

# 创建 tokenizer
tokenizer = RegexTokenizer()

# 训练（给定文本）
tokenizer.train(text, vocab_size=1000)

# 保存
tokenizer.save("tokenizer.json")

# 编码
tokens = tokenizer.encode("Hello")

# 解码
text = tokenizer.decode(tokens)
```

---

> 📦 代码: [karpathy/minbpe](https://github.com/karpathy/minbpe)