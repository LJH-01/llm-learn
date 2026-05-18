# NanoGPT — 从零训练 GPT-2

> 本课程讲解从零训练 GPT-2 模型，使用 FineWeb 数据集。

---

## 📚 课程概览

| 项目 | 内容 |
|------|------|
| **视频** | [YouTube](https://www.youtube.com/watch?v=l8pRSuU81PU) |
| **源代码** | [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) |
| **核心主题** | GPT-2 训练、FineWeb 数据集、训练监控 |

---

## 🎯 学习目标

1. 理解 GPT-2 的完整训练流程
2. 掌握数据集处理方法
3. 学会监控训练过程
4. 评估生成文本质量

---

## 🏗️ GPT-2 配置

### 模型参数

| 参数 | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
|------|-------------|--------------|-------------|----------|
| n_layer | 12 | 24 | 36 | 48 |
| n_head | 12 | 16 | 20 | 25 |
| n_embd | 768 | 1024 | 1280 | 1600 |
| 参数量 | 124M | 350M | 774M | 1558M |

---

## 📊 训练配置

### 关键超参数

```python
config = GPTConfig(
    vocab_size=50257,      # BPE 词汇表
    block_size=1024,       # 最大上下文长度
    n_layer=12,            # Transformer 层数
    n_head=12,             # 注意力头数
    n_embd=768,            # 嵌入维度
)
```

### 优化器配置

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,               # 学习率
    weight_decay=0.1,      # 权重衰减
    betas=(0.9, 0.99),
    eps=1e-8,
)
```

---

## 🔄 训练循环

### 关键步骤

```python
for step in range(num_steps):
    # 1. 获取批次
    X, Y = get_batch()

    # 2. 前向传播
    logits, loss = model(X, Y)

    # 3. 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 4. 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 5. 更新参数
    optimizer.step()
```

---

## 📈 训练监控

### 使用 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/')

# 记录损失
writer.add_scalar("Loss/train", loss.item(), step)

# 记录学习率
writer.add_scalar("LR", optimizer.param_groups[0]['lr'], step)

# 记录梯度范数
grad_norm = torch.nn.utils.get_grad_norm(model.parameters())
writer.add_scalar("Grad/norm", grad_norm, step)
```

### 监控指标

| 指标 | 说明 |
|------|------|
| loss | 训练损失 |
| eval loss | 验证损失 |
| grad norm | 梯度范数（应 < 1.0） |
| lr | 学习率 |

---

## 🧪 数据集处理

### FineWeb 数据集

```python
# 加载文本
text = open('fineweb.txt').read()

# 分词
tokens = tokenizer.encode(text)

# 创建批次
class Dataset:
    def __getitem__(self, idx):
        x = tokens[idx:idx+block_size]
        y = tokens[idx+1:idx+block_size+1]
        return torch.tensor(x), torch.tensor(y)
```

---

## 🎯 生成评估

### 困惑度 (Perplexity)

```python
def evaluate(model, dataset):
    model.eval()
    total_loss = 0
    count = 0

    for batch in dataset:
        x, y = batch
        logits, loss = model(x, y)
        total_loss += loss.item()
        count += 1

    return math.exp(total_loss / count)
```

---

## 🚀 运行示例

```bash
# 训练 GPT-2
python train_gpt2.py \
    --input-file fineweb.txt \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --max-steps 10000

# 采样生成
python train_gpt2.py --sample-only --checkpoint out/model.pt
```

---

> 📚 视频: [复现 GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
> 📦 代码: [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)