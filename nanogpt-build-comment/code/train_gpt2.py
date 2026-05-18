"""
GPT-2 训练代码 — nanogpt-build-comment 中文注释版

本文件是 Andrej Karpathy 从零构建 GPT-2 的训练代码。
包含完整的 Transformer 架构实现、训练循环和采样生成。

来源: https://github.com/karpathy/build-nanogpt
"""

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# 第一部分：CausalSelfAttention — 因果自注意力
# ============================================================================

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层 (Causal Self-Attention)

    核心机制：Multi-Head Self-Attention + 因果遮蔽 (Causal Mask)

    原理：
    - 对序列中每个位置，计算其与所有前面位置的注意力分数
    - 未来位置被遮蔽，确保自回归性质

    维度变化：
    ┌─────────────────────────────────────────────────────────────┐
    │  输入 x: (B, T, C)                                         │
    │      ↓                                                     │
    │  QKV 投影: c_attn(x) → (B, T, 3C)                          │
    │      ↓                                                     │
    │  分割 q, k, v: 各 (B, T, C)                                │
    │      ↓                                                     │
    │  Reshape: → (B, n_head, T, head_size)                      │
    │      ↓                                                     │
    │  注意力: scaled_dot_product_attention (flash attention)     │
    │      ↓                                                     │
    │  输出 y: (B, n_head, T, head_size) → (B, T, C)             │
    └─────────────────────────────────────────────────────────────┘

    Example (GPT-2 124M):
    - n_head = 12
    - head_size (hs) = C / n_head = 768 / 12 = 64
    - block_size = 1024
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # QKV 投影：将输入 x 映射到 query, key, value 空间
        # 输入: (B, T, n_embd) → 输出: (B, T, 3*n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 输出投影
        # 输入: (B, T, n_embd) → 输出: (B, T, n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # 用于残差初始化的缩放因子

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, T, C) — batch size, sequence length, embedding dimension

        Returns:
            y: (B, T, C) — 注意力输出
        """
        B, T, C = x.size()

        # 1. QKV 投影
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # 各 (B, T, C)

        # 2. Reshape 为多头格式
        # (B, T, C) → (B, n_head, T, head_size)
        # 其中 head_size = C / n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # 3. 使用 Flash Attention 计算注意力（更快更省内存）
        # is_causal=True 自动应用因果遮蔽
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 4. 重新组装所有头的输出
        # (B, n_head, T, head_size) → (B, T, n_head * head_size) = (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. 输出投影
        y = self.c_proj(y)

        return y


# ============================================================================
# 第二部分：MLP — 前馈神经网络
# ============================================================================

class MLP(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)

    结构：线性层 → GELU 激活 → 线性层

    维度变化：
    ┌─────────────────────────────────────────────────────────────┐
    │  输入 x: (B, T, n_embd)                                    │
    │      ↓                                                     │
    │  c_fc: n_embd → 4*n_embd                                  │
    │      ↓                                                     │
    │  GELU: 激活函数                                            │
    │      ↓                                                     │
    │  c_proj: 4*n_embd → n_embd                                │
    │      ↓                                                     │
    │  输出 y: (B, T, n_embd)                                    │
    └─────────────────────────────────────────────────────────────┘

    注意：4*n_embd 是 GPT-2 的标准配置（约 4 倍扩展）
    """

    def __init__(self, config):
        super().__init__()

        # 扩展层：n_embd → 4*n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # GELU 激活函数（近似 Tanh 版本，更稳定）
        self.gelu = nn.GELU(approximate='tanh')

        # 压缩层：4*n_embd → n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """前向传播"""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# ============================================================================
# 第三部分：Block — Transformer 块
# ============================================================================

class Block(nn.Module):
    """
    Transformer 块

    结构：LayerNorm → Self-Attention → 残差连接
        → LayerNorm → MLP → 残差连接

    维度：输入输出保持不变 (B, T, n_embd)

    这是 GPT 的核心构建块，多个 Block 堆叠形成完整的 Transformer
    """

    def __init__(self, config):
        super().__init__()

        # Pre-LayerNorm（现代 GPT 常用）
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # 自注意力层
        self.attn = CausalSelfAttention(config)

        # 第二个 LayerNorm
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # 前馈网络
        self.mlp = MLP(config)

    def forward(self, x):
        """
        前向传播：带残差连接

        残差连接的作用：
        1. 缓解梯度消失问题
        2. 稳定训练
        3. 允许深层网络有效学习
        """
        # 自注意力残差块
        x = x + self.attn(self.ln_1(x))

        # MLP 残差块
        x = x + self.mlp(self.ln_2(x))

        return x


# ============================================================================
# 第四部分：GPTConfig — GPT 配置
# ============================================================================

@dataclass
class GPTConfig:
    """
    GPT 模型配置

    GPT-2 参数对比：
    ┌────────────────────────────────────────────────────────────┐
    │  模型      │ 层数 │ 头数 │ 嵌入维 │ 参数量    │ 上下文   │
    ├───────────┼──────┼──────┼────────┼───────────┼──────────┤
    │ GPT-2 小  │ 12   │ 12   │ 768    │ 124M     │ 1024    │
    │ GPT-2 中  │ 24   │ 16   │ 1024   │ 350M     │ 1024    │
    │ GPT-2 大  │ 36   │ 20   │ 1280   │ 774M     │ 1024    │
    │ GPT-2 XL  │ 48   │ 25   │ 1600   │ 1558M    │ 1024    │
    └───────────┴──────┴──────┴────────┴───────────┴──────────┘
    """
    block_size: int = 1024      # 最大序列长度（位置编码大小）
    vocab_size: int = 50257      # BPE 词汇表：50,000 merges + 256 bytes + 1 <|endoftext|>
    n_layer: int = 12           # Transformer 层数
    n_head: int = 12            # 注意力头数
    n_embd: int = 768           # 嵌入维度


# ============================================================================
# 第五部分：GPT — 主模型
# ============================================================================

class GPT(nn.Module):
    """
    GPT 主模型

    完整架构：
    ┌─────────────────────────────────────────────────────────────┐
    │  Token Embeddings: (B, T) → (B, T, n_embd)                 │
    │      ↓                                                      │
    │  Position Embeddings: (T) → (T, n_embd)                    │
    │      ↓                                                      │
    │  相加: tok_emb + pos_emb                                    │
    │      ↓                                                      │
    │  Transformer Blocks: n_layer × Block                       │
    │      ↓                                                      │
    │  Final LayerNorm                                            │
    │      ↓                                                      │
    │  LM Head: (B, T, n_embd) → (B, T, vocab_size)               │
    └─────────────────────────────────────────────────────────────┘

    参数共享：token embedding 和 lm_head 共享权重
    - 这样可以减少参数量
    - 学习到的词嵌入可以直接用于预测
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer 主干网络
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),      # Token 嵌入
            wpe=nn.Embedding(config.block_size, config.n_embd),     # 位置嵌入
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer 层
            ln_f=nn.LayerNorm(config.n_embd),                        # 最终 LayerNorm
        ))

        # 语言模型头：不使用偏置（GPT-2 的设计选择）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：token embedding 和 lm_head 共享权重
        self.transformer.wte.weight = self.lm_head.weight

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """参数初始化"""
        if isinstance(module, nn.Linear):
            std = 0.02
            # 对于残差块，使用缩放初始化
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播

        Args:
            idx: (B, T) — token 索引序列
            targets: (B, T) — 目标序列（用于计算损失）

        Returns:
            logits: (B, T, vocab_size) — 每个位置的预测 logits
            loss: 交叉熵损失（如果提供了 targets）
        """
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # 位置索引
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)

        # 嵌入
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        # 相加
        x = tok_emb + pos_emb

        # 通过 Transformer 层
        for block in self.transformer.h:
            x = block(x)

        # 最终 LayerNorm
        x = self.transformer.ln_f(x)

        # 语言模型头
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """从 HuggingFace 加载预训练 GPT-2 权重"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()

        # 加载 HuggingFace 权重
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制权重（处理 Conv1D → Linear 的转置问题）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_hf.keys():
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """配置优化器"""
        # 分离需要 weight decay 和不需要的参数
        decay_params = [p for n, p in self.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # 使用 AdamW 优化器
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=use_fused)
        return optimizer


# ============================================================================
# 第六部分：采样函数
# ============================================================================

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    自回归采样生成

    Args:
        idx: (B, T) 条件序列
        max_new_tokens: 生成的新 token 数量
        temperature: 温度参数（越高越随机）
        top_k: 只考虑前 k 个最高概率的 token
    """
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)

    return idx


# ============================================================================
# 第七部分：训练循环示例
# ============================================================================

"""
训练循环示例:

# 创建模型
model = GPT(GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768
))

# 优化器
optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=1e-3,
    device_type='cuda'
)

# 训练循环
for step in range(num_steps):
    # 前向
    logits, loss = model(idx, targets)

    # 反向
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 更新
    optimizer.step()

    # 日志
    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.4f}")
"""