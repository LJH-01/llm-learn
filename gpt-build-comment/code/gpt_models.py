"""
GPT 构建 — Transformer 语言模型

来源: https://github.com/karpathy/makemore
视频: https://www.youtube.com/watch?v=kCc8FmEb1nY

============ 核心概念速查 ============

【维度标记】
- B: batch size (批量大小)
- T: sequence length (序列长度)
- C: channels / embedding dimension (通道数/嵌入维度)
- V: vocab size (词汇表大小)
- nh: number of heads (注意力头数)
- hs: head size (每头维度)

【Transformer 维度流程】
  输入: idx (B, T)
    ↓
  嵌入: wte(idx) (B, T, n_embd) + wpe(pos) (T, n_embd)
    ↓
  相加: tok_emb + pos_emb (B, T, n_embd)
    ↓
  Blocks: [LayerNorm → SelfAttn → residual] × n_layer
    ↓
  [LayerNorm → residual]
    ↓
  LM Head: (B, T, n_embd) → (B, T, vocab_size)
"""

import os, sys, time, math, argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ModelConfig:
    block_size: int = None
    vocab_size: int = None
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_embd2: int = 64


class NewGELU(nn.Module):
    """GELU 激活函数 — Google BERT 版本"""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    因果自注意力层

    维度变化:
    ┌─────────────────────────────────────────────────────────────┐
    │  输入 x: (B, T, n_embd)                                  │
    │      ↓                                                     │
    │  QKV 投影: c_attn(x) → (B, T, 3*n_embd)                  │
    │      ↓                                                     │
    │  分割 q, k, v: 各 (B, T, n_embd)                         │
    │      ↓                                                     │
    │  Reshape: → (B, n_head, T, head_size)                   │
    │      ↓                                                     │
    │  因果注意力: softmax(QK^T/√d)V                          │
    │      ↓                                                     │
    │  输出 y: (B, T, n_embd)                                   │
    └─────────────────────────────────────────────────────────────┘
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """Transformer 块: LayerNorm → SelfAttn → residual; LayerNorm → MLP → residual"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer 语言模型 (GPT-2 架构)

    完整架构:
    Token Embeddings + Position Embeddings → Blocks × n_layer → LayerNorm → LM Head
    """
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def get_block_size(self):
        return self.block_size


class Bigram(nn.Module):
    """Bigram 语言模型 — 查表预测下一个字符"""
    def __init__(self, config):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((config.vocab_size, config.vocab_size)))

    def forward(self, idx, targets=None):
        logits = self.logits[idx]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def get_block_size(self):
        return 1


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """自回归采样生成"""
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# 完整代码见 https://github.com/karpathy/makemore/blob/master/makemore.py