"""
Minbpe — 最简 BPE 算法实现

来源: https://github.com/karpathy/minbpe

============ BPE 算法 ============

1. 初始化: 将所有字符作为词汇表
2. 统计: 计算所有字节对的频率
3. 合并: 找到最频繁的对，添加新词到词汇表
4. 重复: 直到达到目标词汇表大小

============ 核心类 ============

Base — 基础类，实现 encode/decode 接口
BasicTokenizer — 最简单的 BPE 实现
RegexTokenizer — 带正则化预处理的 BPE

============ 使用示例 ============

from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train(text, vocab_size=1000)

tokens = tokenizer.encode("Hello")
text = tokenizer.decode(tokens)
"""

import re
from typing import List, Dict, Tuple


class Base:
    """基础 Tokenizer 类"""

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError

    def save(self, path: str):
        """保存 tokenizer 到文件"""
        raise NotImplementedError

    def load(self, path: str):
        """从文件加载 tokenizer"""
        raise NotImplementedError


class BasicTokenizer(Base):
    """
    最简单的 BPE Tokenizer

    训练复杂度: O(n * k)
    其中 n = 训练文本长度，k = 目标词汇表大小
    """

    def __init__(self):
        self.merges = {}  # {(byte1, byte2): new_byte}
        self.vocab_size = 256

    def train(self, text: str, vocab_size: int = 1000, min_freq: int = 1):
        """
        训练 BPE

        Args:
            text: 训练文本
            vocab_size: 目标词汇表大小
            min_freq: 最小频率阈值
        """
        # 初始化字节词汇表
        tokens = list(text.encode('utf-8'))
        self.vocab = {i: bytes([i]) for i in range(256)}

        # 迭代合并
        while self.vocab_size < vocab_size:
            # 统计相邻对的频率
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            # 找到最频繁的对
            best_pair = None
            best_freq = 0
            for pair, freq in pairs.items():
                if freq > best_freq:
                    best_pair = pair
                    best_freq = freq

            if best_freq < min_freq:
                break

            # 合并
            new_token = self.vocab_size
            self.merges[best_pair] = new_token

            # 更新词汇表
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # 应用合并
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """编码"""
        tokens = list(text.encode('utf-8'))

        # 贪婪合并
        while True:
            pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    pairs.append((i, self.merges[pair]))

            if not pairs:
                break

            # 找到第一个可合并的位置
            i, new_token = pairs[0]
            tokens = tokens[:i] + [new_token] + tokens[i + 2:]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """解码"""
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(self.vocab[token])
            else:
                result.append(bytes([token]))  # 未知 token

        return b''.join(result).decode('utf-8', errors='replace')

    def save(self, path: str):
        """保存"""
        import json
        data = {
            'merges': {f'{k[0]},{k[1]}': v for k, v in self.merges.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """加载"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        self.vocab_size = data['vocab_size']


class RegexTokenizer(Base):
    """
    带正则化预处理的 BPE Tokenizer

    预处理规则:
    - 分离标点符号
    - 分割数字
    - 处理空白
    """

    def __init__(self):
        self.basic = BasicTokenizer()
        # 正则表达式模式
        self.pattern = re.compile(r"""(?:[^\s\w]|_)+|(?:\d+(?:\.\d*)?)|(?:\d*(?:\.\d+)?)""")

    def train(self, text: str, vocab_size: int = 1000, min_freq: int = 1):
        # 预处理
        pieces = self.pattern.findall(text)
        text = ' '.join(pieces)
        self.basic.train(text, vocab_size, min_freq)

    def encode(self, text: str) -> List[int]:
        pieces = self.pattern.findall(text)
        text = ' '.join(pieces)
        return self.basic.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.basic.decode(tokens).replace(' ', '')

    def save(self, path: str):
        self.basic.save(path)

    def load(self, path: str):
        self.basic.load(path)


if __name__ == '__main__':
    # 测试
    text = "Hello, world! 123"

    tokenizer = RegexTokenizer()
    tokenizer.train("Hello world " * 100, vocab_size=500)

    tokens = tokenizer.encode("Hello, world! 123")
    print(f"Encoded: {tokens}")

    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")