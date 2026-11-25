from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    context: int = 64
    layers: int = 2
    heads: int = 4
    width: int = 64

    def __post_init__(self):
        if min(self.context, self.layers, self.heads, self.width) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.attn_norm(x))
        qkv = qkv.reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.ff_norm(x))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(256, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        return F.linear(self.norm(self.blocks(x)), self.tokens.weight)
