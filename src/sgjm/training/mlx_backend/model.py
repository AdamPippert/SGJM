from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from sgjm.training.config import ModelConfig, is_attn_layer
from sgjm.training.mlx_backend.mamba2 import Mamba2Block

# Re-export under both names for backward compatibility and test imports
_is_attn_layer = is_attn_layer


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must divide n_heads")
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = [
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff)
            if is_attn_layer(i, cfg.attn_every_n)
            else Mamba2Block(
                cfg.d_model,
                state_size=cfg.mamba_state_size,
                expand=cfg.mamba_expand,
                d_conv=cfg.mamba_d_conv,
                head_dim=cfg.mamba_head_dim,
                chunk_size=cfg.mamba_chunk_size,
            )
            for i in range(cfg.n_layers)
        ]
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def __call__(self, idx: mx.array) -> tuple[mx.array, mx.array]:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = mx.arange(T)[None, :]
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, self.lm_head(x)


class Drafter(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.drafter_d_model
        self.in_proj = nn.Linear(cfg.d_model, d, bias=False)
        self.queries = mx.random.normal((cfg.block_size, d)) * 0.02
        self.blocks = [Block(d, cfg.drafter_heads, cfg.drafter_d_ff) for _ in range(cfg.drafter_layers)]
        self.norm = nn.RMSNorm(d)
        self.out = nn.Linear(d, cfg.vocab_size, bias=False)
        self.latent_out = nn.Linear(d, cfg.d_model, bias=False)
        self.block_size = cfg.block_size

    def __call__(self, parent_hidden: mx.array) -> tuple[mx.array, mx.array]:
        B, T, _ = parent_hidden.shape
        h = self.in_proj(parent_hidden)
        x = h[:, :, None, :] + self.queries[None, None, :, :]
        x = x.reshape(B * T, self.block_size, -1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.out(x).reshape(B, T, self.block_size, -1)
        latents = self.latent_out(x).reshape(B, T, self.block_size, -1)
        return logits, latents


class JepaJudge(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.judge_hidden
        self.fc1 = nn.Linear(cfg.d_model, h)
        self.fc2 = nn.Linear(h, cfg.d_model)

    def __call__(self, parent_hidden: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(parent_hidden)))


class Verifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.verifier_hidden
        self.fc1 = nn.Linear(cfg.d_model * 2, h)
        self.fc2 = nn.Linear(h, 1)

    def __call__(self, parent_hidden: mx.array, child_hidden: mx.array) -> mx.array:
        x = mx.concatenate([parent_hidden, child_hidden], axis=-1)
        return self.fc2(nn.gelu(self.fc1(x))).squeeze(-1)


class SGJM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.drafter = Drafter(cfg)
        self.judge = JepaJudge(cfg)
        self.verifier = Verifier(cfg)

    def __call__(self, idx: mx.array) -> tuple[mx.array, mx.array]:
        return self.backbone(idx)

    def num_parameters(self) -> int:
        from mlx.utils import tree_flatten
        return sum(int(v.size) for _, v in tree_flatten(self.parameters()))
