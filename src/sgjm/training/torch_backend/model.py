from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgjm.training.config import ModelConfig, is_attn_layer
from sgjm.training.torch_backend.mamba2 import Mamba2Block

# Re-export under both names for backward compatibility and test imports
_is_attn_layer = is_attn_layer


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must divide n_heads")
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
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
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, self.lm_head(x)


class Drafter(nn.Module):
    """Block-future predictor: from each parent latent emit `block_size` token logits."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.drafter_d_model
        self.in_proj = nn.Linear(cfg.d_model, d, bias=False)
        self.queries = nn.Parameter(torch.randn(cfg.block_size, d) * 0.02)
        self.blocks = nn.ModuleList(
            Block(d, cfg.drafter_heads, cfg.drafter_d_ff, cfg.dropout)
            for _ in range(cfg.drafter_layers)
        )
        self.norm = RMSNorm(d)
        self.out = nn.Linear(d, cfg.vocab_size, bias=False)
        self.latent_out = nn.Linear(d, cfg.d_model, bias=False)
        self.block_size = cfg.block_size

    def forward(self, parent_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = parent_hidden.shape
        h = self.in_proj(parent_hidden)
        x = h.unsqueeze(2) + self.queries.view(1, 1, self.block_size, -1)
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
        self.predictor = nn.Sequential(
            nn.Linear(cfg.d_model, h),
            nn.GELU(),
            nn.Linear(h, cfg.d_model),
        )

    def forward(self, parent_hidden: torch.Tensor) -> torch.Tensor:
        return self.predictor(parent_hidden)

    def score(self, parent_hidden: torch.Tensor, child_hidden: torch.Tensor) -> torch.Tensor:
        pred = self.forward(parent_hidden)
        return -((pred - child_hidden) ** 2).mean(dim=-1)


class Verifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.verifier_hidden
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, parent_hidden: torch.Tensor, child_hidden: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([parent_hidden, child_hidden], dim=-1)).squeeze(-1)


class SGJM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.drafter = Drafter(cfg)
        self.judge = JepaJudge(cfg)
        self.verifier = Verifier(cfg)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def param_breakdown(self) -> dict[str, int]:
        return {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in (
                ("backbone", self.backbone),
                ("drafter", self.drafter),
                ("judge", self.judge),
                ("verifier", self.verifier),
            )
        }
