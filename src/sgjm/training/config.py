from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 512
    block_size: int = 4
    drafter_layers: int = 2
    drafter_d_model: int = 192
    drafter_heads: int = 4
    drafter_d_ff: int = 768
    judge_hidden: int = 512
    verifier_hidden: int = 256
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Hybrid backbone: 0 = pure transformer; k = full attention every k layers
    attn_every_n: int = 0
    mamba_state_size: int = 64   # SSM state dim N
    mamba_expand: int = 2        # d_inner = d_model * mamba_expand
    mamba_d_conv: int = 4        # depthwise conv kernel size
    mamba_head_dim: int = 64     # SSM head dim; n_heads_mamba = d_inner // mamba_head_dim
    mamba_chunk_size: int = 64   # chunk size for SSD parallel scan
    # Baseline backbone is sized to match the SGJM *total* (backbone + drafter
    # + judge + verifier) so comparisons are at equal parameter budget.
    baseline_n_layers: int = 11

    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads


def is_attn_layer(layer_idx: int, attn_every_n: int) -> bool:
    """Return True if layer_idx should be a full-attention block.

    When attn_every_n == 0 all layers are attention (pure transformer).
    Otherwise, layers where (layer_idx + 1) % attn_every_n == 0 are attention;
    the rest are Mamba-2 SSM blocks.
    """
    if attn_every_n == 0:
        return True
    return (layer_idx + 1) % attn_every_n == 0


@dataclass
class OptimConfig:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_clip: float = 1.0
    batch_size: int = 16
    seq_len: int = 256
    eval_batches: int = 16


@dataclass
class LossWeights:
    token: float = 1.0
    drafter: float = 0.5
    jepa: float = 0.25
    verifier: float = 0.1


@dataclass
class TrainingConfig:
    backend: str = "auto"
    arch: str = "sgjm"  # "sgjm" | "baseline"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    data_path: str | None = None
    data_source: str = "auto"  # "auto" | "synthetic" | "tinyshakespeare" | "file"
    corpus_bytes: int = 1 << 20
    checkpoint_dir: str = "runs/sgjm-25m"
    log_every: int = 25
    eval_every: int = 500
    checkpoint_every: int = 500
    amp: str = "auto"  # "auto" | "off" | "bf16" | "fp16"
    compile: bool = False

    @classmethod
    def sgjm_25m(cls) -> "TrainingConfig":
        return cls()

    @classmethod
    def sgjm_100m(cls) -> "TrainingConfig":
        return cls(
            model=ModelConfig(
                d_model=768,
                n_layers=9,
                n_heads=12,
                d_ff=3072,
                drafter_layers=2,
                drafter_d_model=384,
                drafter_heads=6,
                drafter_d_ff=1536,
                judge_hidden=1024,
                verifier_hidden=512,
                block_size=4,
                max_seq_len=1024,
                baseline_n_layers=10,
            ),
            optim=OptimConfig(
                lr=1.5e-4,
                batch_size=4,
                seq_len=512,
                max_steps=20000,
                warmup_steps=1000,
                eval_batches=8,
            ),
            checkpoint_dir="runs/sgjm-100m",
        )

    @classmethod
    def sgjm_250m(cls) -> "TrainingConfig":
        return cls(
            model=ModelConfig(
                d_model=1024,
                n_layers=14,
                n_heads=16,
                d_ff=4096,
                drafter_layers=2,
                drafter_d_model=512,
                drafter_heads=8,
                drafter_d_ff=2048,
                judge_hidden=2048,
                verifier_hidden=1024,
                block_size=4,
                max_seq_len=1024,
                baseline_n_layers=18,
            ),
            optim=OptimConfig(
                lr=1e-4,
                batch_size=4,
                seq_len=512,
                max_steps=10000,
                warmup_steps=1000,
                eval_batches=8,
            ),
            corpus_bytes=32 << 20,  # 32 MiB — current local benchmark corpus
            checkpoint_dir="runs/sgjm-250m",
        )

    @classmethod
    def sgjm_1b(cls) -> "TrainingConfig":
        return cls(
            model=ModelConfig(
                d_model=2048,
                n_layers=20,
                n_heads=16,
                d_ff=8192,
                drafter_layers=3,
                drafter_d_model=768,
                drafter_heads=12,
                drafter_d_ff=3072,
                judge_hidden=4096,
                verifier_hidden=2048,
                block_size=2,
                max_seq_len=4096,
                baseline_n_layers=26,
            ),
            optim=OptimConfig(
                lr=6e-5,
                batch_size=1,
                seq_len=2048,
                max_steps=50_000,
                warmup_steps=5_000,
                eval_batches=8,
            ),
            corpus_bytes=256 << 20,  # 256 MiB
            checkpoint_dir="runs/sgjm-1b",
        )

    @classmethod
    def sgjm_1b_smoke(cls) -> "TrainingConfig":
        """Fast smoke-test variant of the 1B config."""
        return cls(
            model=ModelConfig(
                d_model=256,
                n_layers=2,
                n_heads=8,
                d_ff=1024,
                drafter_layers=1,
                drafter_d_model=128,
                drafter_heads=8,
                drafter_d_ff=512,
                judge_hidden=256,
                verifier_hidden=128,
                block_size=2,
                max_seq_len=128,
                baseline_n_layers=3,
            ),
            optim=OptimConfig(
                lr=6e-5,
                batch_size=1,
                seq_len=64,
                max_steps=4,
                warmup_steps=1,
                eval_batches=2,
            ),
            log_every=1,
            eval_every=1000,
            checkpoint_every=1000,
            corpus_bytes=8192,
            checkpoint_dir="runs/sgjm-1b-smoke",
            amp="off",
        )

    @classmethod
    def sgjm_250m_smoke(cls) -> "TrainingConfig":
        """Fast smoke-test variant of the 250M config."""
        return cls(
            model=ModelConfig(
                d_model=128,
                n_layers=2,
                n_heads=4,
                d_ff=512,
                drafter_layers=1,
                drafter_d_model=64,
                drafter_heads=4,
                drafter_d_ff=256,
                judge_hidden=128,
                verifier_hidden=64,
                block_size=4,
                max_seq_len=128,
                baseline_n_layers=3,
            ),
            optim=OptimConfig(
                lr=1e-3,
                batch_size=4,
                seq_len=64,
                max_steps=4,
                warmup_steps=1,
                eval_batches=2,
            ),
            log_every=1,
            eval_every=1000,
            checkpoint_every=1000,
            corpus_bytes=8192,
            checkpoint_dir="runs/sgjm-250m-smoke",
            amp="off",
        )

    @classmethod
    def sgjm_100m_smoke(cls) -> "TrainingConfig":
        """Fast smoke-test variant that exercises the 100M config class."""
        return cls(
            model=ModelConfig(
                d_model=64,
                n_layers=2,
                n_heads=4,
                d_ff=256,
                drafter_layers=1,
                drafter_d_model=32,
                drafter_heads=4,
                drafter_d_ff=128,
                judge_hidden=64,
                verifier_hidden=64,
                block_size=4,
                max_seq_len=128,
                baseline_n_layers=3,
            ),
            optim=OptimConfig(
                lr=1.5e-4,
                batch_size=4,
                seq_len=64,
                max_steps=4,
                warmup_steps=1,
                eval_batches=2,
            ),
            log_every=1,
            eval_every=1000,
            checkpoint_every=1000,
            corpus_bytes=8192,
            checkpoint_dir="runs/sgjm-100m-smoke",
            amp="off",
        )

    @classmethod
    def smoke(cls) -> "TrainingConfig":
        return cls(
            model=ModelConfig(
                d_model=64,
                n_layers=2,
                n_heads=2,
                d_ff=128,
                drafter_layers=1,
                drafter_d_model=32,
                drafter_heads=2,
                drafter_d_ff=64,
                judge_hidden=64,
                verifier_hidden=64,
                block_size=2,
                max_seq_len=64,
                baseline_n_layers=3,
            ),
            optim=OptimConfig(
                batch_size=4,
                seq_len=32,
                max_steps=4,
                warmup_steps=1,
                lr=1e-3,
                eval_batches=2,
            ),
            log_every=1,
            eval_every=1000,
            checkpoint_every=1000,
            corpus_bytes=8192,
            checkpoint_dir="runs/sgjm-smoke",
            amp="off",
        )

    @classmethod
    def sgjm_25m_hybrid(cls) -> "TrainingConfig":
        """25M hybrid: same depth/width as 25M baseline, 1 attention + 9 Mamba-2 blocks.

        Parameter count is ~13-15M (smaller than 25M baseline since Mamba-2 blocks
        have fewer params than transformer blocks at the same d_model).
        """
        cfg = cls.sgjm_25m()
        return replace(
            cfg,
            model=replace(cfg.model, attn_every_n=8, mamba_state_size=64),
            checkpoint_dir="runs/sgjm-25m-hybrid",
        )

    @classmethod
    def sgjm_250m_hybrid(cls) -> "TrainingConfig":
        """250M hybrid: same depth/width as 250M baseline, 1 attention + 13 Mamba-2 blocks."""
        cfg = cls.sgjm_250m()
        return replace(
            cfg,
            model=replace(cfg.model, attn_every_n=8, mamba_state_size=128),
            checkpoint_dir="runs/sgjm-250m-hybrid",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        model = ModelConfig(**data.get("model", {}))
        optim = OptimConfig(**data.get("optim", {}))
        loss = LossWeights(**data.get("loss", {}))
        rest = {
            k: v
            for k, v in data.items()
            if k not in {"model", "optim", "loss"}
        }
        return cls(model=model, optim=optim, loss=loss, **rest)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "TrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def with_overrides(self, **kwargs: Any) -> "TrainingConfig":
        return replace(self, **kwargs)
