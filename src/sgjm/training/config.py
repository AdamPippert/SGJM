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
    # Baseline backbone is sized to match the SGJM *total* (backbone + drafter
    # + judge + verifier) so comparisons are at equal parameter budget.
    baseline_n_layers: int = 11

    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads


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
