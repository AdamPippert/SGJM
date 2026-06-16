"""HuggingFace configuration for the SGJM family.

Self-contained on the `transformers` side: this file mirrors every field of the
library's :class:`sgjm.training.config.ModelConfig` so a published `config.json`
round-trips without the training package. Building an actual model
(:meth:`to_model_config`) does import the library — see ``modeling_sgjm.py``.
"""
from __future__ import annotations

from transformers import PretrainedConfig

# The architecture-defining fields, in ModelConfig order. Kept as a list so the
# config and the converter stay in lockstep with the library dataclass.
MODEL_FIELDS = (
    "vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len",
    "block_size", "drafter_layers", "drafter_d_model", "drafter_heads",
    "drafter_d_ff", "judge_hidden", "verifier_hidden", "dropout",
    "tie_embeddings", "attn_every_n", "mamba_state_size", "mamba_expand",
    "mamba_d_conv", "mamba_head_dim", "mamba_chunk_size", "baseline_n_layers",
)


class SGJMConfig(PretrainedConfig):
    model_type = "sgjm"

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 384,
        n_layers: int = 10,
        n_heads: int = 6,
        d_ff: int = 1536,
        max_seq_len: int = 512,
        block_size: int = 4,
        drafter_layers: int = 2,
        drafter_d_model: int = 192,
        drafter_heads: int = 4,
        drafter_d_ff: int = 768,
        judge_hidden: int = 512,
        verifier_hidden: int = 256,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        attn_every_n: int = 0,
        mamba_state_size: int = 64,
        mamba_expand: int = 2,
        mamba_d_conv: int = 4,
        mamba_head_dim: int = 64,
        mamba_chunk_size: int = 64,
        baseline_n_layers: int = 11,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.drafter_layers = drafter_layers
        self.drafter_d_model = drafter_d_model
        self.drafter_heads = drafter_heads
        self.drafter_d_ff = drafter_d_ff
        self.judge_hidden = judge_hidden
        self.verifier_hidden = verifier_hidden
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.attn_every_n = attn_every_n
        self.mamba_state_size = mamba_state_size
        self.mamba_expand = mamba_expand
        self.mamba_d_conv = mamba_d_conv
        self.mamba_head_dim = mamba_head_dim
        self.mamba_chunk_size = mamba_chunk_size
        self.baseline_n_layers = baseline_n_layers
        # Informational; derived from attn_every_n.
        self.backbone = "hybrid-mamba2-attention" if attn_every_n else "transformer"
        # Standard HF aliases so generic transformers utilities can introspect us.
        self.num_hidden_layers = n_layers
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        # The backbone keeps no KV cache; generation re-feeds the context each step.
        self.use_cache = kwargs.pop("use_cache", False)
        # Let HF know input/output embeddings are tied (matches the library model).
        kwargs.setdefault("tie_word_embeddings", tie_embeddings)
        super().__init__(**kwargs)

    def to_model_config(self):
        """Return the library :class:`ModelConfig` this describes."""
        from sgjm.training.config import ModelConfig

        return ModelConfig(**{f: getattr(self, f) for f in MODEL_FIELDS})

    @classmethod
    def from_model_config(cls, model_cfg, **kwargs) -> "SGJMConfig":
        """Build an HF config from a library ModelConfig (used by the converter)."""
        return cls(**{f: getattr(model_cfg, f) for f in MODEL_FIELDS}, **kwargs)
