"""Tests for the hybrid Mamba-2 / full-attention backbone.

Written FIRST per TDD mandate. All tests should fail before implementation.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlx_hybrid_config(
    d_model: int = 32,
    n_layers: int = 10,
    n_heads: int = 4,
    d_ff: int = 64,
    attn_every_n: int = 8,
    mamba_state_size: int = 8,
    mamba_expand: int = 2,
    mamba_d_conv: int = 4,
    mamba_head_dim: int = 8,
    mamba_chunk_size: int = 4,
    vocab_size: int = 64,
    max_seq_len: int = 64,
):
    from sgjm.training.config import ModelConfig

    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        attn_every_n=attn_every_n,
        mamba_state_size=mamba_state_size,
        mamba_expand=mamba_expand,
        mamba_d_conv=mamba_d_conv,
        mamba_head_dim=mamba_head_dim,
        mamba_chunk_size=mamba_chunk_size,
    )


# ---------------------------------------------------------------------------
# 1. Config tests (no MLX required)
# ---------------------------------------------------------------------------


class TestHybridConfig:
    def test_sgjm_25m_hybrid_config_loads(self):
        """TrainingConfig.sgjm_25m_hybrid() must load without error."""
        from sgjm.training.config import TrainingConfig

        cfg = TrainingConfig.sgjm_25m_hybrid()
        assert cfg is not None

    def test_sgjm_25m_hybrid_has_attn_every_n_8(self):
        """25m hybrid must set attn_every_n=8."""
        from sgjm.training.config import TrainingConfig

        cfg = TrainingConfig.sgjm_25m_hybrid()
        assert cfg.model.attn_every_n == 8

    def test_sgjm_25m_hybrid_checkpoint_dir(self):
        """25m hybrid must use a distinct checkpoint directory."""
        from sgjm.training.config import TrainingConfig

        cfg = TrainingConfig.sgjm_25m_hybrid()
        assert "hybrid" in cfg.checkpoint_dir

    def test_sgjm_250m_hybrid_config_loads(self):
        """TrainingConfig.sgjm_250m_hybrid() must load without error."""
        from sgjm.training.config import TrainingConfig

        cfg = TrainingConfig.sgjm_250m_hybrid()
        assert cfg is not None

    def test_sgjm_250m_hybrid_has_attn_every_n_8(self):
        """250m hybrid must set attn_every_n=8."""
        from sgjm.training.config import TrainingConfig

        cfg = TrainingConfig.sgjm_250m_hybrid()
        assert cfg.model.attn_every_n == 8

    def test_sgjm_250m_hybrid_checkpoint_dir(self):
        """250m hybrid must use a distinct checkpoint directory."""
        from sgjm.training.config import TrainingConfig

        cfg = TrainingConfig.sgjm_250m_hybrid()
        assert "hybrid" in cfg.checkpoint_dir

    def test_default_modelconfig_attn_every_n_zero(self):
        """Default ModelConfig must have attn_every_n=0 (pure transformer)."""
        from sgjm.training.config import ModelConfig

        cfg = ModelConfig()
        assert cfg.attn_every_n == 0

    def test_mamba_fields_have_defaults(self):
        """All new mamba fields must have sensible defaults."""
        from sgjm.training.config import ModelConfig

        cfg = ModelConfig()
        assert cfg.mamba_state_size == 64
        assert cfg.mamba_expand == 2
        assert cfg.mamba_d_conv == 4
        assert cfg.mamba_head_dim == 64
        assert cfg.mamba_chunk_size == 64


# ---------------------------------------------------------------------------
# 2. Layer allocation logic (pure Python, no ML framework required)
# ---------------------------------------------------------------------------


class TestLayerAllocation:
    def test_is_attn_layer_zero_means_all_attention(self):
        """attn_every_n=0 → is_attn_layer always returns True (pure transformer)."""
        from sgjm.training.config import is_attn_layer

        for i in range(12):
            assert is_attn_layer(i, 0) is True

    def test_is_attn_layer_n8_correct_indices(self):
        """attn_every_n=8 → attention only at layers where (i+1) % 8 == 0."""
        from sgjm.training.config import is_attn_layer

        attn_layers = [i for i in range(16) if is_attn_layer(i, 8)]
        # layer indices 7, 15 (0-indexed) are attention
        assert attn_layers == [7, 15]

    def test_is_attn_layer_n8_ten_layers(self):
        """With n_layers=10 and attn_every_n=8, exactly one attention layer (idx 7)."""
        from sgjm.training.config import is_attn_layer

        attn_layers = [i for i in range(10) if is_attn_layer(i, 8)]
        assert attn_layers == [7]
        assert len(attn_layers) == 1

    def test_mlx_model_exports_is_attn_layer(self):
        """mlx_backend.model re-exports _is_attn_layer for backward compat."""
        pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.model import _is_attn_layer

        assert _is_attn_layer(7, 8) is True
        assert _is_attn_layer(6, 8) is False

    def test_hybrid_backbone_block_types(self):
        """Backbone with attn_every_n=8 has correct ratio of block types."""
        pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.mamba2 import Mamba2Block
        from sgjm.training.mlx_backend.model import Backbone, Block

        cfg = _make_mlx_hybrid_config(n_layers=10, attn_every_n=8)
        backbone = Backbone(cfg)
        attn_count = sum(1 for b in backbone.blocks if isinstance(b, Block))
        mamba_count = sum(1 for b in backbone.blocks if isinstance(b, Mamba2Block))
        assert attn_count == 1
        assert mamba_count == 9

    def test_pure_transformer_all_attention_blocks(self):
        """attn_every_n=0 → all blocks are attention blocks."""
        pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.model import Backbone, Block

        cfg = _make_mlx_hybrid_config(n_layers=4, attn_every_n=0)
        backbone = Backbone(cfg)
        assert all(isinstance(b, Block) for b in backbone.blocks)


# ---------------------------------------------------------------------------
# 3. MLX Mamba2Block tests
# ---------------------------------------------------------------------------


class TestMamba2BlockMLX:
    def test_mamba2_block_output_shape(self):
        """Mamba2Block([d_model=32, ...]) with [2, 16, 32] input → [2, 16, 32] output."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.mamba2 import Mamba2Block

        block = Mamba2Block(
            d_model=32,
            state_size=8,
            expand=2,
            d_conv=4,
            head_dim=8,
            chunk_size=4,
        )
        x = mx.random.normal((2, 16, 32))
        y = block(x)
        mx.eval(y)
        assert y.shape == (2, 16, 32)

    def test_mamba2_block_causal(self):
        """Output at position ≤7 is unchanged when position ≥8 input changes."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.mamba2 import Mamba2Block

        block = Mamba2Block(
            d_model=32,
            state_size=8,
            expand=2,
            d_conv=4,
            head_dim=8,
            chunk_size=4,
        )
        x1 = mx.random.normal((1, 16, 32))
        x2 = mx.array(x1)
        noise = mx.random.normal((1, 8, 32)) * 0.1
        # Change positions 8-15 only
        x2 = mx.concatenate([x2[:, :8, :], x2[:, 8:, :] + noise], axis=1)

        y1 = block(x1)
        y2 = block(x2)
        mx.eval(y1, y2)

        # First 8 positions must be identical
        diff = mx.abs(y1[:, :8, :] - y2[:, :8, :]).max().item()
        assert diff < 1e-5, f"Causality violated: max diff = {diff}"

    def test_mamba2_block_dtype_preserved(self):
        """Output dtype matches input dtype."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.mamba2 import Mamba2Block

        block = Mamba2Block(d_model=32, state_size=8, expand=2, d_conv=4, head_dim=8, chunk_size=4)
        x = mx.random.normal((1, 8, 32)).astype(mx.float32)
        y = block(x)
        mx.eval(y)
        assert y.dtype == mx.float32

    def test_mamba2_block_residual_connection(self):
        """Output has the correct shape (transformation applies without error)."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.mamba2 import Mamba2Block

        block = Mamba2Block(d_model=32, state_size=8, expand=2, d_conv=4, head_dim=8, chunk_size=4)
        x = mx.ones((1, 8, 32))
        y = block(x)
        mx.eval(y)
        assert y.shape == (1, 8, 32)

    def test_mamba2_block_batch_independence(self):
        """Each batch element is processed independently."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.mamba2 import Mamba2Block

        block = Mamba2Block(d_model=32, state_size=8, expand=2, d_conv=4, head_dim=8, chunk_size=4)
        x = mx.random.normal((3, 8, 32))
        y_batch = block(x)

        # Process each sample individually
        y_singles = mx.concatenate([block(x[i : i + 1]) for i in range(3)], axis=0)
        mx.eval(y_batch, y_singles)

        diff = mx.abs(y_batch - y_singles).max().item()
        assert diff < 1e-4, f"Batch independence violated: max diff = {diff}"

    def test_mamba2_block_chunk_boundary(self):
        """Output shape is correct when T is not a multiple of chunk_size."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.mamba2 import Mamba2Block

        block = Mamba2Block(d_model=32, state_size=8, expand=2, d_conv=4, head_dim=8, chunk_size=4)
        # T=7 is not a multiple of chunk_size=4
        x = mx.random.normal((2, 7, 32))
        y = block(x)
        mx.eval(y)
        assert y.shape == (2, 7, 32)


# ---------------------------------------------------------------------------
# 4. MLX SGJM hybrid forward pass tests
# ---------------------------------------------------------------------------


class TestSGJMHybridMLX:
    def test_sgjm_hybrid_forward_shape(self):
        """SGJM with hybrid config produces hidden states [B, T, d_model] and logits [B, T, vocab]."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.model import SGJM

        cfg = _make_mlx_hybrid_config(
            d_model=32,
            n_layers=10,
            n_heads=4,
            d_ff=64,
            attn_every_n=8,
            mamba_state_size=8,
            mamba_expand=2,
            mamba_d_conv=4,
            mamba_head_dim=8,
            mamba_chunk_size=4,
            vocab_size=64,
            max_seq_len=32,
        )
        model = SGJM(cfg)
        idx = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 2)
        h, logits = model(idx)
        mx.eval(h, logits)
        assert h.shape == (2, 8, 32)
        assert logits.shape == (2, 8, 64)

    def test_sgjm_pure_transformer_backward_compat(self):
        """attn_every_n=0 (default) → same shape output as before (backward compat)."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.model import SGJM

        cfg = _make_mlx_hybrid_config(
            d_model=32,
            n_layers=4,
            n_heads=4,
            d_ff=64,
            attn_every_n=0,  # pure transformer
            vocab_size=64,
            max_seq_len=32,
        )
        model = SGJM(cfg)
        idx = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 2)
        h, logits = model(idx)
        mx.eval(h, logits)
        assert h.shape == (2, 8, 32)
        assert logits.shape == (2, 8, 64)

    def test_sgjm_25m_hybrid_instantiates(self):
        """TrainingConfig.sgjm_25m_hybrid() can instantiate SGJM without error."""
        pytest.importorskip("mlx.core")
        from sgjm.training.config import TrainingConfig
        from sgjm.training.mlx_backend.model import SGJM

        cfg = TrainingConfig.sgjm_25m_hybrid()
        model = SGJM(cfg.model)
        assert model is not None


# ---------------------------------------------------------------------------
# 5. PyTorch Mamba2Block tests
# ---------------------------------------------------------------------------


class TestMamba2BlockTorch:
    def test_mamba2_block_output_shape_torch(self):
        """PyTorch Mamba2Block with [2, 16, 32] input → [2, 16, 32] output."""
        torch = pytest.importorskip("torch")
        from sgjm.training.torch_backend.mamba2 import Mamba2Block

        block = Mamba2Block(
            d_model=32,
            state_size=8,
            expand=2,
            d_conv=4,
            head_dim=8,
            chunk_size=4,
        )
        x = torch.randn(2, 16, 32)
        y = block(x)
        assert y.shape == (2, 16, 32)

    def test_mamba2_block_causal_torch(self):
        """PyTorch Mamba2Block: output at position ≤7 unchanged when position ≥8 input changes."""
        torch = pytest.importorskip("torch")
        from sgjm.training.torch_backend.mamba2 import Mamba2Block

        block = Mamba2Block(d_model=32, state_size=8, expand=2, d_conv=4, head_dim=8, chunk_size=4)
        block.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 16, 32)
            x2 = x1.clone()
            x2[:, 8:, :] += torch.randn(1, 8, 32) * 0.1
            y1 = block(x1)
            y2 = block(x2)
        diff = (y1[:, :8, :] - y2[:, :8, :]).abs().max().item()
        assert diff < 1e-5, f"Causality violated: max diff = {diff}"

    def test_mamba2_block_chunk_boundary_torch(self):
        """PyTorch Mamba2Block: correct output when T is not a multiple of chunk_size."""
        torch = pytest.importorskip("torch")
        from sgjm.training.torch_backend.mamba2 import Mamba2Block

        block = Mamba2Block(d_model=32, state_size=8, expand=2, d_conv=4, head_dim=8, chunk_size=4)
        x = torch.randn(2, 7, 32)
        y = block(x)
        assert y.shape == (2, 7, 32)

    def test_torch_hybrid_backbone_forward_shape(self):
        """PyTorch Backbone with hybrid config produces correct shapes."""
        torch = pytest.importorskip("torch")
        from sgjm.training.config import ModelConfig
        from sgjm.training.torch_backend.model import Backbone

        cfg = ModelConfig(
            vocab_size=64,
            d_model=32,
            n_layers=10,
            n_heads=4,
            d_ff=64,
            max_seq_len=32,
            attn_every_n=8,
            mamba_state_size=8,
            mamba_expand=2,
            mamba_d_conv=4,
            mamba_head_dim=8,
            mamba_chunk_size=4,
        )
        backbone = Backbone(cfg)
        idx = torch.randint(0, 64, (2, 8))
        h, logits = backbone(idx)
        assert h.shape == (2, 8, 32)
        assert logits.shape == (2, 8, 64)

    def test_torch_is_attn_layer_helper(self):
        """_is_attn_layer re-exported from torch_backend.model works correctly."""
        pytest.importorskip("torch")
        from sgjm.training.torch_backend.model import _is_attn_layer

        assert _is_attn_layer(7, 8) is True
        assert _is_attn_layer(6, 8) is False
        assert _is_attn_layer(0, 0) is True

    def test_no_nan_in_output_torch(self):
        """SSD scan must not produce NaN regardless of sequence length."""
        torch = pytest.importorskip("torch")
        from sgjm.training.torch_backend.mamba2 import Mamba2Block

        block = Mamba2Block(d_model=32, state_size=8, expand=2, d_conv=4, head_dim=8, chunk_size=4)
        block.eval()
        with torch.no_grad():
            # Long sequence to stress the upper-triangle exp overflow path
            x = torch.randn(2, 128, 32)
            y = block(x)
        assert not torch.isnan(y).any(), "NaN in Mamba2Block output"
        assert not torch.isinf(y).any(), "Inf in Mamba2Block output"

    def test_no_nan_mlx(self):
        """MLX SSD scan must not produce NaN regardless of sequence length."""
        mx = pytest.importorskip("mlx.core")
        from sgjm.training.mlx_backend.mamba2 import Mamba2Block

        block = Mamba2Block(d_model=32, state_size=8, expand=2, d_conv=4, head_dim=8, chunk_size=4)
        x = mx.random.normal((2, 128, 32))
        y = block(x)
        mx.eval(y)
        assert not mx.isnan(y).any().item(), "NaN in MLX Mamba2Block output"
