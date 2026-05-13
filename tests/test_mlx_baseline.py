from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")

from sgjm.training.config import TrainingConfig
from sgjm.training.data import ByteDataset, synthetic_corpus


def test_baseline_param_count_within_10pct_of_sgjm():
    """Baseline parameter count must be within 10% of SGJM total."""
    from sgjm.training.mlx_backend.baseline import BaselineLM
    from sgjm.training.mlx_backend.model import SGJM

    cfg = TrainingConfig.sgjm_25m()
    sgjm = SGJM(cfg.model)
    baseline = BaselineLM(cfg.model)
    sgjm_n = sgjm.num_parameters()
    base_n = baseline.num_parameters()
    assert 0.9 * sgjm_n <= base_n <= 1.1 * sgjm_n, (
        f"baseline {base_n/1e6:.2f}M vs sgjm {sgjm_n/1e6:.2f}M — more than 10% apart"
    )


def test_baseline_forward_returns_correct_shapes():
    """BaselineLM.__call__ returns (hidden, logits) with correct shapes."""
    from sgjm.training.mlx_backend.baseline import BaselineLM

    cfg = TrainingConfig.smoke()
    model = BaselineLM(cfg.model)
    import mlx.core as mx

    B, T = 2, cfg.optim.seq_len
    idx = mx.zeros((B, T), dtype=mx.int32)
    hidden, logits = model(idx)
    assert hidden.shape == (B, T, cfg.model.d_model)
    assert logits.shape == (B, T, cfg.model.vocab_size)


def test_mlx_baseline_num_parameters_positive():
    """num_parameters returns a positive integer."""
    from sgjm.training.mlx_backend.baseline import BaselineLM

    cfg = TrainingConfig.smoke()
    model = BaselineLM(cfg.model)
    n = model.num_parameters()
    assert isinstance(n, int)
    assert n > 0


def test_mlx_baseline_smoke_train(tmp_path):
    """Baseline training with arch='baseline' completes without error."""
    cfg = TrainingConfig.smoke()
    cfg.arch = "baseline"
    cfg.checkpoint_dir = str(tmp_path / "baseline")
    from sgjm.training.mlx_backend.trainer import train

    result = train(cfg, "mlx")
    assert result.final_step == cfg.optim.max_steps - 1
    assert result.checkpoint_path is not None
