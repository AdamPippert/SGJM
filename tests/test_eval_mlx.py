from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")

from sgjm.training.config import TrainingConfig
from sgjm.training.data import ByteDataset, synthetic_corpus


def test_mlx_evaluate_sgjm_returns_finite_metrics():
    """evaluate_sgjm over MLX model returns valid metric values."""
    from sgjm.eval.mlx_metrics import evaluate_sgjm
    from sgjm.training.mlx_backend.model import SGJM

    cfg = TrainingConfig.smoke()
    model = SGJM(cfg.model)
    corpus = synthetic_corpus(4096, seed=0)
    ds = ByteDataset(corpus, cfg.optim.seq_len)
    metrics = evaluate_sgjm(
        model,
        cfg,
        ds,
        n_batches=2,
        n_distractors=4,
        n_merge_pairs=128,
        drafts_per_step=2,
        seed=42,
    )
    assert 0.0 <= metrics.branch_acceptance_rate <= 1.0
    assert 0.0 <= metrics.jepa_top1_acc <= 1.0
    assert metrics.compute_per_accepted_token > 0


def test_mlx_evaluate_baseline_returns_finite_metrics():
    """evaluate_baseline over MLX model returns valid metric values."""
    from sgjm.eval.mlx_metrics import evaluate_baseline
    from sgjm.training.mlx_backend.baseline import BaselineLM

    cfg = TrainingConfig.smoke()
    model = BaselineLM(cfg.model)
    corpus = synthetic_corpus(4096, seed=0)
    ds = ByteDataset(corpus, cfg.optim.seq_len)
    metrics = evaluate_baseline(model, cfg, ds, n_batches=2, seed=42)
    assert metrics.token_nll > 0
    assert metrics.compute_per_token > 0


def test_mlx_evaluate_sgjm_non_nan_fields():
    """All non-merge fields in SGJM metrics should be finite."""
    from sgjm.eval.mlx_metrics import evaluate_sgjm
    from sgjm.training.mlx_backend.model import SGJM

    cfg = TrainingConfig.smoke()
    model = SGJM(cfg.model)
    corpus = synthetic_corpus(4096, seed=1)
    ds = ByteDataset(corpus, cfg.optim.seq_len)
    metrics = evaluate_sgjm(
        model,
        cfg,
        ds,
        n_batches=2,
        n_distractors=4,
        n_merge_pairs=128,
        drafts_per_step=2,
        seed=99,
    )
    nan_ok = {"merge_precision_js", "random_pair_js"}
    for k, v in metrics.to_dict().items():
        if isinstance(v, float) and k not in nan_ok:
            assert v == v, f"{k} is NaN"


def test_mlx_evaluate_baseline_token_count_correct():
    """n_tokens in BaselineEvalMetrics matches expected token count."""
    from sgjm.eval.mlx_metrics import evaluate_baseline
    from sgjm.training.mlx_backend.baseline import BaselineLM

    cfg = TrainingConfig.smoke()
    model = BaselineLM(cfg.model)
    corpus = synthetic_corpus(4096, seed=2)
    ds = ByteDataset(corpus, cfg.optim.seq_len)
    n_batches = 3
    metrics = evaluate_baseline(model, cfg, ds, n_batches=n_batches, seed=7)
    expected = n_batches * cfg.optim.batch_size * cfg.optim.seq_len
    assert metrics.n_tokens == expected
