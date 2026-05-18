import importlib

import pytest


torch = pytest.importorskip("torch")

from sgjm.training.config import ModelConfig, TrainingConfig
from sgjm.training.data import ByteDataset, synthetic_corpus
from sgjm.training.torch_backend.losses import compute_losses
from sgjm.training.torch_backend.model import SGJM


def _smoke_cfg() -> TrainingConfig:
    cfg = TrainingConfig.smoke()
    return cfg


def test_model_param_count_25m_target():
    cfg = TrainingConfig.sgjm_25m()
    model = SGJM(cfg.model)
    n = model.num_parameters()
    # 10 layers @ d_model=384 with SwiGLU FFNs lands around 22-28M
    assert 22e6 <= n <= 28e6, f"unexpected param count {n}"


def test_smoke_model_forward_shapes():
    cfg = _smoke_cfg()
    model = SGJM(cfg.model)
    x = torch.zeros((2, cfg.optim.seq_len), dtype=torch.long)
    hidden, logits = model.backbone(x)
    assert hidden.shape == (2, cfg.optim.seq_len, cfg.model.d_model)
    assert logits.shape == (2, cfg.optim.seq_len, cfg.model.vocab_size)


def test_compute_losses_runs_and_backprops():
    cfg = _smoke_cfg()
    model = SGJM(cfg.model)
    corpus = synthetic_corpus(2048, seed=7)
    ds = ByteDataset(corpus, cfg.optim.seq_len)
    import random
    rng = random.Random(0)
    xs, ys = ds.batch(cfg.optim.batch_size, rng)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    total, parts = compute_losses(model, (x, y), cfg)
    assert torch.isfinite(total)
    for k in ("token", "drafter", "jepa", "verifier", "accept_acc"):
        assert k in parts
    total.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_trainer_smoke_run(tmp_path):
    from sgjm.training.torch_backend.trainer import train

    cfg = _smoke_cfg()
    cfg.checkpoint_dir = str(tmp_path / "run")
    result = train(cfg, backend="cpu")
    assert result.final_step == cfg.optim.max_steps - 1
    assert result.checkpoint_path is not None
    assert (tmp_path / "run" / "config.json").exists()
    assert (tmp_path / "run" / "train.jsonl").exists()


def test_verifier_negatives_differ_at_batch_size_1():
    """Regression: rolling on dim=0 at B=1 returns the identical tensor,
    giving the verifier zero net gradient and pinning accept_acc at 0.5."""
    import torch
    from sgjm.training.torch_backend.losses import compute_losses

    cfg = TrainingConfig.smoke()
    # Force batch_size=1 — the failure mode
    cfg.optim.batch_size = 1
    model = SGJM(cfg.model)
    corpus = synthetic_corpus(4096, seed=99)
    ds = ByteDataset(corpus, cfg.optim.seq_len)
    import random
    rng = random.Random(0)
    xs, ys = ds.batch(1, rng)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)

    total, parts = compute_losses(model, (x, y), cfg)

    # Verifier gradient must be non-zero at B=1; if negatives = positives the
    # gradient cancels and the verifier parameter norms never change.
    total.backward()
    verifier_grad_norm = sum(
        p.grad.abs().sum().item()
        for p in model.verifier.parameters()
        if p.grad is not None
    )
    assert verifier_grad_norm > 0, (
        "Verifier has zero gradient at batch_size=1 — "
        "negatives are identical to positives (batch-roll collapse)"
    )


def test_adapters_drive_harness(tmp_path):
    from sgjm.harness.runner import HarnessConfig, HarnessRunner
    from sgjm.training.torch_backend.adapters import bundle_for_harness

    cfg = _smoke_cfg()
    model = SGJM(cfg.model)
    backbone, drafter, judge = bundle_for_harness(model, device="cpu", temperature=1.0)
    runner = HarnessRunner(
        backbone=backbone,
        drafter=drafter,
        judge=judge,
        config=HarnessConfig(
            branches_per_step=2,
            block_size=cfg.model.block_size,
            max_steps=2,
            keep_top_k=1,
            merge_radius=2,
        ),
    )
    snap = runner.run(prompt_tokens=[1, 2, 3, 4])
    assert snap.steps > 0
    assert snap.committed >= 0
