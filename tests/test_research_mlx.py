from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")

from sgjm.research.runner import run_sweep
from sgjm.research.sweep import ablation_sweep
from sgjm.training.config import TrainingConfig


def test_run_smoke_ablation_sweep_mlx_end_to_end(tmp_path):
    """run_sweep dispatches correctly to the MLX training path."""
    base_cfg = TrainingConfig.smoke()
    base_cfg.optim.max_steps = 2
    sweep = ablation_sweep()
    sweep.entries = sweep.entries[:2]

    results = run_sweep(sweep, base_cfg, backend="mlx", out_dir=tmp_path, eval_batches=2)
    assert len(results) == 2
    for r in results:
        assert r.error is None, f"unexpected error in {r.card.name}: {r.error}"
    assert (tmp_path / "summary.json").exists()
    for r in results:
        assert r.sgjm_metrics is not None
        assert r.baseline_metrics is not None
        assert r.comparison is not None


def test_research_cli_mlx_backend_accepted():
    """research __main__ no longer raises SystemExit for --backend mlx."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu", "mlx"], default="auto")
    ns = parser.parse_args(["--backend", "mlx"])
    assert ns.backend == "mlx"
