"""Smoke tests for the MLX benchmark adapters and bench functions."""
from __future__ import annotations

import pytest

pytest.importorskip("mlx.core", reason="MLX not available")

import mlx.core as mx

from sgjm.bench.mlx_bench import (
    BenchResult,
    MLXBackboneAdapter,
    MLXDrafterAdapter,
    MLXJudgeAdapter,
    run_ar_bench,
    run_sgjm_bench,
)
from sgjm.training.config import TrainingConfig
from sgjm.training.mlx_backend.model import SGJM


def _smoke_model() -> tuple[SGJM, TrainingConfig]:
    cfg = TrainingConfig.smoke()
    model = SGJM(cfg.model)
    mx.eval(model.parameters())
    return model, cfg


def test_backbone_adapter_encode_returns_valid_state():
    model, cfg = _smoke_model()
    adapter = MLXBackboneAdapter(model)
    state = adapter.encode([10, 20, 30])
    assert state.tokens == (10, 20, 30)
    assert len(state.latent) == cfg.model.d_model
    assert all(isinstance(v, float) for v in state.latent)


def test_backbone_adapter_step_appends_token():
    model, cfg = _smoke_model()
    adapter = MLXBackboneAdapter(model)
    state = adapter.encode([1, 2])
    stepped = adapter.step(state, 99)
    assert stepped.tokens == (1, 2, 99)
    assert len(stepped.latent) == cfg.model.d_model


def test_drafter_adapter_returns_k_samples():
    model, cfg = _smoke_model()
    backbone = MLXBackboneAdapter(model)
    drafter = MLXDrafterAdapter(model, seed=7)
    state = backbone.encode([5, 6, 7, 8])
    samples = drafter.draft(state, k=3, block=cfg.model.block_size)
    assert len(samples) == 3
    for s in samples:
        assert len(s.tokens) == cfg.model.block_size
        assert len(s.latent) == cfg.model.d_model
        assert isinstance(s.log_prob, float)


def test_judge_adapter_returns_scalar():
    model, cfg = _smoke_model()
    judge = MLXJudgeAdapter(model)
    D = cfg.model.d_model
    parent = [0.1] * D
    child = [0.2] * D
    score = judge.score(parent, child)
    assert isinstance(score, float)


def test_run_sgjm_bench_smoke():
    model, cfg = _smoke_model()
    prompt = list(range(16))
    result = run_sgjm_bench(model, cfg.model, prompt, n_steps=2)
    assert isinstance(result, BenchResult)
    assert result.steps_completed >= 1
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.elapsed_sec > 0.0


def test_run_ar_bench_smoke():
    model, _ = _smoke_model()
    prompt = list(range(8))
    result = run_ar_bench(model, prompt, n_steps=2)
    assert result.tokens_generated == 2
    assert result.steps_completed == 2
    assert result.elapsed_sec > 0.0
    assert result.tokens_per_sec > 0.0
