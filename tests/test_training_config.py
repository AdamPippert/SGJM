import json

import pytest

from sgjm.training.backends import resolve_backend
from sgjm.training.config import LossWeights, ModelConfig, OptimConfig, TrainingConfig


def test_default_25m_config_is_sane():
    cfg = TrainingConfig.sgjm_25m()
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.model.drafter_d_model % cfg.model.drafter_heads == 0
    assert cfg.model.block_size >= 1
    assert cfg.optim.seq_len > cfg.model.block_size
    assert cfg.optim.batch_size >= 1


def test_smoke_config_is_tiny():
    cfg = TrainingConfig.smoke()
    assert cfg.optim.max_steps <= 8
    assert cfg.optim.batch_size <= 8
    assert cfg.optim.seq_len <= 64


def test_config_round_trip_json(tmp_path):
    cfg = TrainingConfig.sgjm_25m()
    cfg.optim.max_steps = 17
    cfg.loss.token = 0.7
    path = tmp_path / "cfg.json"
    cfg.save_json(path)
    loaded = TrainingConfig.load_json(path)
    assert loaded.optim.max_steps == 17
    assert loaded.loss.token == 0.7
    assert loaded.model.d_model == cfg.model.d_model


def test_resolve_backend_explicit():
    assert resolve_backend("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_backend("metal")


def test_resolve_backend_auto_returns_known():
    assert resolve_backend("auto") in {"cpu", "cuda", "rocm", "mlx"}


def test_100m_config_is_sane():
    cfg = TrainingConfig.sgjm_100m()
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.model.drafter_d_model % cfg.model.drafter_heads == 0
    assert cfg.optim.seq_len <= cfg.model.max_seq_len
    assert cfg.checkpoint_dir == "runs/sgjm-100m"


def test_100m_smoke_config_is_tiny():
    cfg = TrainingConfig.sgjm_100m_smoke()
    assert cfg.optim.max_steps <= 8
    assert cfg.model.d_model <= 128
    assert cfg.checkpoint_dir == "runs/sgjm-100m-smoke"
