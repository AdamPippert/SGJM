"""Behavior tests for the 1B TrainingConfig."""
from __future__ import annotations

from pathlib import Path

from sgjm.training.config import TrainingConfig


def test_1b_config_head_divisibility():
    cfg = TrainingConfig.sgjm_1b()
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.model.drafter_d_model % cfg.model.drafter_heads == 0


def test_1b_config_seq_fits_in_max_seq_len():
    cfg = TrainingConfig.sgjm_1b()
    assert cfg.optim.seq_len <= cfg.model.max_seq_len


def test_1b_config_checkpoint_dir():
    cfg = TrainingConfig.sgjm_1b()
    assert "1b" in cfg.checkpoint_dir


def test_1b_config_param_scale():
    """1B backbone d_model is larger than 250M."""
    cfg_250m = TrainingConfig.sgjm_250m()
    cfg_1b = TrainingConfig.sgjm_1b()
    assert cfg_1b.model.d_model > cfg_250m.model.d_model
    assert cfg_1b.model.d_model >= 2048


def test_1b_config_corpus_larger_than_250m():
    cfg_250m = TrainingConfig.sgjm_250m()
    cfg_1b = TrainingConfig.sgjm_1b()
    assert cfg_1b.corpus_bytes > cfg_250m.corpus_bytes


def test_1b_config_steps_at_least_20k():
    cfg = TrainingConfig.sgjm_1b()
    assert cfg.optim.max_steps >= 20_000


def test_1b_smoke_config_is_tiny():
    cfg = TrainingConfig.sgjm_1b_smoke()
    assert cfg.optim.max_steps <= 8
    assert cfg.model.d_model <= 256


def test_1b_config_round_trips_json(tmp_path: Path):
    cfg = TrainingConfig.sgjm_1b()
    p = tmp_path / "cfg_1b.json"
    cfg.save_json(p)
    loaded = TrainingConfig.load_json(p)
    assert loaded.model.d_model == cfg.model.d_model
    assert loaded.model.n_layers == cfg.model.n_layers
    assert loaded.checkpoint_dir == cfg.checkpoint_dir
    assert loaded.corpus_bytes == cfg.corpus_bytes


def test_1b_config_lr_lower_than_250m():
    """Larger models need a smaller peak LR to stay stable."""
    cfg_250m = TrainingConfig.sgjm_250m()
    cfg_1b = TrainingConfig.sgjm_1b()
    assert cfg_1b.optim.lr < cfg_250m.optim.lr
