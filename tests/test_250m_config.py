"""Behavior tests for the 250M config and extended Python corpus loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from sgjm.training.config import TrainingConfig
from sgjm.training.data import ByteDataset, load_corpus


# ---------------------------------------------------------------------------
# 250M config
# ---------------------------------------------------------------------------

def test_250m_config_head_divisibility():
    cfg = TrainingConfig.sgjm_250m()
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.model.drafter_d_model % cfg.model.drafter_heads == 0


def test_250m_config_seq_fits_in_max_seq_len():
    cfg = TrainingConfig.sgjm_250m()
    assert cfg.optim.seq_len <= cfg.model.max_seq_len


def test_250m_config_checkpoint_dir():
    cfg = TrainingConfig.sgjm_250m()
    assert "250m" in cfg.checkpoint_dir


def test_250m_config_param_scale():
    """250M config is larger than 100M."""
    cfg_100m = TrainingConfig.sgjm_100m()
    cfg_250m = TrainingConfig.sgjm_250m()
    assert cfg_250m.model.d_model > cfg_100m.model.d_model
    assert cfg_250m.model.d_model >= 1024


def test_250m_smoke_config_is_tiny():
    cfg = TrainingConfig.sgjm_250m_smoke()
    assert cfg.optim.max_steps <= 8
    assert cfg.model.d_model <= 128


def test_250m_config_round_trips_json(tmp_path: Path):
    cfg = TrainingConfig.sgjm_250m()
    p = tmp_path / "cfg.json"
    cfg.save_json(p)
    loaded = TrainingConfig.load_json(p)
    assert loaded.model.d_model == cfg.model.d_model
    assert loaded.model.n_layers == cfg.model.n_layers
    assert loaded.checkpoint_dir == cfg.checkpoint_dir


# ---------------------------------------------------------------------------
# Extended Python corpus (stdlib + site-packages)
# ---------------------------------------------------------------------------

def test_load_python_extended_returns_more_bytes_than_stdlib():
    """python_extended produces a larger corpus than python (stdlib only)."""
    stdlib_corpus = load_corpus(source="python", n_bytes=1 << 23)  # 8 MiB cap
    extended_corpus = load_corpus(source="python_extended", n_bytes=1 << 23)
    # Both capped at same n_bytes, but extended should at least equal stdlib
    assert len(extended_corpus) >= len(stdlib_corpus)


def test_load_python_extended_respects_n_bytes():
    corpus = load_corpus(source="python_extended", n_bytes=16384)
    assert len(corpus) <= 16384


def test_load_python_extended_contains_python_syntax():
    corpus = load_corpus(source="python_extended", n_bytes=65536)
    text = corpus.decode("utf-8", errors="replace")
    assert "def " in text
    assert "import " in text


def test_load_python_extended_usable_in_bytedataset():
    corpus = load_corpus(source="python_extended", n_bytes=32768)
    ds = ByteDataset(corpus, seq_len=64)
    import random
    x, y = ds.sample(random.Random(7))
    assert len(x) == 64
    assert all(0 <= t <= 255 for t in x)
