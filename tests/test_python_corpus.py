"""Behavior tests for Python-code corpus loader and demo CLI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sgjm.training.data import ByteDataset, load_corpus


# ---------------------------------------------------------------------------
# Python corpus loader
# ---------------------------------------------------------------------------

def test_load_python_source_from_directory(tmp_path: Path):
    """load_corpus(source='python', path=dir) collects .py files."""
    (tmp_path / "a.py").write_text("x = 1\nprint(x)\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    (tmp_path / "readme.txt").write_text("not python")
    corpus = load_corpus(path=str(tmp_path), source="python", n_bytes=4096)
    text = corpus.decode("utf-8", errors="replace")
    assert "x = 1" in text
    assert "y = 2" in text
    assert "not python" not in text


def test_load_python_source_auto_discovers_stdlib():
    """load_corpus(source='python') without path finds Python stdlib files."""
    corpus = load_corpus(source="python", n_bytes=65536)
    assert len(corpus) > 0
    text = corpus.decode("utf-8", errors="replace")
    # stdlib contains 'def ', 'import ', 'class '
    assert "def " in text
    assert "import " in text


def test_load_python_respects_n_bytes_limit():
    """load_corpus truncates to n_bytes."""
    corpus = load_corpus(source="python", n_bytes=8192)
    assert len(corpus) <= 8192


def test_load_python_corpus_usable_in_bytedataset():
    """Python corpus produces a valid ByteDataset for training."""
    corpus = load_corpus(source="python", n_bytes=16384)
    ds = ByteDataset(corpus, seq_len=64)
    import random
    rng = random.Random(0)
    x, y = ds.sample(rng)
    assert len(x) == 64
    assert len(y) == 64
    assert all(0 <= t <= 255 for t in x)


def test_load_python_empty_directory_raises(tmp_path: Path):
    """load_corpus raises if the directory has no .py files."""
    (tmp_path / "notes.txt").write_text("no python here")
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        load_corpus(path=str(tmp_path), source="python", n_bytes=4096)


# ---------------------------------------------------------------------------
# Demo CLI (import-only smoke test — full run covered by bench tests)
# ---------------------------------------------------------------------------

def test_demo_module_importable():
    """sgjm.demo package is importable."""
    import importlib
    mod = importlib.import_module("sgjm.demo")
    assert mod is not None


def test_demo_completion_returns_string():
    """generate_completion returns a non-empty string for a tiny model."""
    pytest.importorskip("mlx.core", reason="MLX not available")
    import mlx.core as mx
    from sgjm.demo.generate import generate_completion
    from sgjm.training.config import TrainingConfig
    from sgjm.training.mlx_backend.model import SGJM

    cfg = TrainingConfig.smoke()
    model = SGJM(cfg.model)
    mx.eval(model.parameters())
    result = generate_completion(model, cfg.model, prompt=b"def f", n_tokens=8)
    assert isinstance(result, bytes)
    assert len(result) >= 8
