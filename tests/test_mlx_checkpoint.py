from __future__ import annotations

import json
from pathlib import Path

import pytest

mlx = pytest.importorskip("mlx.core")

from sgjm.training.config import TrainingConfig
from sgjm.training.mlx_backend.trainer import train


def _train_and_get_checkpoint(arch: str, tmp_path: Path) -> Path:
    cfg = TrainingConfig.smoke()
    cfg.arch = arch
    cfg.checkpoint_dir = str(tmp_path / arch)
    result = train(cfg, "mlx")
    assert result.checkpoint_path is not None
    return result.checkpoint_path


def test_load_mlx_sgjm_checkpoint(tmp_path):
    """load_mlx_checkpoint correctly reconstructs an SGJM model."""
    from sgjm.eval.checkpoint import load_mlx_checkpoint
    from sgjm.training.mlx_backend.model import SGJM

    ckpt_path = _train_and_get_checkpoint("sgjm", tmp_path)
    loaded = load_mlx_checkpoint(ckpt_path)

    assert loaded.arch == "sgjm"
    assert isinstance(loaded.model, SGJM)
    assert loaded.step >= 0
    assert loaded.path == ckpt_path


def test_load_mlx_baseline_checkpoint(tmp_path):
    """load_mlx_checkpoint correctly reconstructs a BaselineLM model."""
    from sgjm.eval.checkpoint import load_mlx_checkpoint
    from sgjm.training.mlx_backend.baseline import BaselineLM

    ckpt_path = _train_and_get_checkpoint("baseline", tmp_path)
    loaded = load_mlx_checkpoint(ckpt_path)

    assert loaded.arch == "baseline"
    assert isinstance(loaded.model, BaselineLM)
    assert loaded.step >= 0


def test_load_mlx_checkpoint_unknown_arch_raises(tmp_path):
    """load_mlx_checkpoint raises ValueError for unknown arch in meta."""
    import mlx.core as mx
    from sgjm.eval.checkpoint import load_mlx_checkpoint
    from sgjm.training.mlx_backend.model import SGJM

    cfg = TrainingConfig.smoke()
    cfg.arch = "sgjm"
    ckpt_dir = tmp_path / "bad"
    ckpt_dir.mkdir()
    # write a safetensors with a bad arch in meta
    weights_path = ckpt_dir / "final.safetensors"
    model = SGJM(cfg.model)
    from mlx.utils import tree_flatten
    mx.save_safetensors(str(weights_path), dict(tree_flatten(model.parameters())))
    meta = {"step": 0, "config": {**cfg.to_dict(), "arch": "unknown_arch"}}
    (ckpt_dir / "final.meta.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="unknown arch"):
        load_mlx_checkpoint(weights_path)
