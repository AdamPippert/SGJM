from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from sgjm.training.config import TrainingConfig
from sgjm.training.torch_backend.baseline import BaselineLM
from sgjm.training.torch_backend.model import SGJM


@dataclass
class LoadedCheckpoint:
    model: object  # nn.Module (torch) or mlx nn.Module
    config: TrainingConfig
    arch: str
    step: int
    path: Path


def load_checkpoint(path: str | Path, device: str = "cpu") -> LoadedCheckpoint:
    p = Path(path)
    ckpt = torch.load(p, map_location=device, weights_only=False)
    cfg = TrainingConfig.from_dict(ckpt["config"])
    arch = ckpt.get("arch", cfg.arch)
    if arch == "sgjm":
        model: nn.Module = SGJM(cfg.model)
    elif arch == "baseline":
        model = BaselineLM(cfg.model)
    else:
        raise ValueError(f"unknown arch in checkpoint {p}: {arch!r}")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return LoadedCheckpoint(model=model, config=cfg, arch=arch, step=int(ckpt.get("step", -1)), path=p)


def load_mlx_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load an MLX .safetensors checkpoint with companion .meta.json."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    p = Path(path)
    weights = mx.load(str(p))
    meta_path = p.parent / (p.stem + ".meta.json")
    meta = json.loads(meta_path.read_text())
    cfg = TrainingConfig.from_dict(meta["config"])
    arch = cfg.arch

    if arch == "sgjm":
        from sgjm.training.mlx_backend.model import SGJM as MlxSGJM
        model: object = MlxSGJM(cfg.model)
    elif arch == "baseline":
        from sgjm.training.mlx_backend.baseline import BaselineLM as MlxBaselineLM
        model = MlxBaselineLM(cfg.model)
    else:
        raise ValueError(f"unknown arch {arch!r}")

    import mlx.nn as mlx_nn
    assert isinstance(model, mlx_nn.Module)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return LoadedCheckpoint(
        model=model,
        config=cfg,
        arch=arch,
        step=int(meta.get("step", -1)),
        path=p,
    )
