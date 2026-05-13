from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from sgjm.training.config import TrainingConfig
from sgjm.training.torch_backend.baseline import BaselineLM
from sgjm.training.torch_backend.model import SGJM


@dataclass
class LoadedCheckpoint:
    model: nn.Module
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
