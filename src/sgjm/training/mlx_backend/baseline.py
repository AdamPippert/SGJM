from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from sgjm.training.config import ModelConfig
from sgjm.training.mlx_backend.model import Backbone


class BaselineLM(nn.Module):
    """Pure decoder-only LM at the same parameter budget as the full SGJM model.

    Uses baseline_n_layers instead of n_layers so the backbone alone consumes
    the parameter budget that SGJM splits across backbone/drafter/judge/verifier.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        backbone_cfg = replace(cfg, n_layers=cfg.baseline_n_layers)
        self.cfg = backbone_cfg
        self.backbone = Backbone(backbone_cfg)

    def __call__(self, idx: mx.array) -> tuple[mx.array, mx.array]:
        return self.backbone(idx)

    def num_parameters(self) -> int:
        return sum(int(v.size) for _, v in tree_flatten(self.parameters()))
