from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgjm.training.config import ModelConfig, TrainingConfig
from sgjm.training.torch_backend.model import Backbone


class BaselineLM(nn.Module):
    """Pure decoder-only LM at the same param budget as the full SGJM model.

    Sized through baseline_n_layers in ModelConfig so backbone alone consumes
    the parameter budget the SGJM splits across backbone/drafter/judge/verifier.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        backbone_cfg = replace(cfg, n_layers=cfg.baseline_n_layers)
        self.cfg = backbone_cfg
        self.backbone = Backbone(backbone_cfg)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(idx)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_baseline_losses(
    model: BaselineLM,
    batch: tuple[torch.Tensor, torch.Tensor],
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x, y = batch
    _, logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return loss, {"total": loss.detach(), "token": loss.detach()}
