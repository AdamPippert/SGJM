from __future__ import annotations

import torch
import torch.nn.functional as F

from sgjm.training.config import TrainingConfig
from sgjm.training.torch_backend.model import SGJM


def _block_targets(y: torch.Tensor, block: int) -> torch.Tensor:
    # Build (B, valid_T, block) where slot t holds y[t : t+block].
    B, T = y.shape
    valid = T - block
    if valid <= 0:
        return y.new_empty((B, 0, block))
    idx = torch.arange(valid, device=y.device).unsqueeze(1) + torch.arange(block, device=y.device)
    return y[:, idx]


def compute_losses(
    model: SGJM,
    batch: tuple[torch.Tensor, torch.Tensor],
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x, y = batch
    hidden, logits = model.backbone(x)
    V = logits.size(-1)

    token_loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    block = cfg.model.block_size
    valid = hidden.size(1) - block
    zero = token_loss.new_zeros(())
    if valid <= 0:
        drafter_loss = jepa_loss = verifier_loss = zero
        accept_acc = zero
    else:
        parent_hidden = hidden[:, :valid]
        future_hidden = hidden[:, block - 1 : block - 1 + valid].detach()

        draft_logits, draft_latents = model.drafter(parent_hidden)
        draft_targets = _block_targets(y, block)
        drafter_loss = F.cross_entropy(
            draft_logits.reshape(-1, V),
            draft_targets.reshape(-1),
        )

        # Predict the final-block latent (drafter's trajectory endpoint should
        # align with the actual future hidden state).
        drafter_endpoint = draft_latents[:, :, -1]
        jepa_pred = model.judge(parent_hidden)
        jepa_loss = 0.5 * (
            F.mse_loss(jepa_pred, future_hidden)
            + F.mse_loss(drafter_endpoint, future_hidden)
        )

        pos = future_hidden
        neg = future_hidden.roll(shifts=1, dims=0)
        v_pos = model.verifier(parent_hidden, pos)
        v_neg = model.verifier(parent_hidden, neg)
        verifier_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(v_pos, torch.ones_like(v_pos))
            + F.binary_cross_entropy_with_logits(v_neg, torch.zeros_like(v_neg))
        )
        accept_acc = ((v_pos > 0).float().mean() + (v_neg <= 0).float().mean()) * 0.5

    total = (
        cfg.loss.token * token_loss
        + cfg.loss.drafter * drafter_loss
        + cfg.loss.jepa * jepa_loss
        + cfg.loss.verifier * verifier_loss
    )
    parts = {
        "total": total.detach(),
        "token": token_loss.detach(),
        "drafter": drafter_loss.detach(),
        "jepa": jepa_loss.detach(),
        "verifier": verifier_loss.detach(),
        "accept_acc": accept_acc.detach(),
    }
    return total, parts
