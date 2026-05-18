from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from sgjm.training.config import TrainingConfig
from sgjm.training.mlx_backend.model import SGJM


def _block_targets(y: mx.array, block: int) -> mx.array:
    B, T = y.shape
    valid = T - block
    if valid <= 0:
        return mx.zeros((B, 0, block), dtype=y.dtype)
    base = mx.arange(valid)[:, None]
    offset = mx.arange(block)[None, :]
    idx = base + offset
    return y[:, idx]


def compute_losses(
    model: SGJM,
    x: mx.array,
    y: mx.array,
    cfg: TrainingConfig,
) -> tuple[mx.array, dict[str, mx.array]]:
    hidden, logits = model.backbone(x)
    V = logits.shape[-1]

    token_loss = nn.losses.cross_entropy(
        logits.reshape(-1, V), y.reshape(-1), reduction="mean"
    )

    block = cfg.model.block_size
    valid = hidden.shape[1] - block
    zero = mx.zeros((), dtype=token_loss.dtype)
    if valid <= 0:
        drafter_loss = jepa_loss = verifier_loss = zero
        accept_acc = zero
    else:
        parent_hidden = hidden[:, :valid]
        future_hidden = mx.stop_gradient(hidden[:, block - 1 : block - 1 + valid])

        draft_logits, draft_latents = model.drafter(parent_hidden)
        draft_targets = _block_targets(y, block)
        drafter_loss = nn.losses.cross_entropy(
            draft_logits.reshape(-1, V),
            draft_targets.reshape(-1),
            reduction="mean",
        )

        drafter_endpoint = draft_latents[:, :, -1]
        jepa_pred = model.judge(parent_hidden)
        jepa_loss = 0.5 * (
            nn.losses.mse_loss(jepa_pred, future_hidden, reduction="mean")
            + nn.losses.mse_loss(drafter_endpoint, future_hidden, reduction="mean")
        )

        pos = future_hidden
        # Roll along the sequence axis so negatives are genuinely distinct even
        # when batch_size=1. Rolling on axis=0 (batch) returns the identity at B=1,
        # causing the verifier to receive contradictory zero-net gradients.
        neg = mx.concatenate([future_hidden[:, -1:], future_hidden[:, :-1]], axis=1)
        v_pos = model.verifier(parent_hidden, pos)
        v_neg = model.verifier(parent_hidden, neg)
        verifier_loss = 0.5 * (
            nn.losses.binary_cross_entropy(
                v_pos, mx.ones_like(v_pos), with_logits=True, reduction="mean"
            )
            + nn.losses.binary_cross_entropy(
                v_neg, mx.zeros_like(v_neg), with_logits=True, reduction="mean"
            )
        )
        accept_acc = 0.5 * (
            (v_pos > 0).astype(mx.float32).mean()
            + (v_neg <= 0).astype(mx.float32).mean()
        )

    total = (
        cfg.loss.token * token_loss
        + cfg.loss.drafter * drafter_loss
        + cfg.loss.jepa * jepa_loss
        + cfg.loss.verifier * verifier_loss
    )
    parts = {
        "total": total,
        "token": token_loss,
        "drafter": drafter_loss,
        "jepa": jepa_loss,
        "verifier": verifier_loss,
        "accept_acc": accept_acc,
    }
    return total, parts
