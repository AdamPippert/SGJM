from __future__ import annotations

import math
import random
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from sgjm.eval.metrics import BaselineEvalMetrics, SGJMEvalMetrics
from sgjm.training.config import TrainingConfig
from sgjm.training.data import ByteDataset
from sgjm.training.mlx_backend.baseline import BaselineLM
from sgjm.training.mlx_backend.model import SGJM


def _iter_batches_mlx(
    cfg: TrainingConfig,
    dataset: ByteDataset,
    n_batches: int,
    seed: int,
) -> Iterable[tuple[mx.array, mx.array]]:
    rng = random.Random(seed)
    for _ in range(n_batches):
        xs, ys = dataset.batch(cfg.optim.batch_size, rng)
        yield mx.array(xs, dtype=mx.int32), mx.array(ys, dtype=mx.int32)


def _module_params_mlx(module: nn.Module) -> int:
    return sum(int(v.size) for _, v in tree_flatten(module.parameters()))


def _simhash_signs_mlx(latents: mx.array, n_bits: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    proj = mx.random.normal((latents.shape[-1], n_bits))
    return (latents @ proj) >= 0


def evaluate_sgjm(
    model: SGJM,
    cfg: TrainingConfig,
    dataset: ByteDataset,
    n_batches: int = 32,
    *,
    n_distractors: int = 8,
    n_merge_pairs: int = 4096,
    merge_radius_bits: int = 6,
    drafts_per_step: int = 4,
    seed: int = 1234,
) -> SGJMEvalMetrics:
    """Evaluate an MLX SGJM model and return structured metrics."""
    model.eval()
    block = cfg.model.block_size

    tokens_total = 0
    nll_total = 0.0
    accept_total = 0.0
    accept_n = 0
    rank_total = 0.0
    rank_n = 0
    merge_js_sum = 0.0
    merge_js_n = 0
    rand_js_sum = 0.0
    rand_js_n = 0
    positions_total = 0

    for batch_idx, (x, y) in enumerate(_iter_batches_mlx(cfg, dataset, n_batches, seed)):
        hidden, logits = model.backbone(x)
        mx.eval(hidden, logits)
        B, T, D = hidden.shape
        V = logits.shape[-1]
        valid = T - block
        if valid <= 0:
            continue

        # Token NLL (sum over all tokens)
        n_tok = int(y.size)
        loss_mean = nn.losses.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1), reduction="mean"
        )
        mx.eval(loss_mean)
        nll_total += float(loss_mean) * n_tok
        tokens_total += n_tok

        parent_hidden = hidden[:, :valid]
        future_hidden = hidden[:, block - 1 : block - 1 + valid]
        future_logits = logits[:, block - 1 : block - 1 + valid]

        n_pos = B * valid
        positions_total += n_pos
        parent_flat = parent_hidden.reshape(n_pos, D)
        future_flat = future_hidden.reshape(n_pos, D)
        future_logit_flat = future_logits.reshape(n_pos, V)
        mx.eval(parent_flat, future_flat, future_logit_flat)

        # 1) Branch acceptance: drafter produces draft latents per position;
        #    verifier scores (parent, draft_endpoint).
        _, draft_latents = model.drafter(parent_hidden)
        draft_endpoint = draft_latents[:, :, -1]
        v_drafts = model.verifier(parent_hidden, draft_endpoint)
        mx.eval(v_drafts)
        accept_total += float((v_drafts > 0).astype(mx.float32).sum())
        accept_n += int(v_drafts.size)

        # 2) JEPA top-1 ranking: judge prediction vs (true + K distractors).
        judge_pred = model.judge(parent_hidden).reshape(n_pos, D)
        mx.eval(judge_pred)

        mx.random.seed(seed + batch_idx)
        distractor_idx = mx.random.randint(0, n_pos, (n_pos, n_distractors))
        mx.eval(distractor_idx)
        distractor_futures = future_flat[distractor_idx]
        true_score = -((judge_pred - future_flat) ** 2).mean(axis=-1)
        distractor_scores = -((judge_pred[:, None, :] - distractor_futures) ** 2).mean(axis=-1)
        top1 = (true_score[:, None] > distractor_scores).all(axis=1).astype(mx.float32)
        mx.eval(top1)
        rank_total += float(top1.sum())
        rank_n += int(top1.size)

        # 3) Merge precision: SimHash on draft_endpoint latents; for sampled
        #    pairs within merge_radius, measure JS divergence of true
        #    next-token distributions. Compare against pairs outside the radius.
        draft_flat = draft_endpoint.reshape(n_pos, D)
        signs = _simhash_signs_mlx(draft_flat, n_bits=64, seed=42)
        mx.eval(signs)

        mx.random.seed(seed + 1000 + batch_idx)
        i_idx = mx.random.randint(0, n_pos, (n_merge_pairs,))
        j_idx = mx.random.randint(0, n_pos, (n_merge_pairs,))
        mx.eval(i_idx, j_idx)

        valid_pair_mask = (i_idx != j_idx)
        mx.eval(valid_pair_mask)
        valid_pair_np = valid_pair_mask.tolist()
        if any(valid_pair_np):
            vi = mx.array([k for k, ok in enumerate(valid_pair_np) if ok], dtype=mx.int32)
            i_idx_v = i_idx[vi]
            j_idx_v = j_idx[vi]
            mx.eval(i_idx_v, j_idx_v)

            hamming = mx.sum(
                (signs[i_idx_v] ^ signs[j_idx_v]).astype(mx.float32), axis=-1
            )
            would_merge = hamming <= merge_radius_bits
            mx.eval(would_merge)

            p = mx.softmax(future_logit_flat[i_idx_v], axis=-1)
            q = mx.softmax(future_logit_flat[j_idx_v], axis=-1)
            m_dist = 0.5 * (p + q)
            eps = 1e-12
            # mx.clip requires (array, a_min, a_max); use mx.maximum for lower-bound only
            kl_pm = (
                p * (mx.log(mx.maximum(p, eps)) - mx.log(mx.maximum(m_dist, eps)))
            ).sum(axis=-1)
            kl_qm = (
                q * (mx.log(mx.maximum(q, eps)) - mx.log(mx.maximum(m_dist, eps)))
            ).sum(axis=-1)
            js = 0.5 * (kl_pm + kl_qm)
            mx.eval(js, would_merge)

            wm_list = would_merge.tolist()
            js_list = js.tolist()
            for ok, jv in zip(wm_list, js_list):
                if ok:
                    merge_js_sum += jv
                    merge_js_n += 1
                else:
                    rand_js_sum += jv
                    rand_js_n += 1

    if tokens_total == 0:
        raise RuntimeError("no tokens evaluated; dataset/config too small")

    token_nll = nll_total / tokens_total
    branch_accept = accept_total / max(1, accept_n)
    jepa_top1 = rank_total / max(1, rank_n)
    chance_top1 = 1.0 / (n_distractors + 1)
    if merge_js_n == 0 or rand_js_n == 0:
        merge_js = float("nan")
        rand_js = rand_js_sum / rand_js_n if rand_js_n else float("nan")
        merge_advantage = 1.0
    else:
        merge_js = merge_js_sum / merge_js_n
        rand_js = rand_js_sum / rand_js_n
        denom = max(merge_js, 1e-9)
        merge_advantage = rand_js / denom

    backbone_p = _module_params_mlx(model.backbone)
    drafter_p = _module_params_mlx(model.drafter)
    verifier_p = _module_params_mlx(model.verifier)
    accepted_per_step = max(branch_accept, 1e-6) * drafts_per_step * block
    cost_per_step = backbone_p + drafts_per_step * (drafter_p + verifier_p)
    compute_per_accepted = cost_per_step / accepted_per_step

    model.train()
    return SGJMEvalMetrics(
        n_tokens=tokens_total,
        n_positions=positions_total,
        token_nll=token_nll,
        token_ppl=math.exp(token_nll),
        branch_acceptance_rate=branch_accept,
        jepa_top1_acc=jepa_top1,
        jepa_chance_top1=chance_top1,
        merge_precision_js=merge_js,
        random_pair_js=rand_js,
        merge_precision_advantage=merge_advantage,
        compute_per_accepted_token=compute_per_accepted,
    )


def evaluate_baseline(
    model: BaselineLM,
    cfg: TrainingConfig,
    dataset: ByteDataset,
    n_batches: int = 32,
    *,
    seed: int = 1234,
) -> BaselineEvalMetrics:
    """Evaluate an MLX BaselineLM and return structured metrics."""
    model.eval()
    nll_total = 0.0
    tokens_total = 0
    for x, y in _iter_batches_mlx(cfg, dataset, n_batches, seed):
        _, logits = model(x)
        V = logits.shape[-1]
        n_tok = int(y.size)
        loss_mean = nn.losses.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1), reduction="mean"
        )
        mx.eval(loss_mean)
        nll_total += float(loss_mean) * n_tok
        tokens_total += n_tok

    nll = nll_total / tokens_total
    compute = float(_module_params_mlx(model.backbone))
    model.train()
    return BaselineEvalMetrics(
        n_tokens=tokens_total,
        token_nll=nll,
        token_ppl=math.exp(nll),
        compute_per_token=compute,
    )
