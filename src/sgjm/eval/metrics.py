from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from sgjm.training.config import TrainingConfig
from sgjm.training.data import ByteDataset
from sgjm.training.torch_backend.baseline import BaselineLM
from sgjm.training.torch_backend.model import SGJM


@dataclass
class SGJMEvalMetrics:
    n_tokens: int
    n_positions: int
    token_nll: float
    token_ppl: float
    branch_acceptance_rate: float
    jepa_top1_acc: float
    jepa_chance_top1: float
    merge_precision_js: float
    random_pair_js: float
    merge_precision_advantage: float
    compute_per_accepted_token: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BaselineEvalMetrics:
    n_tokens: int
    token_nll: float
    token_ppl: float
    compute_per_token: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonReport:
    sgjm: SGJMEvalMetrics
    baseline: BaselineEvalMetrics
    nll_delta: float
    compute_advantage: float
    gate_passed: bool
    gate_reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "sgjm": self.sgjm.to_dict(),
            "baseline": self.baseline.to_dict(),
            "nll_delta": self.nll_delta,
            "compute_advantage": self.compute_advantage,
            "gate_passed": self.gate_passed,
            "gate_reasons": self.gate_reasons,
        }


def _iter_batches(
    cfg: TrainingConfig,
    dataset: ByteDataset,
    n_batches: int,
    seed: int,
    device: str,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    for _ in range(n_batches):
        xs, ys = dataset.batch(cfg.optim.batch_size, rng)
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        yield x, y


def _module_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _simhash_signs(latents: torch.Tensor, n_bits: int, seed: int) -> torch.Tensor:
    # (N, D) -> (N, n_bits) bool
    gen = torch.Generator(device=latents.device).manual_seed(seed)
    proj = torch.randn(latents.size(-1), n_bits, generator=gen, device=latents.device)
    return (latents @ proj) >= 0


@torch.no_grad()
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
    device: str = "cpu",
    seed: int = 1234,
) -> SGJMEvalMetrics:
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

    for batch_idx, (x, y) in enumerate(_iter_batches(cfg, dataset, n_batches, seed, device)):
        hidden, logits = model.backbone(x)
        B, T, D = hidden.shape
        V = logits.size(-1)
        valid = T - block
        if valid <= 0:
            continue

        nll = F.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1), reduction="sum"
        )
        nll_total += float(nll)
        tokens_total += int(y.numel())

        parent_hidden = hidden[:, :valid]
        future_hidden = hidden[:, block - 1 : block - 1 + valid]
        future_logits = logits[:, block - 1 : block - 1 + valid]

        n_pos = B * valid
        positions_total += n_pos
        parent_flat = parent_hidden.reshape(n_pos, D)
        future_flat = future_hidden.reshape(n_pos, D)
        future_logit_flat = future_logits.reshape(n_pos, V)

        # 1) Branch acceptance: drafter produces draft latents per position;
        #    verifier scores (parent, draft_endpoint).
        _, draft_latents = model.drafter(parent_hidden)
        draft_endpoint = draft_latents[:, :, -1]
        v_drafts = model.verifier(parent_hidden, draft_endpoint)
        accept_total += float((v_drafts > 0).float().sum())
        accept_n += int(v_drafts.numel())

        # 2) JEPA top-1 ranking: judge prediction vs (true + K distractors).
        judge_pred = model.judge(parent_hidden).reshape(n_pos, D)
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
        distractor_idx = torch.randint(
            0, n_pos, (n_pos, n_distractors), generator=gen, device=device
        )
        distractor_futures = future_flat[distractor_idx]
        true_score = -((judge_pred - future_flat) ** 2).mean(dim=-1)
        distractor_scores = -((judge_pred.unsqueeze(1) - distractor_futures) ** 2).mean(dim=-1)
        top1 = (true_score.unsqueeze(1) > distractor_scores).all(dim=1).float()
        rank_total += float(top1.sum())
        rank_n += int(top1.numel())

        # 3) Merge precision: SimHash on draft_endpoint latents; for sampled
        #    pairs within merge_radius, measure JS divergence of true
        #    next-token distributions. Compare against pairs outside the radius.
        draft_flat = draft_endpoint.reshape(n_pos, D)
        signs = _simhash_signs(draft_flat, n_bits=64, seed=42)
        pair_gen = torch.Generator(device=device).manual_seed(seed + 1000 + batch_idx)
        i_idx = torch.randint(0, n_pos, (n_merge_pairs,), generator=pair_gen, device=device)
        j_idx = torch.randint(0, n_pos, (n_merge_pairs,), generator=pair_gen, device=device)
        valid_pair = i_idx != j_idx
        if valid_pair.any():
            i_idx = i_idx[valid_pair]
            j_idx = j_idx[valid_pair]
            hamming = (signs[i_idx] ^ signs[j_idx]).float().sum(dim=-1)
            would_merge = hamming <= merge_radius_bits
            p = F.softmax(future_logit_flat[i_idx], dim=-1)
            q = F.softmax(future_logit_flat[j_idx], dim=-1)
            m = 0.5 * (p + q)
            eps = 1e-12
            kl_pm = (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
            kl_qm = (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
            js = 0.5 * (kl_pm + kl_qm)
            if would_merge.any():
                merge_js_sum += float(js[would_merge].sum())
                merge_js_n += int(would_merge.sum())
            non_merge = ~would_merge
            if non_merge.any():
                rand_js_sum += float(js[non_merge].sum())
                rand_js_n += int(non_merge.sum())

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

    backbone_p = _module_params(model.backbone)
    drafter_p = _module_params(model.drafter)
    verifier_p = _module_params(model.verifier)
    accepted_per_step = max(branch_accept, 1e-6) * drafts_per_step * block
    cost_per_step = backbone_p + drafts_per_step * (drafter_p + verifier_p)
    compute_per_accepted = cost_per_step / accepted_per_step

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


@torch.no_grad()
def evaluate_baseline(
    model: BaselineLM,
    cfg: TrainingConfig,
    dataset: ByteDataset,
    n_batches: int = 32,
    *,
    device: str = "cpu",
    seed: int = 1234,
) -> BaselineEvalMetrics:
    model.eval()
    nll_total = 0.0
    tokens_total = 0
    for x, y in _iter_batches(cfg, dataset, n_batches, seed, device):
        _, logits = model(x)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        nll_total += float(nll)
        tokens_total += int(y.numel())
    nll = nll_total / tokens_total
    compute = float(_module_params(model.backbone))
    return BaselineEvalMetrics(
        n_tokens=tokens_total,
        token_nll=nll,
        token_ppl=math.exp(nll),
        compute_per_token=compute,
    )


def compare(
    sgjm: SGJMEvalMetrics,
    baseline: BaselineEvalMetrics,
    *,
    require_lower_or_equal_nll: bool = True,
    require_branch_accept_above: float = 0.5,
    require_jepa_above_chance: bool = True,
    require_merge_advantage: float = 1.5,
    require_compute_advantage: float = 1.0,
) -> ComparisonReport:
    nll_delta = sgjm.token_nll - baseline.token_nll
    compute_advantage = baseline.compute_per_token / max(sgjm.compute_per_accepted_token, 1e-9)
    reasons: list[str] = []
    if require_lower_or_equal_nll and nll_delta > 0.05:
        reasons.append(f"sgjm_token_nll-{nll_delta:.3f} > baseline+0.05")
    if sgjm.branch_acceptance_rate < require_branch_accept_above:
        reasons.append(
            f"branch_acceptance_rate={sgjm.branch_acceptance_rate:.3f}"
            f" < {require_branch_accept_above}"
        )
    if require_jepa_above_chance and sgjm.jepa_top1_acc <= sgjm.jepa_chance_top1 + 0.05:
        reasons.append(
            f"jepa_top1_acc={sgjm.jepa_top1_acc:.3f}"
            f" not meaningfully above chance={sgjm.jepa_chance_top1:.3f}"
        )
    if sgjm.merge_precision_advantage < require_merge_advantage:
        reasons.append(
            f"merge_precision_advantage={sgjm.merge_precision_advantage:.2f}"
            f" < {require_merge_advantage}"
        )
    if compute_advantage < require_compute_advantage:
        reasons.append(
            f"compute_advantage={compute_advantage:.2f} < {require_compute_advantage}"
        )
    return ComparisonReport(
        sgjm=sgjm,
        baseline=baseline,
        nll_delta=nll_delta,
        compute_advantage=compute_advantage,
        gate_passed=len(reasons) == 0,
        gate_reasons=reasons,
    )
