from __future__ import annotations

from dataclasses import asdict, dataclass

# Backend-agnostic eval result types and comparison logic. Kept free of any
# torch/mlx import so that importing sgjm.eval (which every eval submodule
# pulls in via __init__.py) never forces a backend dependency that an
# MLX-only or torch-only install may not have.


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
