from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentCard:
    name: str
    hypothesis: str
    overrides: dict[str, Any]
    expected_signal: str = ""
    arch: str = "sgjm"
    pair_with_baseline: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SweepResult:
    card: ExperimentCard
    elapsed_sec: float
    sgjm_metrics: dict | None
    baseline_metrics: dict | None
    comparison: dict | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "card": self.card.to_dict(),
            "elapsed_sec": self.elapsed_sec,
            "sgjm_metrics": self.sgjm_metrics,
            "baseline_metrics": self.baseline_metrics,
            "comparison": self.comparison,
            "error": self.error,
        }

    @property
    def primary_score(self) -> float:
        if self.error or not self.sgjm_metrics:
            return float("-inf")
        # Composite: high acceptance, JEPA above chance, low NLL.
        accept = self.sgjm_metrics.get("branch_acceptance_rate", 0.0)
        jepa = self.sgjm_metrics.get("jepa_top1_acc", 0.0)
        chance = self.sgjm_metrics.get("jepa_chance_top1", 0.0)
        nll = self.sgjm_metrics.get("token_nll", float("inf"))
        return accept + (jepa - chance) - 0.1 * nll
