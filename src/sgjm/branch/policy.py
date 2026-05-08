from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from sgjm.graph.address import Signature


@dataclass(frozen=True)
class ScoredCandidate:
    tokens: tuple[int, ...]
    latent: tuple[float, ...]
    signature: Signature
    draft_score: float
    judge_score: float

    @property
    def combined(self) -> float:
        return self.draft_score + self.judge_score


JudgeFn = Callable[[Sequence[int], Sequence[float]], float]


@dataclass
class BranchPolicy:
    keep_top_k: int = 4
    min_combined: float = -1e9
    judge_weight: float = 1.0

    def rank(self, candidates: Sequence[ScoredCandidate]) -> tuple[ScoredCandidate, ...]:
        scored = [
            ScoredCandidate(
                tokens=c.tokens,
                latent=c.latent,
                signature=c.signature,
                draft_score=c.draft_score,
                judge_score=self.judge_weight * c.judge_score,
            )
            for c in candidates
        ]
        scored.sort(key=lambda c: c.combined, reverse=True)
        kept = [c for c in scored if c.combined >= self.min_combined]
        return tuple(kept[: self.keep_top_k])
