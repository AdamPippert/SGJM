from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sgjm.branch.policy import ScoredCandidate


@dataclass(frozen=True)
class VerifierDecision:
    accepted: tuple[ScoredCandidate, ...]
    rejected: tuple[ScoredCandidate, ...]

    @property
    def acceptance_rate(self) -> float:
        total = len(self.accepted) + len(self.rejected)
        return 0.0 if total == 0 else len(self.accepted) / total


@dataclass
class VerifierStub:
    accept_threshold: float = 0.0

    def verify(self, candidates: Sequence[ScoredCandidate]) -> VerifierDecision:
        accepted = tuple(c for c in candidates if c.combined >= self.accept_threshold)
        rejected = tuple(c for c in candidates if c.combined < self.accept_threshold)
        return VerifierDecision(accepted=accepted, rejected=rejected)
