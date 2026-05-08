from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    def score(self, parent_latent: Sequence[float], child_latent: Sequence[float]) -> float: ...


@dataclass
class StubJudge:
    # Negative L2 distance in latent space: closer transitions get higher
    # scores. Real implementation will be a learned JEPA predictor.
    def score(self, parent_latent: Sequence[float], child_latent: Sequence[float]) -> float:
        if len(parent_latent) != len(child_latent):
            raise ValueError("latent dim mismatch")
        sq = sum((a - b) ** 2 for a, b in zip(parent_latent, child_latent))
        return -math.sqrt(sq)
