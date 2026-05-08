from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from sgjm.modules.backbone import Backbone, BackboneState


@dataclass(frozen=True)
class DraftSample:
    tokens: tuple[int, ...]
    latent: tuple[float, ...]
    log_prob: float


@runtime_checkable
class Drafter(Protocol):
    def draft(self, state: BackboneState, *, k: int, block: int) -> tuple[DraftSample, ...]: ...


@dataclass
class StubDrafter:
    backbone: Backbone
    vocab_size: int = 32
    seed: int = 1

    def draft(self, state: BackboneState, *, k: int, block: int) -> tuple[DraftSample, ...]:
        rng = random.Random(hash((self.seed, tuple(state.tokens), k, block)))
        samples: list[DraftSample] = []
        for _ in range(k):
            cur = state
            tokens: list[int] = []
            log_prob = 0.0
            for _ in range(block):
                tok = rng.randrange(self.vocab_size)
                cur = self.backbone.step(cur, tok)
                tokens.append(tok)
                log_prob += -math.log(self.vocab_size)
            samples.append(
                DraftSample(
                    tokens=tuple(tokens),
                    latent=tuple(cur.latent),
                    log_prob=log_prob,
                )
            )
        return tuple(samples)
