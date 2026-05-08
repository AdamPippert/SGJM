from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class BackboneState:
    tokens: tuple[int, ...]
    latent: tuple[float, ...]


@runtime_checkable
class Backbone(Protocol):
    latent_dim: int

    def encode(self, tokens: Sequence[int]) -> BackboneState: ...

    def step(self, state: BackboneState, token: int) -> BackboneState: ...


@dataclass
class StubBackbone:
    latent_dim: int = 16
    seed: int = 0

    def encode(self, tokens: Sequence[int]) -> BackboneState:
        latent = self._latent_for(tokens)
        return BackboneState(tokens=tuple(tokens), latent=latent)

    def step(self, state: BackboneState, token: int) -> BackboneState:
        new_tokens = state.tokens + (token,)
        return BackboneState(tokens=new_tokens, latent=self._latent_for(new_tokens))

    def _latent_for(self, tokens: Sequence[int]) -> tuple[float, ...]:
        acc = [0.0] * self.latent_dim
        for i, t in enumerate(tokens):
            for d in range(self.latent_dim):
                acc[d] += math.sin((self.seed + 1) * (i + 1) * (d + 1) * (int(t) + 1) * 0.017)
        norm = math.sqrt(sum(x * x for x in acc)) or 1.0
        return tuple(x / norm for x in acc)
