from __future__ import annotations

import os
import random
from dataclasses import dataclass


def synthetic_corpus(n_bytes: int = 1 << 20, seed: int = 0) -> bytes:
    # Deterministic byte stream with both periodic structure and a 2nd-order
    # Markov component. Gives the model something learnable but non-trivial
    # without pulling external data.
    rng = random.Random(seed)
    pattern = bytes(rng.randrange(256) for _ in range(96))
    period = len(pattern)
    out = bytearray(n_bytes)
    prev1 = 0
    prev2 = 0
    for i in range(n_bytes):
        mix = (pattern[i % period] ^ ((prev1 * 17 + prev2 * 31 + i * 7) & 0xFF))
        out[i] = mix & 0xFF
        prev2 = prev1
        prev1 = out[i]
    return bytes(out)


def load_corpus(path: str | None = None, n_bytes: int = 1 << 20, seed: int = 0) -> bytes:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return synthetic_corpus(n_bytes, seed)


@dataclass
class ByteDataset:
    data: bytes
    seq_len: int

    def __post_init__(self) -> None:
        if len(self.data) <= self.seq_len + 1:
            raise ValueError(
                f"corpus too small ({len(self.data)} bytes) for seq_len={self.seq_len}"
            )

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 1

    def sample(self, rng: random.Random) -> tuple[list[int], list[int]]:
        i = rng.randrange(len(self))
        chunk = self.data[i : i + self.seq_len + 1]
        return list(chunk[:-1]), list(chunk[1:])

    def batch(
        self,
        batch_size: int,
        rng: random.Random,
    ) -> tuple[list[list[int]], list[list[int]]]:
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        for _ in range(batch_size):
            x, y = self.sample(rng)
            xs.append(x)
            ys.append(y)
        return xs, ys
