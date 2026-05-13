from __future__ import annotations

import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def _cache_dir() -> Path:
    base = os.environ.get("SGJM_CACHE_DIR")
    if base:
        path = Path(base)
    else:
        path = Path.home() / ".cache" / "sgjm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_tinyshakespeare(force: bool = False) -> Path:
    target = _cache_dir() / "tinyshakespeare.txt"
    if target.exists() and not force:
        return target
    try:
        with urllib.request.urlopen(TINYSHAKESPEARE_URL, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(
            f"failed to download tinyshakespeare from {TINYSHAKESPEARE_URL}: {e}. "
            "Pre-download it manually and pass --data-path."
        ) from e
    target.write_bytes(data)
    return target


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


def load_corpus(
    path: str | None = None,
    n_bytes: int = 1 << 20,
    seed: int = 0,
    source: str = "auto",
) -> bytes:
    """Load a byte corpus from `path`, a known source, or synthetic fallback.

    source:
      "auto"            — use path if given, else synthetic
      "synthetic"       — always synthetic, ignore path
      "tinyshakespeare" — download (and cache) Karpathy's tinyshakespeare
      "file"            — require path; raise if missing
    """
    if source == "synthetic":
        return synthetic_corpus(n_bytes, seed)
    if source == "tinyshakespeare":
        target = download_tinyshakespeare()
        return target.read_bytes()
    if source == "file":
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"data file not found: {path!r}")
        with open(path, "rb") as f:
            return f.read()
    # auto
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
