from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Address:
    id: int

    def __str__(self) -> str:
        return f"a{self.id}"


@dataclass(frozen=True)
class Signature:
    digest: bytes

    @classmethod
    def from_latent(cls, latent: Sequence[float], bits: int = 64) -> "Signature":
        # SimHash-style locality-sensitive digest. Two latents with small L2
        # distance produce digests with small Hamming distance, which the
        # AddressBook uses to merge near-duplicate branches.
        if bits % 8:
            raise ValueError("bits must be a multiple of 8")
        rng_seed = b"sgjm/sig/v1"
        acc = [0.0] * bits
        for i, x in enumerate(latent):
            seed = hashlib.blake2b(rng_seed + i.to_bytes(8, "little"), digest_size=bits // 8).digest()
            for b in range(bits):
                bit = (seed[b // 8] >> (b % 8)) & 1
                acc[b] += x if bit else -x
        out = bytearray(bits // 8)
        for b in range(bits):
            if acc[b] >= 0.0:
                out[b // 8] |= 1 << (b % 8)
        return cls(bytes(out))

    @classmethod
    def from_tokens(cls, tokens: Iterable[int]) -> "Signature":
        h = hashlib.blake2b(digest_size=8)
        for t in tokens:
            h.update(int(t).to_bytes(8, "little", signed=True))
        return cls(h.digest())

    def hamming(self, other: "Signature") -> int:
        if len(self.digest) != len(other.digest):
            raise ValueError("signature length mismatch")
        return sum((a ^ b).bit_count() for a, b in zip(self.digest, other.digest))


@dataclass
class AddressBook:
    merge_radius: int = 4
    _next_id: int = 0
    _by_signature: dict[Signature, Address] = field(default_factory=dict)

    def allocate(self) -> Address:
        addr = Address(self._next_id)
        self._next_id += 1
        return addr

    def lookup(self, sig: Signature) -> Address | None:
        if sig in self._by_signature:
            return self._by_signature[sig]
        if self.merge_radius <= 0:
            return None
        for known, addr in self._by_signature.items():
            if known.hamming(sig) <= self.merge_radius:
                return addr
        return None

    def bind(self, sig: Signature, addr: Address) -> None:
        self._by_signature[sig] = addr

    def resolve_or_allocate(self, sig: Signature) -> tuple[Address, bool]:
        existing = self.lookup(sig)
        if existing is not None:
            return existing, False
        addr = self.allocate()
        self.bind(sig, addr)
        return addr, True
