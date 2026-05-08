from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from sgjm.graph.address import Address, Signature


class NodeStatus(str, Enum):
    DRAFT = "draft"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMMITTED = "committed"
    MERGED = "merged"


@dataclass
class Node:
    address: Address
    parents: tuple[Address, ...]
    tokens: tuple[int, ...]
    signature: Signature
    latent: tuple[float, ...] = ()
    status: NodeStatus = NodeStatus.DRAFT
    score: float = 0.0
    depth: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return not self.parents

    @property
    def is_terminal(self) -> bool:
        return self.status in (NodeStatus.COMMITTED, NodeStatus.REJECTED)
