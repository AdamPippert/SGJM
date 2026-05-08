from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from sgjm.branch.policy import BranchPolicy, ScoredCandidate
from sgjm.branch.verifier import VerifierDecision, VerifierStub
from sgjm.graph.address import Address
from sgjm.graph.manager import GraphManager
from sgjm.graph.node import Node, NodeStatus


class BranchPhase(str, Enum):
    DRAFTED = "drafted"
    SCORED = "scored"
    PRUNED = "pruned"
    VERIFIED = "verified"
    COMMITTED = "committed"


@dataclass
class Branch:
    parent: Address
    candidate: ScoredCandidate
    address: Address | None = None
    phase: BranchPhase = BranchPhase.DRAFTED
    fresh: bool = True


@dataclass
class StepReport:
    parent: Address
    drafted: int
    pruned: int
    accepted: int
    merged: int
    committed_addresses: tuple[Address, ...]


@dataclass
class BranchLifecycle:
    graph: GraphManager
    policy: BranchPolicy = field(default_factory=BranchPolicy)
    verifier: VerifierStub = field(default_factory=VerifierStub)

    def step(
        self,
        parent: Address,
        candidates: Sequence[ScoredCandidate],
    ) -> StepReport:
        ranked = self.policy.rank(candidates)
        decision = self.verifier.verify(ranked)
        committed = self._commit(parent, decision)
        merged = sum(1 for b in committed if not b.fresh)
        return StepReport(
            parent=parent,
            drafted=len(candidates),
            pruned=len(candidates) - len(ranked),
            accepted=len(decision.accepted),
            merged=merged,
            committed_addresses=tuple(b.address for b in committed if b.address is not None),
        )

    def _commit(self, parent: Address, decision: VerifierDecision) -> list[Branch]:
        committed: list[Branch] = []
        for cand in decision.accepted:
            node, fresh = self.graph.add_child(
                parent=parent,
                tokens=cand.tokens,
                latent=cand.latent,
                score=cand.combined,
                signature=cand.signature,
            )
            self.graph.set_status(node.address, NodeStatus.COMMITTED)
            committed.append(
                Branch(
                    parent=parent,
                    candidate=cand,
                    address=node.address,
                    phase=BranchPhase.COMMITTED,
                    fresh=fresh,
                )
            )
        return committed
