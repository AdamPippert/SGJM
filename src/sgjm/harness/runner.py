from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sgjm.branch.lifecycle import BranchLifecycle
from sgjm.branch.policy import BranchPolicy, ScoredCandidate
from sgjm.branch.verifier import VerifierStub
from sgjm.graph.address import Signature
from sgjm.graph.manager import GraphManager
from sgjm.graph.node import NodeStatus
from sgjm.harness.metrics import Metrics, MetricSnapshot
from sgjm.modules.backbone import Backbone
from sgjm.modules.drafter import Drafter
from sgjm.modules.judge import Judge


@dataclass
class HarnessConfig:
    branches_per_step: int = 4
    block_size: int = 4
    max_steps: int = 8
    keep_top_k: int = 2
    accept_threshold: float = -1e9
    judge_weight: float = 1.0
    merge_radius: int = 4


@dataclass
class HarnessRunner:
    backbone: Backbone
    drafter: Drafter
    judge: Judge
    config: HarnessConfig = field(default_factory=HarnessConfig)
    graph: GraphManager = field(init=False)
    lifecycle: BranchLifecycle = field(init=False)
    metrics: Metrics = field(init=False, default_factory=Metrics)

    def __post_init__(self) -> None:
        self.graph = GraphManager()
        self.graph.address_book.merge_radius = self.config.merge_radius
        policy = BranchPolicy(
            keep_top_k=self.config.keep_top_k,
            judge_weight=self.config.judge_weight,
        )
        verifier = VerifierStub(accept_threshold=self.config.accept_threshold)
        self.lifecycle = BranchLifecycle(graph=self.graph, policy=policy, verifier=verifier)

    def run(self, prompt_tokens: Sequence[int]) -> MetricSnapshot:
        state = self.backbone.encode(prompt_tokens)
        root = self.graph.add_root(tokens=prompt_tokens, latent=state.latent)
        frontier = [root.address]
        for _ in range(self.config.max_steps):
            next_frontier: list = []
            for parent_addr in frontier:
                parent = self.graph.get(parent_addr)
                parent_state = self.backbone.encode(parent.tokens)
                drafts = self.drafter.draft(
                    parent_state,
                    k=self.config.branches_per_step,
                    block=self.config.block_size,
                )
                candidates = [
                    ScoredCandidate(
                        tokens=d.tokens,
                        latent=d.latent,
                        signature=Signature.from_latent(d.latent),
                        draft_score=d.log_prob,
                        judge_score=self.judge.score(parent.latent, d.latent),
                    )
                    for d in drafts
                ]
                report = self.lifecycle.step(parent_addr, candidates)
                self.metrics.record(report)
                next_frontier.extend(report.committed_addresses)
            frontier = [
                a
                for a in next_frontier
                if self.graph.get(a).status == NodeStatus.COMMITTED
            ]
            if not frontier:
                break
        return self.metrics.snapshot()
