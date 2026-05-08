from sgjm.branch.lifecycle import BranchLifecycle
from sgjm.branch.policy import BranchPolicy, ScoredCandidate
from sgjm.branch.verifier import VerifierStub
from sgjm.graph.address import Signature
from sgjm.graph.manager import GraphManager
from sgjm.graph.node import NodeStatus


def _candidate(tokens, latent, draft=0.0, judge=0.0):
    return ScoredCandidate(
        tokens=tuple(tokens),
        latent=tuple(latent),
        signature=Signature.from_latent(latent) if latent else Signature.from_tokens(tokens),
        draft_score=draft,
        judge_score=judge,
    )


def test_policy_keeps_top_k():
    policy = BranchPolicy(keep_top_k=2)
    cands = [
        _candidate([1], [0.1, 0.2], draft=1.0, judge=0.0),
        _candidate([2], [0.3, 0.4], draft=0.5, judge=0.5),
        _candidate([3], [0.5, 0.6], draft=-1.0, judge=0.0),
        _candidate([4], [0.7, 0.8], draft=0.2, judge=0.2),
    ]
    kept = policy.rank(cands)
    assert len(kept) == 2
    assert kept[0].combined >= kept[1].combined


def test_lifecycle_step_commits_accepted_candidates():
    g = GraphManager()
    root = g.add_root(tokens=[0])
    life = BranchLifecycle(
        graph=g,
        policy=BranchPolicy(keep_top_k=2),
        verifier=VerifierStub(accept_threshold=-1e9),
    )
    cands = [
        _candidate([1, 1], [0.1] * 16, draft=0.0, judge=0.0),
        _candidate([2, 2], [0.2] * 16, draft=0.0, judge=0.0),
    ]
    report = life.step(root.address, cands)
    assert report.drafted == 2
    assert report.accepted == 2
    for addr in report.committed_addresses:
        assert g.get(addr).status == NodeStatus.COMMITTED


def test_lifecycle_rejects_below_threshold():
    g = GraphManager()
    root = g.add_root(tokens=[0])
    life = BranchLifecycle(
        graph=g,
        policy=BranchPolicy(keep_top_k=4),
        verifier=VerifierStub(accept_threshold=10.0),
    )
    cands = [_candidate([1], [0.1] * 16, draft=0.0, judge=0.0)]
    report = life.step(root.address, cands)
    assert report.accepted == 0
    assert report.committed_addresses == ()
