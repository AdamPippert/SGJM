from sgjm.graph.manager import GraphManager
from sgjm.graph.node import NodeStatus


def test_root_and_children():
    g = GraphManager()
    root = g.add_root(tokens=[1, 2, 3])
    child, fresh = g.add_child(root.address, tokens=[4, 5])
    assert fresh
    assert child.parents == (root.address,)
    assert g.children(root.address) == (child,)
    assert root in g.parents(child.address)


def test_duplicate_child_signature_merges():
    g = GraphManager()
    root = g.add_root(tokens=[1])
    a, fresh_a = g.add_child(root.address, tokens=[2, 3])
    b, fresh_b = g.add_child(root.address, tokens=[2, 3])
    assert fresh_a
    assert not fresh_b
    assert a.address == b.address


def test_frontier_excludes_merged_and_rejected():
    g = GraphManager()
    root = g.add_root(tokens=[0])
    a, _ = g.add_child(root.address, tokens=[1, 1])
    b, _ = g.add_child(root.address, tokens=[2, 2])
    g.set_status(a.address, NodeStatus.REJECTED)
    front = {n.address for n in g.frontier()}
    assert b.address in front
    assert a.address not in front


def test_walk_visits_all_reachable_nodes():
    g = GraphManager()
    root = g.add_root(tokens=[0])
    a, _ = g.add_child(root.address, tokens=[1])
    b, _ = g.add_child(a.address, tokens=[2])
    seen = {n.address for n in g.walk()}
    assert seen == {root.address, a.address, b.address}
