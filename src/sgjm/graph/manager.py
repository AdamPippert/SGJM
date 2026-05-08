from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from sgjm.graph.address import Address, AddressBook, Signature
from sgjm.graph.node import Node, NodeStatus


@dataclass
class GraphManager:
    address_book: AddressBook = field(default_factory=AddressBook)
    _nodes: dict[Address, Node] = field(default_factory=dict)
    _children: dict[Address, set[Address]] = field(default_factory=lambda: defaultdict(set))
    _root: Address | None = None

    def __len__(self) -> int:
        return len(self._nodes)

    def root(self) -> Node | None:
        return self._nodes.get(self._root) if self._root is not None else None

    def get(self, addr: Address) -> Node:
        return self._nodes[addr]

    def children(self, addr: Address) -> tuple[Node, ...]:
        return tuple(self._nodes[c] for c in self._children.get(addr, ()))

    def parents(self, addr: Address) -> tuple[Node, ...]:
        return tuple(self._nodes[p] for p in self._nodes[addr].parents)

    def frontier(self) -> tuple[Node, ...]:
        leaves = []
        for addr, node in self._nodes.items():
            if node.status in (NodeStatus.REJECTED, NodeStatus.MERGED):
                continue
            if not self._children.get(addr):
                leaves.append(node)
        return tuple(leaves)

    def add_root(
        self,
        tokens: Sequence[int],
        latent: Sequence[float] = (),
        signature: Signature | None = None,
    ) -> Node:
        if self._root is not None:
            raise RuntimeError("graph already has a root")
        sig = signature or self._signature_for(tokens, latent)
        addr, _ = self.address_book.resolve_or_allocate(sig)
        node = Node(
            address=addr,
            parents=(),
            tokens=tuple(tokens),
            signature=sig,
            latent=tuple(latent),
            status=NodeStatus.COMMITTED,
            depth=0,
        )
        self._nodes[addr] = node
        self._root = addr
        return node

    def add_child(
        self,
        parent: Address,
        tokens: Sequence[int],
        latent: Sequence[float] = (),
        score: float = 0.0,
        signature: Signature | None = None,
    ) -> tuple[Node, bool]:
        # Returns (node, fresh). When fresh is False, the candidate signature
        # collided with an existing node and the parent edge was added instead.
        if parent not in self._nodes:
            raise KeyError(f"unknown parent {parent}")
        sig = signature or self._signature_for(tokens, latent)
        addr, fresh = self.address_book.resolve_or_allocate(sig)
        if not fresh and addr in self._nodes:
            existing = self._nodes[addr]
            if parent not in existing.parents:
                merged_parents = existing.parents + (parent,)
                self._nodes[addr] = Node(
                    address=existing.address,
                    parents=merged_parents,
                    tokens=existing.tokens,
                    signature=existing.signature,
                    latent=existing.latent,
                    status=existing.status,
                    score=max(existing.score, score),
                    depth=existing.depth,
                    metadata=existing.metadata,
                )
                self._children[parent].add(addr)
            return self._nodes[addr], False
        depth = self._nodes[parent].depth + 1
        node = Node(
            address=addr,
            parents=(parent,),
            tokens=tuple(tokens),
            signature=sig,
            latent=tuple(latent),
            status=NodeStatus.DRAFT,
            score=score,
            depth=depth,
        )
        self._nodes[addr] = node
        self._children[parent].add(addr)
        return node, True

    def set_status(self, addr: Address, status: NodeStatus) -> None:
        node = self._nodes[addr]
        self._nodes[addr] = Node(
            address=node.address,
            parents=node.parents,
            tokens=node.tokens,
            signature=node.signature,
            latent=node.latent,
            status=status,
            score=node.score,
            depth=node.depth,
            metadata=node.metadata,
        )

    def merge_into(self, src: Address, dst: Address) -> None:
        if src == dst:
            return
        for grand in self._children.pop(src, set()):
            grand_node = self._nodes[grand]
            new_parents = tuple(p if p != src else dst for p in grand_node.parents)
            self._nodes[grand] = Node(
                address=grand_node.address,
                parents=new_parents,
                tokens=grand_node.tokens,
                signature=grand_node.signature,
                latent=grand_node.latent,
                status=grand_node.status,
                score=grand_node.score,
                depth=grand_node.depth,
                metadata=grand_node.metadata,
            )
            self._children[dst].add(grand)
        for kids in self._children.values():
            kids.discard(src)
        self.set_status(src, NodeStatus.MERGED)

    def walk(self, start: Address | None = None) -> Iterator[Node]:
        seen: set[Address] = set()
        stack = [start if start is not None else self._root]
        while stack:
            cur = stack.pop()
            if cur is None or cur in seen:
                continue
            seen.add(cur)
            yield self._nodes[cur]
            stack.extend(self._children.get(cur, ()))

    def _signature_for(self, tokens: Sequence[int], latent: Sequence[float]) -> Signature:
        if latent:
            return Signature.from_latent(latent)
        return Signature.from_tokens(tokens)
