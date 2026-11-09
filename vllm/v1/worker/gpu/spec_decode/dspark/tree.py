# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TreeSpark proposal and verification for DSpark."""

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


def _log_acceptance(log_q: float, a: float, b: float) -> float:
    if log_q == -math.inf:
        return -math.inf
    logit = log_q - math.log1p(-min(math.exp(log_q), 1 - 1e-9))
    z = a * logit + b
    return min(z, 0.0) - math.log1p(math.exp(-abs(z)))


@dataclass
class DraftTree:
    # The anchor is row 0. Nodes are appended in parent-before-child order.
    tokens: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    # Recompute the exact conditional only for parents reached by verification.
    conditional: Callable[[int, int], torch.Tensor] | None = None

    def attention_mask(self, count: int) -> list[list[bool]]:
        mask = [[False] * (count + 1) for _ in range(count + 1)]
        mask[0][0] = True
        for node in range(1, count + 1):
            mask[node] = mask[self.parents[node - 1]].copy()
            mask[node][node] = True
        return mask


def expand_tree(
    anchor: int,
    conditional: Callable[[int, int], torch.Tensor],
    temperature: float,
    theta: float,
    max_depth: int,
    max_nodes: int,
    max_children: int,
    a: float,
    b: float,
    generator: torch.Generator,
) -> DraftTree:
    tree = DraftTree(conditional=conditional if temperature else None)
    cutoff = math.log(theta) if theta else -math.inf
    # (-log survival, insertion order, parent row, depth, token).
    # A token of None represents a stochastic sibling slot, before its draw.
    frontier: list[tuple[float, int, int, int, int | None]] = []
    next_order = 0
    survival = [0.0]
    draws: dict[int, list[tuple[int, float]]] = {}
    bounds: dict[int, list[tuple[int, float]]] = {}
    drawn: dict[int, set[int]] = {}

    def push(score: float, parent: int, depth: int, token: int | None) -> None:
        nonlocal next_order
        if score >= cutoff and score > -math.inf:
            heapq.heappush(frontier, (-score, next_order, parent, depth, token))
            next_order += 1

    def push_sibling(parent: int, depth: int) -> None:
        used = drawn[parent]
        if len(used) >= len(draws[parent]):
            return
        bound = next(log_q for token, log_q in bounds[parent] if token not in used)
        push(survival[parent] + _log_acceptance(bound, a, b), parent, depth, None)

    def expand_parent(parent: int, token: int, depth: int) -> None:
        if depth > max_depth:
            return
        logits = conditional(depth - 1, token).float()
        log_q = logits.log_softmax(dim=-1)
        k = min(max_children, logits.numel())
        values, ids = log_q.topk(k)
        if not temperature:
            for child, value in zip(ids.tolist(), values.tolist(), strict=True):
                push(
                    survival[parent] + _log_acceptance(value, a, b),
                    parent,
                    depth,
                    child,
                )
            return
        # Gumbel top-k gives the ordered, without-replacement sample. Only
        # the upper bound on remaining log q is read before admitting a slot.
        noise = torch.empty_like(logits).exponential_(generator=generator).log()
        ranked = (logits / temperature - noise).topk(k).indices
        sample = zip(ranked.tolist(), log_q[ranked].tolist(), strict=True)
        draws[parent] = [
            (child, value) for child, value in sample if math.isfinite(value)
        ]
        bounds[parent] = list(zip(ids.tolist(), values.tolist(), strict=True))
        drawn[parent] = set()
        push_sibling(parent, depth)

    expand_parent(0, anchor, 1)
    while frontier and len(tree.tokens) < max_nodes:
        neg_score, _, parent, depth, token = heapq.heappop(frontier)
        if token is None:
            used = drawn[parent]
            token, log_q = draws[parent][len(used)]
            used.add(token)
            score = survival[parent] + _log_acceptance(log_q, a, b)
            push_sibling(parent, depth)
        else:
            score = -neg_score
        tree.tokens.append(token)
        tree.parents.append(parent)
        tree.depths.append(depth)
        survival.append(score)
        if len(tree.tokens) < max_nodes:
            expand_parent(len(tree.tokens), token, depth + 1)
    return tree


def verify_tree(
    tree: DraftTree,
    count: int,
    anchor: int,
    target_logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
) -> tuple[list[int], list[int]]:
    """Return accepted node rows and output tokens, including the bonus token."""
    children: list[list[int]] = [[] for _ in range(count + 1)]
    for child, parent in enumerate(tree.parents[:count], start=1):
        children[parent].append(child)
    path: list[int] = []
    output: list[int] = []
    row = 0
    while True:
        kids = children[row]
        if not temperature:
            token = int(target_logits[row].argmax())
            accepted = next((i for i in kids if tree.tokens[i - 1] == token), None)
        else:
            p = target_logits[row].float().softmax(dim=-1)
            accepted = None
            if kids:
                assert tree.conditional is not None
                depth = tree.depths[kids[0] - 1]
                parent_token = anchor if row == 0 else tree.tokens[row - 1]
                draft_logits = tree.conditional(depth - 1, parent_token).float()
                draft_logits = draft_logits / temperature
                for child in kids:
                    token = tree.tokens[child - 1]
                    q = draft_logits.softmax(dim=-1)
                    u = torch.rand((), device=p.device, generator=generator)
                    if bool(u < p[token] / q[token]):
                        accepted = child
                        break
                    residual = (p - q).clamp_min_(0)
                    mass = residual.sum()
                    # A zero residual is unreachable after rejection in exact
                    # arithmetic. Keep a valid distribution if it rounds to zero.
                    p = torch.where(mass > 0, residual / mass, p)
                    draft_logits[token] = -torch.inf
            if accepted is None:
                token = int(torch.multinomial(p, 1, generator=generator))
        output.append(token)
        if accepted is None:
            return path, output
        path.append(accepted)
        row = accepted
