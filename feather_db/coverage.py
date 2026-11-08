"""Choosing context for coverage, not just rank.

A ranking fills a budget with the most relevant items, and the same fact
told in five sessions takes five slots. Here what is chosen is information:
each item carries units (the index facts it was found by, else its
sentences), units that say about the same thing (word-set Jaccard >= `tau`)
count once, and a unit is worth its item's relevance shared among the
item's units. The value of a set of items is the weight of the units it
covers: monotone submodular (weighted coverage), so greedy is near optimal.

Under a token budget it is a knapsack: lazy greedy by gain per token
(Minoux's lazy evaluation: a gain only shrinks as the set grows, so a stale
bound is re-evaluated only when it reaches the top), and the best single
item if that is worth more (Khuller, Moss and Naor 1999; together a
(1 - 1/e)/2 guarantee). An item whose units are all covered adds nothing and
is left out: the budget goes to something not yet said.

Opt-in (PacketBuilder(select="coverage"), bench.locomo --select coverage):
reader-side changes have cost accuracy before, so this is the default only
after a dev-split win (bench/protocol.py).
"""
from __future__ import annotations

import heapq
import re
from typing import Callable, Sequence

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = frozenset("""a an and are as at be been but by did do does for from had has have he
her hers him his i if in into is it its me my no not of on or our ours she so than that the
their them then there these they this those to too us was we were what when where which who
will with would you your yours""".split())


def units_of(texts: Sequence[str]) -> list[frozenset]:
    """Word sets of each text, stop words out; empty ones dropped."""
    out = []
    for t in texts:
        w = frozenset(x for x in _WORD.findall(t.lower()) if x not in _STOP)
        if w:
            out.append(w)
    return out


def sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if s.strip()]


def _jaccard(a: frozenset, b: frozenset) -> float:
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


def select(relevance: Sequence[float], cost: Sequence[float], units: Sequence[list],
           *, budget: float, max_items: int, tau: float = 0.7) -> list[int]:
    """Indices of the items to keep, in their given (rank) order.

    `relevance[i]` >= 0, `cost[i]` > 0 (tokens, or 1 to count items), and
    `units[i]` a list of word sets (units_of); an item with none is one
    unit of its own. At most `max_items` items, at most `budget` cost."""
    n = len(relevance)
    # Group units that say about the same thing: each joins the first group
    # whose first unit it matches.
    reps: list[frozenset] = []
    groups: list[set] = []
    for i in range(n):
        g = set()
        for u in units[i]:
            for c, r in enumerate(reps):
                if _jaccard(u, r) >= tau:
                    break
            else:
                reps.append(u)
                c = len(reps) - 1
            g.add(c)
        if not g:
            reps.append(frozenset({f"#item{i}"}))
            g.add(len(reps) - 1)
        groups.append(g)
    w = [0.0] * len(reps)
    for i in range(n):
        share = max(0.0, float(relevance[i])) / len(groups[i])
        for c in groups[i]:
            w[c] = max(w[c], share)

    def gain(i: int, covered: set) -> float:
        return sum(w[c] for c in groups[i] if c not in covered)

    covered: set = set()
    chosen: list[int] = []
    spent = 0.0
    heap = [(-gain(i, covered) / cost[i], i) for i in range(n) if cost[i] <= budget]
    heapq.heapify(heap)
    while heap and len(chosen) < max_items:
        _, i = heapq.heappop(heap)
        if spent + cost[i] > budget:
            continue                      # never fits: spent only grows
        g = gain(i, covered)
        if g <= 0.0:
            continue                      # says nothing new, and never will
        d = g / cost[i]
        if heap and d < -heap[0][0]:
            heapq.heappush(heap, (-d, i))     # its bound was stale: back in line
            continue
        chosen.append(i)
        covered |= groups[i]
        spent += cost[i]
    value = sum(w[c] for c in covered)
    singles = [i for i in range(n) if cost[i] <= budget]
    if singles:
        best = max(singles, key=lambda i: (gain(i, set()), -i))
        if gain(best, set()) > value:
            chosen = [best]
    return sorted(chosen)
