from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .core import (
    compress_block,
    MD5_IV,
    MASK32,
)
from .conditions import BitCond
from .verify import check_T_restrictions_tail, check_next_block_iv_conditions
from .stevens_full import Block1FullSearcher, Block2FullSearcher, search_collision_full


@dataclass
class BlockResult:
    ihv: Tuple[int, int, int, int]
    trace: Dict[str, List[int]]
    m_words: List[int]


def sample_q_from_pattern(prev_q: Optional[int], pat: BitCond) -> int:
    # Random sampling of a Q value satisfying pattern pat relative to prev_q
    q = 0
    for i, ch in enumerate(pat.pattern):
        b = 31 - i
        if ch == ".":
            v = random.getrandbits(1)
        elif ch == "0":
            v = 0
        elif ch == "1":
            v = 1
        elif ch == "^":
            if prev_q is None:
                v = 0
            else:
                v = (prev_q >> b) & 1
        elif ch == "!":
            if prev_q is None:
                v = 1
            else:
                v = 1 - ((prev_q >> b) & 1)
        else:
            v = 0
        if v:
            q |= (1 << b)
    return q & MASK32


def words_from_trace(trace: Dict[str, List[int]]) -> List[int]:
    # Reconstruct m words from trace by reading W[t] at first occurrence of each word index
    # But we already stored W[t] in trace; mapping back by index schedule
    # Here we simply invert schedule by computing k per t and populating m[k] = W[t]
    W = trace["W"]
    # Build wt_index inline to avoid circular import
    def wt_index(t: int) -> int:
        if 0 <= t < 16:
            return t
        if 16 <= t < 32:
            return (5 * t + 1) % 16
        if 32 <= t < 48:
            return (3 * t + 5) % 16
        if 48 <= t < 64:
            return (7 * t) % 16
        raise ValueError

    m = [0] * 16
    filled = [False] * 16
    for t in range(64):
        k = wt_index(t)
        if not filled[k]:
            m[k] = W[t]
            filled[k] = True
    return m


def run_block(ihv: Tuple[int, int, int, int], m_words: List[int]) -> BlockResult:
    ihv2, trace = compress_block(ihv, m_words)
    # derive W already in trace, reconstruct m for convenience
    m_back = words_from_trace(trace)
    return BlockResult(ihv=ihv2, trace=trace, m_words=m_back)


def random_block() -> List[int]:
    return [random.getrandbits(32) for _ in range(16)]


def search_block1_once(
    ihv: Tuple[int, int, int, int] = MD5_IV,
    rng: Optional[random.Random] = None,
    max_restarts: int = 100,
) -> Tuple[bool, Optional[BlockResult], Dict[str, object]]:
    searcher = Block1FullSearcher(rng)
    res = searcher.search(ihv, max_restarts=max_restarts)
    ok = res is not None
    stats: Dict[str, object] = {"ok": ok, "restarts": max_restarts}
    if res is not None:
        ok_t, issues_t = check_T_restrictions_tail(res.trace)
        ok_iv, issues_iv = check_next_block_iv_conditions(res.ihv)
        stats.update({"okT": ok_t, "okIV": ok_iv, "issuesT": issues_t, "issuesIV": issues_iv})
    return ok, res, stats


def search_block2_once(
    ihv: Tuple[int, int, int, int],
    rng: Optional[random.Random] = None,
    max_restarts: int = 100,
) -> Tuple[bool, Optional[BlockResult], Dict[str, object]]:
    searcher = Block2FullSearcher(rng)
    res = searcher.search(ihv, max_restarts=max_restarts)
    ok = res is not None
    stats: Dict[str, object] = {"ok": ok, "restarts": max_restarts}
    if res is not None:
        ok_t, issues_t = check_T_restrictions_tail(res.trace)
        stats.update({"okT": ok_t, "issuesT": issues_t})
    return ok, res, stats


def search_two_block_collision(
    max_restarts: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[bool, Dict[str, object]]:
    res = search_collision_full(seed=seed, max_restarts=max_restarts)
    if res is None:
        return False, {"restarts": max_restarts, "seed": seed}
    b1, b2 = res
    return True, {
        "restarts": max_restarts,
        "seed": seed,
        "block1": b1,
        "block2": b2,
    }
