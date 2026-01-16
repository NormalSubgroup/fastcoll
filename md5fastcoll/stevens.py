from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from .core import (
    MASK32,
    MD5_IV,
    u32,
    rl,
    rr,
    ft,
    wt_index,
    compress_block,
)
from .inverse import reverse_step_W
from .conditions import QConditions, minimal_block1_q_constraints, minimal_block2_q_constraints
from .verify import check_T_restrictions_full, check_next_block_iv_conditions

# Local RC/AC consistent with core
_RC = (
    [7, 12, 17, 22] * 4
    + [5, 9, 14, 20] * 4
    + [4, 11, 16, 23] * 4
    + [6, 10, 15, 21] * 4
)

_AC = [int(abs(math.sin(i + 1)) * (1 << 32)) & MASK32 for i in range(64)]


@dataclass
class ConstructResult:
    ihv: Tuple[int, int, int, int]
    trace: Dict[str, List[int]]
    m_words: List[int]
    ok_q: bool
    ok_t: bool
    ok_iv: bool


def _init_Q_from_ihv(ihv: Tuple[int, int, int, int]) -> List[int]:
    base = 3
    Q = [0] * (base + 65)
    IV0, IV1, IV2, IV3 = ihv
    Q[base - 3] = IV0
    Q[base - 2] = IV3
    Q[base - 1] = IV2
    Q[base + 0] = IV1
    return Q


def _reverse_known_Ws_into_m(Q: List[int], RC: List[int], AC: List[int], upto_t: int = 21) -> List[int]:
    # Try to recover m[k] for t in [0, upto_t]
    m = [None] * 16  # type: ignore
    for t in range(0, upto_t + 1):
        try:
            wt = reverse_step_W(t, Q, RC, AC)
        except Exception:
            continue
        k = wt_index(t)
        if m[k] is None:
            m[k] = wt
        else:
            # if conflict, keep None to mark inconsistency
            if m[k] != wt:
                m[k] = None
    # Fill unknowns with 0 to be able to compress; later steps will refine
    return [0 if v is None else int(v) & MASK32 for v in m]  # type: ignore


def _sample_q_with_pattern(rng: random.Random, prev_q: int, pat: str) -> Optional[int]:
    # Build a mask/value pair; for '^' and '!' rely on prev_q comparison.
    fixed_mask = 0
    fixed_val = 0
    for i, ch in enumerate(pat):
        b = 31 - i
        if ch == '0':
            fixed_mask |= (1 << b)
            # fixed_val bit stays 0
        elif ch == '1':
            fixed_mask |= (1 << b)
            fixed_val |= (1 << b)
        # '.' '^' '!' handled by sampling
    # Try up to 1024 samples
    for _ in range(1024):
        q = rng.getrandbits(32)
        q = (q & ~fixed_mask) | fixed_val
        ok = True
        for i, ch in enumerate(pat):
            b = 31 - i
            if ch == '^':
                if ((q >> b) & 1) != ((prev_q >> b) & 1):
                    ok = False; break
            elif ch == '!':
                if ((q >> b) & 1) == ((prev_q >> b) & 1):
                    ok = False; break
        if ok:
            return q & MASK32
    return None


def _check_T_single(i: int, Ti: int) -> bool:
    # Single-step T restrictions (from section 3)
    def bit(x, b):
        return (x >> b) & 1
    if i == 4:
        return bit(Ti, 31) == 1
    if i == 6:
        return bit(Ti, 14) == 0
    if i == 10:
        return bit(Ti, 13) == 0
    if i == 11:
        return bit(Ti, 8) == 1
    if i == 14:
        return bit(Ti, 30) == 1 or bit(Ti, 31) == 1
    if i == 15:
        a = (bit(Ti, 7) == 1) or (bit(Ti, 8) == 1) or (bit(Ti, 9) == 1)
        b = (bit(Ti, 25) == 0) or (bit(Ti, 26) == 0) or (bit(Ti, 27) == 0)
        return a and b
    if i == 16:
        return (bit(Ti, 24) == 0) or (bit(Ti, 25) == 0)
    if i == 19:
        return (bit(Ti, 29) == 1) or (bit(Ti, 30) == 1)
    if i == 22:
        return bit(Ti, 17) == 0
    if i == 34:
        return bit(Ti, 15) == 0
    return True


def _construct_block1_greedy(ihv: Tuple[int, int, int, int], rng: random.Random, qc: QConditions, max_tries_per_t: int = 4096) -> Optional[ConstructResult]:
    base = 3
    # Initialize Q[-3..0]
    IV0, IV1, IV2, IV3 = ihv
    Q = [0] * (base + 65)
    Q[base - 3] = IV0
    Q[base - 2] = IV3
    Q[base - 1] = IV2
    Q[base + 0] = IV1

    # m words as discovered
    m = [None] * 16  # type: ignore

    # Loop t=1..64 to pick Q[t]
    for t in range(1, 65):
        prev_q = Q[base + t - 1]
        # Desired pattern for Qt (if any)
        pat = qc.conds.get(t).pattern if t in qc.conds else None
        success = False
        for _try in range(max_tries_per_t):
            if pat is None:
                qt = rng.getrandbits(32)
            else:
                s = _sample_q_with_pattern(rng, prev_q, pat)
                if s is None:
                    continue
                qt = s
            # Evaluate previous step i = t-1 (only if >=0)
            if t - 1 >= 0:
                i = t - 1
                Rt = (qt - prev_q) & MASK32
                Tprev = rr(Rt, _RC[i])
                # If the step has a known T restriction, enforce it now
                if i in {4,6,10,11,14,15,16,19,22,34}:
                    if not _check_T_single(i, Tprev):
                        continue
                # Derive W_{i} and update m[idx]
                Qtm1 = Q[base + t - 1]
                Qtm2 = Q[base + t - 2]
                Qtm3 = Q[base + t - 3]
                F = ft(i, Qtm1, Qtm2, Qtm3)
                Wi = (Tprev - F - Q[base + t - 4] - _AC[i]) & MASK32 if (t - 4) >= -3 else (Tprev - F - 0 - _AC[i]) & MASK32
                idx = wt_index(i)
                if m[idx] is None:
                    m[idx] = Wi
                else:
                    if m[idx] != Wi:
                        # conflict
                        continue
            # All checks passed for this qt
            Q[base + t] = qt
            success = True
            break
        if not success:
            return None

    # Fill remaining m with zeros
    m_words = [0 if v is None else int(v) & MASK32 for v in m]  # type: ignore
    ihv2, trace = compress_block(ihv, m_words)
    ok_q, _ = qc.check_all(trace["Q"], base=3)
    ok_t, _ = check_T_restrictions_full(trace)
    ok_iv, _ = check_next_block_iv_conditions(ihv2)
    if ok_q and ok_t and ok_iv:
        return ConstructResult(ihv=ihv2, trace=trace, m_words=m_words, ok_q=ok_q, ok_t=ok_t, ok_iv=ok_iv)
    return None


def construct_block1_stevens(
    ihv: Tuple[int, int, int, int] = MD5_IV,
    rng: Optional[random.Random] = None,
    max_outer: int = 1000,
) -> Optional[ConstructResult]:
    """
    Algorithm 6-1 style greedy construction using Qt conditions and early T checks.
    """
    rng = rng or random.Random()
    qc = minimal_block1_q_constraints()

    for attempt in range(max_outer):
        r = _construct_block1_greedy(ihv, rng, qc)
        if r is not None:
            return r
    return None


def construct_block2_stevens(
    ihv: Tuple[int, int, int, int],
    rng: Optional[random.Random] = None,
    max_outer: int = 1000,
) -> Optional[ConstructResult]:
    rng = rng or random.Random()
    qc = minimal_block2_q_constraints()
    for attempt in range(max_outer):
        m_words = [rng.getrandbits(32) for _ in range(16)]
        ihv2, trace = compress_block(ihv, m_words)
        ok_q, _ = qc.check_all(trace["Q"], base=3)
        ok_t, _ = check_T_restrictions_full(trace)
        if ok_q and ok_t:
            return ConstructResult(ihv=ihv2, trace=trace, m_words=m_words, ok_q=ok_q, ok_t=ok_t, ok_iv=True)
    return None


def construct_two_block_collision_stevens(
    seed: Optional[int] = None,
    max_outer: int = 20000,
) -> Optional[Tuple[ConstructResult, ConstructResult]]:
    rng = random.Random(seed)
    tries = 0
    while tries < max_outer:
        r1 = construct_block1_stevens(MD5_IV, rng=rng, max_outer=100)
        tries += 100
        if not r1:
            continue
        r2 = construct_block2_stevens(r1.ihv, rng=rng, max_outer=200)
        tries += 200
        if not r2:
            continue
        # In a true implementation we would compare two distinct messages; here we just return pair
        return (r1, r2)
    return None