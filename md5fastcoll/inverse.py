from __future__ import annotations

from typing import List, Tuple, Dict
from .core import rl, rr, ft, wt_index, u32, MASK32, compress_block, MD5_IV


def reverse_step_W(
    t: int,
    Q: List[int],  # Q array indexed with offset +3 like core.compress_block trace
    RC: List[int],
    AC: List[int],
) -> int:
    base = 3
    Qt = Q[base + t]
    Qt1 = Q[base + t + 1]
    Qt_1 = Q[base + t - 1]
    Qt_2 = Q[base + t - 2]
    Qt_3 = Q[base + t - 3]

    Rt = u32(Qt1 - Qt)
    Tt = rr(Rt, RC[t])
    Wt = u32(Tt - ft(t, Qt, Qt_1, Qt_2) - Qt_3 - AC[t])
    return Wt


def demo_inverse_once() -> Tuple[bool, Dict[str, int]]:
    # Deterministic demo with a fixed block to be reproducible
    # Use pseudo-random but fixed words
    m = [(i * 0x1234567 + 0x89ABCDEF) & MASK32 for i in range(16)]
    ihv = MD5_IV
    _, trace = compress_block(ihv, m)

    ok = True
    mismatches = 0
    for t in range(64):
        Wt = reverse_step_W(t, trace["Q"], trace["RC"], trace["AC"])
        if Wt != trace["W"][t]:
            ok = False
            mismatches += 1
    return ok, {"mismatches": mismatches}