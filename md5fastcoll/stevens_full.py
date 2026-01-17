from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from .core import MASK32, MD5_IV, rl, rr, ft, wt_index, compress_block
from .conditions import QConditions, minimal_block1_q_constraints, minimal_block2_q_constraints
from .verify import check_T_restrictions_tail, check_next_block_iv_conditions

# Local copies consistent with core
_RC = (
    [7, 12, 17, 22] * 4
    + [5, 9, 14, 20] * 4
    + [4, 11, 16, 23] * 4
    + [6, 10, 15, 21] * 4
)
_AC = [int(abs(math.sin(i + 1)) * (1 << 32)) & MASK32 for i in range(64)]

QList = List[Optional[int]]
Symbols = Dict[str, Optional[int]]


def _init_Q_from_ihv(ihv: Tuple[int, int, int, int], *, base: int = 3) -> QList:
    Q: QList = [None] * (base + 65)
    IV0, IV1, IV2, IV3 = ihv
    Q[base - 3] = IV0
    Q[base - 2] = IV3
    Q[base - 1] = IV2
    Q[base + 0] = IV1
    return Q


def _q_req(Q: QList, idx: int, *, base: int = 3) -> int:
    val = Q[base + idx]
    if val is None:
        raise ValueError(f"Q[{idx}] is not set")
    return val & MASK32


@dataclass
class _PatternSpec:
    fixed_mask: int
    fixed_val: int
    free_bits: List[int]
    dep_same_mask: int
    dep_flip_mask: int


def _pattern_spec(pat: str) -> _PatternSpec:
    fixed_mask = 0
    fixed_val = 0
    free_bits: List[int] = []
    dep_same_mask = 0
    dep_flip_mask = 0
    for i, ch in enumerate(pat):
        b = 31 - i
        if ch == "0":
            fixed_mask |= (1 << b)
        elif ch == "1":
            fixed_mask |= (1 << b)
            fixed_val |= (1 << b)
        elif ch == ".":
            free_bits.append(b)
        elif ch == "^":
            dep_same_mask |= (1 << b)
        elif ch == "!":
            dep_flip_mask |= (1 << b)
    return _PatternSpec(
        fixed_mask=fixed_mask,
        fixed_val=fixed_val,
        free_bits=free_bits,
        dep_same_mask=dep_same_mask,
        dep_flip_mask=dep_flip_mask,
    )


def _iter_free_masks(bits: List[int]) -> List[int]:
    masks: List[int] = []
    total = 1 << len(bits)
    for i in range(total):
        m = 0
        for j, b in enumerate(bits):
            if (i >> j) & 1:
                m |= (1 << b)
        masks.append(m)
    return masks


def _iter_q9_q10_subspace(Q: QList, qc: QConditions) -> List[Tuple[int, int]]:
    base = 3
    q8 = Q[base + 8]
    q11 = Q[base + 11]
    q9_base = Q[base + 9]
    q10_base = Q[base + 10]
    if None in (q8, q11, q9_base, q10_base):
        return []
    if 9 not in qc.conds or 10 not in qc.conds:
        return []

    spec9 = _pattern_spec(qc.conds[9].pattern)
    spec10 = _pattern_spec(qc.conds[10].pattern)
    q10_dep_mask = spec10.dep_same_mask | spec10.dep_flip_mask

    q9_free = [
        b
        for b in spec9.free_bits
        if ((q11 >> b) & 1) == 1 and ((q10_dep_mask >> b) & 1) == 0
    ]
    q10_free = [b for b in spec10.free_bits if ((q11 >> b) & 1) == 0]

    q9_masks = _iter_free_masks(q9_free)
    q10_masks = _iter_free_masks(q10_free)
    pairs: List[Tuple[int, int]] = []
    # NOTE: keep q10 masks materialized (small) but avoid quadratic pair-list building
    # at call sites when they can early-exit.
    for q9_mask in q9_masks:
        q9 = (q9_base ^ q9_mask) & MASK32
        for q10_mask in q10_masks:
            q10 = (q10_base ^ q10_mask) & MASK32
            pairs.append((q9, q10))
    return pairs


def _forward_q22_q64(
    Q: QList,
    m: List[Optional[int]],
    qc: QConditions,
    start_t: int = 21,
    end_t: int = 63,
    *,
    symbols: Optional[Symbols] = None,
) -> Tuple[bool, Dict[str, int]]:
    base = 3
    symbols = symbols if symbols is not None else {}
    issues: Dict[str, int] = {}
    for t in range(start_t, end_t + 1):
        idx = wt_index(t)
        if idx >= len(m) or m[idx] is None:
            return False, {"missing_m": t}
        Qt = Q[base + t]
        Qtm1 = Q[base + t - 1]
        Qtm2 = Q[base + t - 2]
        Qtm3 = Q[base + t - 3]
        if None in (Qt, Qtm1, Qtm2, Qtm3):
            return False, {"missing_q": t}
        Tt = (ft(t, Qt, Qtm1, Qtm2) + Qtm3 + _AC[t] + (m[idx] & MASK32)) & MASK32
        Rt = rl(Tt, _RC[t])
        Qt1 = (Qt + Rt) & MASK32
        if (t + 1) in qc.conds and not qc.conds[t + 1].check(Qt1, Qt, symbols):
            return False, {"bad_q": t + 1}
        Q[base + t + 1] = Qt1
        if t == 22 and _bit(Tt, 17) != 0:
            issues["T22[17]"] = _bit(Tt, 17)
            return False, issues
        if t == 34 and _bit(Tt, 15) != 0:
            issues["T34[15]"] = _bit(Tt, 15)
            return False, issues
    return True, issues

def _bit(x: int, b: int) -> int:
    return (x >> b) & 1


def _choose_Q_with_constraints(
    rng: random.Random,
    Q: QList,
    qc: QConditions,
    t: int,
    *,
    symbols: Optional[Symbols] = None,
    max_tries: int = 4096,
) -> Optional[int]:
    base = 3
    if len(Q) <= base + t + 1:
        return None

    symbols = symbols if symbols is not None else {}
    prev_q = Q[base + t - 1] if (base + t - 1) >= 0 else None
    next_q = Q[base + t + 1] if (base + t + 1) < len(Q) else None
    pat = qc.conds.get(t).pattern if t in qc.conds else None
    next_pat = qc.conds.get(t + 1).pattern if (t + 1) in qc.conds else None

    # Precompute fixed bit constraints from the pattern (0/1) and from known neighbors.
    fixed_mask = 0
    fixed_val = 0
    if pat is not None:
        for i, ch in enumerate(pat):
            b = 31 - i
            if ch == "0":
                fixed_mask |= (1 << b)
            elif ch == "1":
                fixed_mask |= (1 << b)
                fixed_val |= (1 << b)
            elif ch in "^!" and prev_q is not None:
                pv = (prev_q >> b) & 1
                need = pv if ch == "^" else 1 - pv
                if fixed_mask & (1 << b):
                    if ((fixed_val >> b) & 1) != need:
                        return None
                fixed_mask |= (1 << b)
                if need:
                    fixed_val |= (1 << b)
                else:
                    fixed_val &= ~(1 << b)
            elif ch in ("I", "J", "K"):
                key = "I" if ch == "K" else ch
                bound = symbols.get(key)
                if bound is None:
                    continue
                fixed_mask |= (1 << b)
                if bound:
                    fixed_val |= (1 << b)
                else:
                    fixed_val &= ~(1 << b)

    if next_q is not None and next_pat is not None:
        for i, ch in enumerate(next_pat):
            b = 31 - i
            if ch in "^!":
                nv = (next_q >> b) & 1
                need = nv if ch == "^" else 1 - nv
                if fixed_mask & (1 << b):
                    if ((fixed_val >> b) & 1) != need:
                        return None
                fixed_mask |= (1 << b)
                if need:
                    fixed_val |= (1 << b)
                else:
                    fixed_val &= ~(1 << b)
            elif ch in ("I", "J", "K"):
                key = "I" if ch == "K" else ch
                bound = symbols.get(key)
                if bound is None:
                    # next_q forces the binding.
                    symbols[key] = (next_q >> b) & 1
                    bound = symbols[key]
                need = bound
                if fixed_mask & (1 << b):
                    if ((fixed_val >> b) & 1) != need:
                        return None
                fixed_mask |= (1 << b)
                if need:
                    fixed_val |= (1 << b)
                else:
                    fixed_val &= ~(1 << b)

    for _ in range(max_tries):
        qt = rng.getrandbits(32)
        qt = (qt & ~fixed_mask) | fixed_val
        if pat is not None and prev_q is not None:
            if not qc.conds[t].check(qt, prev_q, symbols):
                continue
        if next_pat is not None and next_q is not None:
            ok = True
            for i, ch in enumerate(next_pat):
                if ch not in "^!":
                    continue
                b = 31 - i
                if ch == "^" and _bit(qt, b) != _bit(next_q, b):
                    ok = False
                    break
                if ch == "!" and _bit(qt, b) == _bit(next_q, b):
                    ok = False
                    break
            if not ok:
                continue
        return qt & MASK32
    return None


def _compute_W_from_Q(Q: QList, i: int) -> int:
    # Reverse computation: Wi = RR(Qi+1 - Qi, RCi) - fi(Qi, Qi-1, Qi-2) - Qi-3 - ACi
    base = 3
    if len(Q) <= base + i + 1:
        raise ValueError(f"Q array too short for step {i}")

    Qt = _q_req(Q, i, base=base)
    Qtm1 = _q_req(Q, i - 1, base=base)
    Qtm2 = _q_req(Q, i - 2, base=base)
    Qtm3 = _q_req(Q, i - 3, base=base)
    Qt1 = _q_req(Q, i + 1, base=base)
    
    # Rt = Qt+1 - Qt (rotation after addition)
    Rt = (Qt1 - Qt) & MASK32
    # Tt = RR(Rt, RCi) (before rotation)
    Tt = rr(Rt, _RC[i]) & MASK32
    # Wi = Tt - fi - Qi-3 - ACi
    fi = ft(i, Qt, Qtm1, Qtm2) & MASK32
    Wi = (Tt - fi - Qtm3 - _AC[i]) & MASK32
    
    return Wi


def _forward_step_set_Q(
    Q: QList,
    m: List[Optional[int]],
    t: int,
    qc: Optional[QConditions] = None,
    *,
    symbols: Optional[Symbols] = None,
) -> bool:
    base = 3
    symbols = symbols if symbols is not None else {}

    idx = wt_index(t)
    if idx >= len(m) or m[idx] is None:
        return False
    Qt = _q_req(Q, t, base=base)
    Qtm1 = _q_req(Q, t - 1, base=base)
    Qtm2 = _q_req(Q, t - 2, base=base)
    Qtm3 = _q_req(Q, t - 3, base=base)
    Tt = (ft(t, Qt, Qtm1, Qtm2) + Qtm3 + _AC[t] + (m[idx] & MASK32)) & MASK32
    Qt1 = (Qt + rl(Tt, _RC[t])) & MASK32
    if qc is not None and (t + 1) in qc.conds:
        if not qc.conds[t + 1].check(Qt1, Qt, symbols):
            return False
    if Q[base + t + 1] is not None and (Q[base + t + 1] & MASK32) != Qt1:
        return False
    Q[base + t + 1] = Qt1
    return True


def _materialize_m(m: List[Optional[int]]) -> Optional[List[int]]:
    if any(v is None for v in m):
        return None
    return [int(v) & MASK32 for v in m]


def _set_m(m: List[Optional[int]], idx: int, val: int) -> bool:
    if m[idx] is None:
        m[idx] = val & MASK32
        return True
    return (m[idx] == (val & MASK32))


@dataclass
class BlockResult:
    ihv: Tuple[int, int, int, int]
    trace: Dict[str, List[int]]
    m_words: List[int]


class Block1FullSearcher:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.qc = minimal_block1_q_constraints()

    def step1_choose_Qs(self, Q: QList, symbols: Optional[Symbols] = None) -> bool:
        # Algorithm 6-1 Step 1: Choose Q1, Q3, ..., Q16 fulfilling conditions
        # Note: We choose odd Qs first as per paper algorithm description
        symbols = symbols if symbols is not None else {}
        base = 3
        for t in [1, 3, 5, 7, 9, 11, 13, 15]:
            qt = _choose_Q_with_constraints(self.rng, Q, self.qc, t, symbols=symbols)
            if qt is None:
                return False
            Q[base + t] = qt
        # Then choose even Qs Q4, Q6, Q8, Q10, Q12, Q14, Q16 (Q2 is computed later)
        for t in [4, 6, 8, 10, 12, 14, 16]:
            qt = _choose_Q_with_constraints(self.rng, Q, self.qc, t, symbols=symbols)
            if qt is None:
                return False
            Q[base + t] = qt
        return True

    def step2_compute_m0_to_m15(self, Q: QList, m: List[Optional[int]]) -> bool:
        # Algorithm 6-1 Step 2: Calculate m0, m6, ..., m15 using reverse steps
        # The paper specifically mentions m0, m6, ..., m15 (not m1-m5)
        base = 3
        for t in range(0, 16):
            # Skip if this would compute m1-m5 which are done in step 3
            idx = wt_index(t)
            if idx in [1, 2, 3, 4, 5]:  # Skip m1-m5 for now
                continue
            if Q[base + t] is None or Q[base + t + 1] is None:
                return False
            wt = _compute_W_from_Q(Q, t)
            if not _set_m(m, idx, wt):
                return False
        return True

    def step3_loop_Q17_to_Q21(self, Q: QList, m: List[Optional[int]], symbols: Optional[Symbols] = None) -> bool:
        # Algorithm 6-1 Step 3: Loop until Q17, ..., Q21 are fulfilling conditions
        # (a) Choose Q17 fulfilling conditions; (b) Calculate m1 at t=16
        # (c) Calculate Q2 and m2, m3, m4, m5; (d) Calculate Q18, ..., Q21
        symbols = symbols if symbols is not None else {}
        base = 3
        for _ in range(4096):  # Increased attempts for better success rate
            # reset Q17..Q21 from previous attempts
            for t in range(17, 22):
                Q[base + t] = None
            # reset m1..m5 from previous attempts
            for idx in range(1, 6):
                m[idx] = None
            # (a) Choose Q17 fulfilling conditions
            qt17 = _choose_Q_with_constraints(self.rng, Q, self.qc, 17, symbols=symbols)
            if qt17 is None:
                continue
            Q[base+17] = qt17
            
            # (b) Calculate m1 at t=16
            try:
                w16 = _compute_W_from_Q(Q, 16)  # t=16 corresponds to m1
            except Exception:
                continue
            m[1] = w16
            
            # (c) Calculate Q2 deterministically using t=1 and m1
            # Q2 = Q1 + RL(ft(1,Q1,Q0,Q-1) + Q-2 + AC1 + m1, RC1)
            try:
                Q1 = Q[base+1]; Q0 = Q[base+0]; Qm1 = Q[base-1]; Qm2 = Q[base-2]
                if None in (Q1, Q0, Qm1, Qm2):
                    continue
                T1 = (ft(1, Q1, Q0, Qm1) + Qm2 + _AC[1] + (w16 & MASK32)) & MASK32
                R1 = rl(T1, _RC[1])
                Q2 = (Q1 + R1) & MASK32
                # Enforce constraints on Q2 if any
                if 2 in self.qc.conds and not self.qc.conds[2].check(Q2, Q1, symbols):
                    continue
                # Also enforce constraints from Q3 (next)
                q3 = Q[base + 3]
                if 3 in self.qc.conds:
                    if q3 is None or not self.qc.conds[3].check(q3, Q2, symbols):
                        continue
                Q[base+2] = Q2
            except Exception:
                continue
            
            # (d) Calculate m2..m5 via reverse steps t=2..5
            ok = True
            for t in range(2, 6):
                try:
                    wt = _compute_W_from_Q(Q, t)
                except Exception:
                    ok = False
                    break
                idx = wt_index(t)
                m[idx] = wt
            if not ok:
                continue
            # (e) Calculate Q18..Q21 forward using known message words
            for t in range(17, 21):
                try:
                    if not _forward_step_set_Q(Q, m, t, self.qc, symbols=symbols):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok:
                return True
        return False

    def step4_q9_q10_subspace(
        self,
        ihv: Tuple[int,int,int,int],
        Q: QList,
        m: List[Optional[int]],
        symbols: Optional[Symbols] = None,
        *,
        budget: int | None = None,
    ) -> bool:
        base = 3
        symbols = symbols if symbols is not None else {}
        base_m11 = m[11] if m[11] is not None else self._m11_from_Q(Q, m)
        if base_m11 is None:
            return False
        old_q9, old_q10 = Q[base + 9], Q[base + 10]
        if old_q9 is None or old_q10 is None:
            return False

        pairs = _iter_q9_q10_subspace(Q, self.qc)
        if not pairs:
            return False

        old_m = {idx: m[idx] for idx in (8, 9, 10, 12, 13)}
        iters = 0
        for q9, q10 in pairs:
            iters += 1
            if budget is not None and iters > budget:
                break
            Q[base + 9], Q[base + 10] = q9, q10
            if self._m11_from_Q(Q, m) != base_m11:
                continue
            ok = True
            for t in (8, 9, 10, 12, 13):
                try:
                    wt = _compute_W_from_Q(Q, t)
                except Exception:
                    ok = False
                    break
                idx = wt_index(t)
                m[idx] = wt
            if not ok:
                for idx, val in old_m.items():
                    m[idx] = val
                continue
            ok_fwd, _ = _forward_q22_q64(Q, m, self.qc, symbols=dict(symbols))
            if not ok_fwd:
                for idx, val in old_m.items():
                    m[idx] = val
                continue
            IV0 = Q[base - 3]
            IV1 = Q[base + 0]
            IV2 = Q[base - 1]
            IV3 = Q[base - 2]
            if None in (IV0, IV1, IV2, IV3):
                for idx, val in old_m.items():
                    m[idx] = val
                continue
            ihv2 = (
                (IV0 + Q[base + 61]) & MASK32,
                (IV1 + Q[base + 64]) & MASK32,
                (IV2 + Q[base + 63]) & MASK32,
                (IV3 + Q[base + 62]) & MASK32,
            )
            ok_iv, _ = check_next_block_iv_conditions(ihv2)
            if ok_iv:
                return True
            for idx, val in old_m.items():
                m[idx] = val

        Q[base + 9], Q[base + 10] = old_q9, old_q10
        for idx, val in old_m.items():
            m[idx] = val
        return False

    def _m11_from_Q(self, Q: QList, m: List[Optional[int]]) -> Optional[int]:
        # m11 = RR(Q12 - Q11, RC11) - f11(Q11, Q10, Q9) - Q8 - AC11
        # 这是论文中的关键公式，用于保持m11不变
        base = 3
        if len(Q) <= base + 12:
            return None
            
        Q12 = Q[base+12]
        Q11 = Q[base+11]
        Q10 = Q[base+10]
        Q9 = Q[base+9]
        Q8 = Q[base+8]
        
        if None in (Q12, Q11, Q10, Q9, Q8):
            return None
            
        t = 11
        # f11 = (Q11 & Q10) | ((~Q11) & Q9)  for first round
        f11 = ft(t, Q11, Q10, Q9) & MASK32
        Rt = (Q12 - Q11) & MASK32
        Tt = rr(Rt, _RC[11]) & MASK32
        m11 = (Tt - f11 - Q8 - _AC[11]) & MASK32
        return m11

    def search(self, ihv: Tuple[int,int,int,int] = MD5_IV, max_restarts: int = 100) -> Optional[BlockResult]:
        # Strict Algorithm 6-1
        for _ in range(max_restarts):
            Q = _init_Q_from_ihv(ihv)
            m: List[Optional[int]] = [None]*16
            symbols: Symbols = {}
            if not self.step1_choose_Qs(Q, symbols):
                continue
            if not self.step2_compute_m0_to_m15(Q, m):
                continue
            if not self.step3_loop_Q17_to_Q21(Q, m, symbols):
                continue
            if not self.step4_q9_q10_subspace(ihv, Q, m, symbols):
                continue
            words = _materialize_m(m)
            if words is None:
                continue
            ihv2, trace = compress_block(ihv, words)
            return BlockResult(ihv=ihv2, trace=trace, m_words=words)
        return None


class Block2FullSearcher:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.qc = minimal_block2_q_constraints()

    def step1_choose_Q2_to_Q16(self, Q: QList, symbols: Optional[Symbols] = None) -> bool:
        # Algorithm 6-2 Step 1: Choose Q2, ..., Q16 fulfilling conditions 
        symbols = symbols if symbols is not None else {}
        base = 3
        for t in range(2, 17):
            qt = _choose_Q_with_constraints(self.rng, Q, self.qc, t, symbols=symbols)
            if qt is None:
                return False
            Q[base + t] = qt
        return True

    def step2_compute_m5_to_m15(self, Q: QList, m: List[Optional[int]]) -> bool:
        # Algorithm 6-2 Step 2: Calculate m5, ..., m15
        base = 3
        for t in range(5, 16):
            if Q[base + t] is None or Q[base + t + 1] is None:
                return False
            wt = _compute_W_from_Q(Q, t)
            idx = wt_index(t)
            if not _set_m(m, idx, wt):
                return False
        return True

    def step3_loop_Q1_and_m0_to_m4(self, Q: QList, m: List[Optional[int]], symbols: Optional[Symbols] = None) -> bool:
        # Algorithm 6-2 Step 3: Loop until Q17, ..., Q21 are fulfilling conditions
        # (a) Choose Q1 fulfilling conditions; (b) Calculate m0, ..., m4
        # (c) Calculate Q17, ..., Q21
        symbols = symbols if symbols is not None else {}
        base = 3
        for _ in range(4096):
            Q[base + 1] = None
            for t in range(17, 22):
                Q[base + t] = None
            for idx in range(0, 5):
                m[idx] = None
            # (a) Choose Q1 fulfilling conditions
            q1 = _choose_Q_with_constraints(self.rng, Q, self.qc, 1, symbols=symbols)
            if q1 is None:
                continue
            Q[base+1] = q1
            
            # (b) Calculate m0, ..., m4
            ok = True
            for t in range(0, 5):
                if Q[base + t] is None or Q[base + t + 1] is None:
                    ok = False
                    break
                wt = _compute_W_from_Q(Q, t)
                idx = wt_index(t)
                m[idx] = wt
            
            if not ok:
                continue
            # (c) Calculate Q17, ..., Q21 forward
            for t in range(16, 21):
                try:
                    if not _forward_step_set_Q(Q, m, t, self.qc, symbols=symbols):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok:
                return True
        return False

    def step4_q9_q10_subspace(
        self,
        ihv: Tuple[int,int,int,int],
        Q: QList,
        m: List[Optional[int]],
        symbols: Optional[Symbols] = None,
    ) -> bool:
        base_m11 = m[11] if m[11] is not None else self._m11_from_Q(Q, m)
        if base_m11 is None:
            return False
        symbols = symbols if symbols is not None else {}
        base = 3
        old_q9, old_q10 = Q[base + 9], Q[base + 10]
        if old_q9 is None or old_q10 is None:
            return False

        pairs = _iter_q9_q10_subspace(Q, self.qc)
        if not pairs:
            return False

        old_m = {idx: m[idx] for idx in (8, 9, 10, 12, 13)}
        for q9, q10 in pairs:
            Q[base + 9], Q[base + 10] = q9, q10
            if self._m11_from_Q(Q, m) != base_m11:
                continue
            ok = True
            for t in (8, 9, 10, 12, 13):
                try:
                    wt = _compute_W_from_Q(Q, t)
                except Exception:
                    ok = False
                    break
                idx = wt_index(t)
                m[idx] = wt
            if not ok:
                for idx, val in old_m.items():
                    m[idx] = val
                continue
            ok_fwd, _ = _forward_q22_q64(Q, m, self.qc, symbols=dict(symbols))
            if ok_fwd:
                return True
            for idx, val in old_m.items():
                m[idx] = val

        Q[base + 9], Q[base + 10] = old_q9, old_q10
        for idx, val in old_m.items():
            m[idx] = val
        return False

    def _m11_from_Q(self, Q: QList, m: List[Optional[int]]) -> Optional[int]:
        Q12 = Q[3+12]; Q11 = Q[3+11]; Q10 = Q[3+10]; Q9 = Q[3+9]; Q8 = Q[3+8]
        t = 11
        if None in (Q12, Q11, Q10, Q9, Q8):
            return None
        f11 = ft(t, Q11, Q10, Q9)
        return (rr((Q12 - Q11) & MASK32, _RC[11]) - f11 - Q8 - _AC[11]) & MASK32

    def search(self, ihv: Tuple[int,int,int,int], max_restarts: int = 100) -> Optional[BlockResult]:
        # Strict Algorithm 6-2
        init_Q = _init_Q_from_ihv(ihv)
        ok_init, _ = self.qc.check_all(init_Q, base=3, start_t=-2, end_t=1)
        if not ok_init:
            return None
        for _ in range(max_restarts):
            Q = _init_Q_from_ihv(ihv)
            m: List[Optional[int]] = [None]*16
            symbols: Symbols = {}
            if not self.step1_choose_Q2_to_Q16(Q, symbols):
                continue
            if not self.step2_compute_m5_to_m15(Q, m):
                continue
            if not self.step3_loop_Q1_and_m0_to_m4(Q, m, symbols):
                continue
            if not self.step4_q9_q10_subspace(ihv, Q, m, symbols):
                continue
            words = _materialize_m(m)
            if words is None:
                continue
            ihv2, trace = compress_block(ihv, words)
            return BlockResult(ihv=ihv2, trace=trace, m_words=words)
        return None


def search_collision_full(
    seed: Optional[int] = None,
    max_restarts: int = 1000,
    ihv: Tuple[int, int, int, int] = MD5_IV,
    block1_restarts: int = 50,
    block2_restarts: int = 100,
) -> Optional[Tuple[BlockResult, BlockResult]]:
    rng = random.Random(seed)
    b1 = Block1FullSearcher(rng)
    for _ in range(max_restarts):
        r1 = b1.search(ihv, max_restarts=block1_restarts)
        if not r1:
            continue
        b2 = Block2FullSearcher(rng)
        r2 = b2.search(r1.ihv, max_restarts=block2_restarts)
        if r2:
            return (r1, r2)
    return None
