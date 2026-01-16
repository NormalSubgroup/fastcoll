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


def _bit(x: int, b: int) -> int:
    return (x >> b) & 1


def _choose_Q_with_constraints(
    rng: random.Random,
    Q: QList,
    qc: QConditions,
    t: int,
    max_tries: int = 4096,
) -> Optional[int]:
    base = 3
    if len(Q) <= base + t + 1:
        return None

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

    if next_q is not None and next_pat is not None:
        for i, ch in enumerate(next_pat):
            if ch not in "^!":
                continue
            b = 31 - i
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

    for _ in range(max_tries):
        qt = rng.getrandbits(32)
        qt = (qt & ~fixed_mask) | fixed_val
        if pat is not None and prev_q is not None:
            if not qc.conds[t].check(qt, prev_q):
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

    def _req(idx: int) -> int:
        val = Q[base + idx]
        if val is None:
            raise ValueError(f"Q[{idx}] is not set")
        return val & MASK32

    Qt = _req(i)
    Qtm1 = _req(i - 1)
    Qtm2 = _req(i - 2)
    Qtm3 = _req(i - 3)
    Qt1 = _req(i + 1)
    
    # Rt = Qt+1 - Qt (rotation after addition)
    Rt = (Qt1 - Qt) & MASK32
    # Tt = RR(Rt, RCi) (before rotation)
    Tt = rr(Rt, _RC[i]) & MASK32
    # Wi = Tt - fi - Qi-3 - ACi
    fi = ft(i, Qt, Qtm1, Qtm2) & MASK32
    Wi = (Tt - fi - Qtm3 - _AC[i]) & MASK32
    
    return Wi


def _forward_step_set_Q(Q: QList, m: List[Optional[int]], t: int, qc: Optional[QConditions] = None) -> bool:
    base = 3

    def _req(idx: int) -> int:
        val = Q[base + idx]
        if val is None:
            raise ValueError(f"Q[{idx}] is not set")
        return val & MASK32

    idx = wt_index(t)
    if idx >= len(m) or m[idx] is None:
        return False
    Qt = _req(t)
    Qtm1 = _req(t - 1)
    Qtm2 = _req(t - 2)
    Qtm3 = _req(t - 3)
    Tt = (ft(t, Qt, Qtm1, Qtm2) + Qtm3 + _AC[t] + (m[idx] & MASK32)) & MASK32
    Qt1 = (Qt + rl(Tt, _RC[t])) & MASK32
    if qc is not None and (t + 1) in qc.conds:
        if not qc.conds[t + 1].check(Qt1, Qt):
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

    def _init_Q(self, ihv: Tuple[int,int,int,int]) -> QList:
        base = 3
        Q: QList = [None] * (base + 65)
        IV0,IV1,IV2,IV3 = ihv
        Q[base-3]=IV0; Q[base-2]=IV3; Q[base-1]=IV2; Q[base+0]=IV1
        return Q

    def step1_choose_Qs(self, Q: QList) -> bool:
        # Algorithm 6-1 Step 1: Choose Q1, Q3, ..., Q16 fulfilling conditions
        # Note: We choose odd Qs first as per paper algorithm description
        base = 3
        for t in [1, 3, 5, 7, 9, 11, 13, 15]:
            qt = _choose_Q_with_constraints(self.rng, Q, self.qc, t)
            if qt is None:
                return False
            Q[base + t] = qt
        # Then choose even Qs Q4, Q6, Q8, Q10, Q12, Q14, Q16 (Q2 is computed later)
        for t in [4, 6, 8, 10, 12, 14, 16]:
            qt = _choose_Q_with_constraints(self.rng, Q, self.qc, t)
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

    def step3_loop_Q17_to_Q21(self, Q: QList, m: List[Optional[int]]) -> bool:
        # Algorithm 6-1 Step 3: Loop until Q17, ..., Q21 are fulfilling conditions
        # (a) Choose Q17 fulfilling conditions; (b) Calculate m1 at t=16
        # (c) Calculate Q2 and m2, m3, m4, m5; (d) Calculate Q18, ..., Q21
        base = 3
        for _ in range(4096):  # Increased attempts for better success rate
            # reset Q17..Q21 from previous attempts
            for t in range(17, 22):
                Q[base + t] = None
            # reset m1..m5 from previous attempts
            for idx in range(1, 6):
                m[idx] = None
            # (a) Choose Q17 fulfilling conditions
            qt17 = _choose_Q_with_constraints(self.rng, Q, self.qc, 17)
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
                if 2 in self.qc.conds and not self.qc.conds[2].check(Q2, Q1):
                    continue
                # Also enforce constraints from Q3 (next)
                q3 = Q[base + 3]
                if 3 in self.qc.conds:
                    if q3 is None or not self.qc.conds[3].check(q3, Q2):
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
                    if not _forward_step_set_Q(Q, m, t, self.qc):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok:
                return True
        return False

    def step4_q9_q10_subspace(self, ihv: Tuple[int,int,int,int], Q: QList, m: List[Optional[int]], budget: int = 1<<15) -> bool:
        # Keep m11 constant while exploring Q9/Q10 freedom. Baseline:
        base_m11 = m[11] if m[11] is not None else self._m11_from_Q(Q, m)
        if base_m11 is None:
            return False
        base = 3
        for _ in range(budget):
            old_q9, old_q10 = Q[base+9], Q[base+10]
            Q[base+10] = None
            q9 = _choose_Q_with_constraints(self.rng, Q, self.qc, 9)
            if q9 is None:
                Q[base+10] = old_q10
                continue
            Q[base+9] = q9
            q10 = _choose_Q_with_constraints(self.rng, Q, self.qc, 10)
            if q10 is None:
                Q[base+9] = old_q9
                Q[base+10] = old_q10
                continue
            old_m = {idx: m[idx] for idx in (8, 9, 10, 12, 13)}
            Q[base+9], Q[base+10] = q9, q10
            if self._m11_from_Q(Q, m) != base_m11:
                Q[base+9], Q[base+10] = old_q9, old_q10
                continue
            # Recompute affected m words deterministically via reverse steps t=8,9,10,12,13
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
                Q[base+9], Q[base+10] = old_q9, old_q10
                for idx, val in old_m.items():
                    m[idx] = val
                continue
            words = _materialize_m(m)
            if words is None:
                Q[base+9], Q[base+10] = old_q9, old_q10
                for idx, val in old_m.items():
                    m[idx] = val
                continue
            ihv2, trace = compress_block(ihv, words)
            ok_q, _ = self.qc.check_all(trace["Q"], base=3)
            ok_tail, _ = check_T_restrictions_tail(trace)
            ok_iv, _ = check_next_block_iv_conditions(ihv2)
            if ok_q and ok_tail and ok_iv:
                return True
            Q[base+9], Q[base+10] = old_q9, old_q10
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
            Q = self._init_Q(ihv)
            m: List[Optional[int]] = [None]*16
            if not self.step1_choose_Qs(Q):
                continue
            if not self.step2_compute_m0_to_m15(Q, m):
                continue
            if not self.step3_loop_Q17_to_Q21(Q, m):
                continue
            if not self.step4_q9_q10_subspace(ihv, Q, m):
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

    def _init_Q(self, ihv: Tuple[int,int,int,int]) -> QList:
        base = 3
        Q: QList = [None] * (base + 65)
        IV0,IV1,IV2,IV3 = ihv
        Q[base-3]=IV0; Q[base-2]=IV3; Q[base-1]=IV2; Q[base+0]=IV1
        return Q

    def step1_choose_Q2_to_Q16(self, Q: QList) -> bool:
        # Algorithm 6-2 Step 1: Choose Q2, ..., Q16 fulfilling conditions 
        base = 3
        for t in range(2, 17):
            qt = _choose_Q_with_constraints(self.rng, Q, self.qc, t)
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

    def step3_loop_Q1_and_m0_to_m4(self, Q: QList, m: List[Optional[int]]) -> bool:
        # Algorithm 6-2 Step 3: Loop until Q17, ..., Q21 are fulfilling conditions
        # (a) Choose Q1 fulfilling conditions; (b) Calculate m0, ..., m4
        # (c) Calculate Q17, ..., Q21
        base = 3
        for _ in range(4096):
            Q[base + 1] = None
            for t in range(17, 22):
                Q[base + t] = None
            for idx in range(0, 5):
                m[idx] = None
            # (a) Choose Q1 fulfilling conditions
            q1 = _choose_Q_with_constraints(self.rng, Q, self.qc, 1)
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
                    if not _forward_step_set_Q(Q, m, t, self.qc):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok:
                return True
        return False

    def step4_q9_q10_subspace(self, ihv: Tuple[int,int,int,int], Q: QList, m: List[Optional[int]], budget: int = 1<<15) -> bool:
        # Same idea: keep f11 constant via Q11 bits relationship
        base_m11 = m[11] if m[11] is not None else self._m11_from_Q(Q, m)
        if base_m11 is None:
            return False
        base = 3
        for _ in range(budget):
            old_q9, old_q10 = Q[base+9], Q[base+10]
            Q[base+10] = None
            q9 = _choose_Q_with_constraints(self.rng, Q, self.qc, 9)
            if q9 is None:
                Q[base+10] = old_q10
                continue
            Q[base+9] = q9
            q10 = _choose_Q_with_constraints(self.rng, Q, self.qc, 10)
            if q10 is None:
                Q[base+9] = old_q9
                Q[base+10] = old_q10
                continue
            old_m = {idx: m[idx] for idx in (8, 9, 10, 12, 13)}
            Q[base+9], Q[base+10] = q9, q10
            if self._m11_from_Q(Q, m) != base_m11:
                Q[base+9], Q[base+10] = old_q9, old_q10
                continue
            # Update affected m words deterministically (t=8,9,10,12,13)
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
                Q[base+9], Q[base+10] = old_q9, old_q10
                for idx, val in old_m.items():
                    m[idx] = val
                continue
            words = _materialize_m(m)
            if words is None:
                Q[base+9], Q[base+10] = old_q9, old_q10
                for idx, val in old_m.items():
                    m[idx] = val
                continue
            ihv2, trace = compress_block(ihv, words)
            ok_q, _ = self.qc.check_all(trace["Q"], base=3)
            ok_tail, _ = check_T_restrictions_tail(trace)
            if ok_q and ok_tail:
                return True
            Q[base+9], Q[base+10] = old_q9, old_q10
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
        init_Q = self._init_Q(ihv)
        ok_init, _ = self.qc.check_all(init_Q, base=3, start_t=-2, end_t=1)
        if not ok_init:
            return None
        for _ in range(max_restarts):
            Q = self._init_Q(ihv)
            m: List[Optional[int]] = [None]*16
            if not self.step1_choose_Q2_to_Q16(Q):
                continue
            if not self.step2_compute_m5_to_m15(Q, m):
                continue
            if not self.step3_loop_Q1_and_m0_to_m4(Q, m):
                continue
            if not self.step4_q9_q10_subspace(ihv, Q, m):
                continue
            words = _materialize_m(m)
            if words is None:
                continue
            ihv2, trace = compress_block(ihv, words)
            return BlockResult(ihv=ihv2, trace=trace, m_words=words)
        return None


def search_collision_full(seed: Optional[int] = None, max_restarts: int = 1000) -> Optional[Tuple[BlockResult, BlockResult]]:
    rng = random.Random(seed)
    b1 = Block1FullSearcher(rng)
    for _ in range(max_restarts):
        r1 = b1.search(MD5_IV, max_restarts=50)
        if not r1:
            continue
        b2 = Block2FullSearcher(rng)
        r2 = b2.search(r1.ihv, max_restarts=100)
        if r2:
            return (r1, r2)
    return None
