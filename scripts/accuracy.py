#!/usr/bin/env python3
"""Accuracy checks for core MD5 + algorithm scaffolding."""
from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from md5fastcoll.core import MD5_IV
from md5fastcoll.md5 import md5_hex
from md5fastcoll.inverse import demo_inverse_once
from md5fastcoll.conditions import minimal_block1_q_constraints, minimal_block2_q_constraints
from md5fastcoll.stevens_full import Block1FullSearcher, Block2FullSearcher


def check_md5_vectors() -> bool:
    vectors = [
        b"",
        b"a",
        b"abc",
        b"message digest",
        b"abcdefghijklmnopqrstuvwxyz",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    ]
    ok = True
    for msg in vectors:
        ours = md5_hex(msg)
        ref = hashlib.md5(msg).hexdigest()
        if ours != ref:
            print(f"MD5 mismatch: {msg!r} ours={ours} ref={ref}")
            ok = False
    print(f"md5_vectors: {'PASS' if ok else 'FAIL'}")
    return ok


def check_inverse() -> bool:
    ok, stats = demo_inverse_once()
    print(f"inverse_consistency: {'PASS' if ok else 'FAIL'} mismatches={stats['mismatches']}")
    return ok


def check_tables() -> bool:
    qc1 = minimal_block1_q_constraints()
    qc2 = minimal_block2_q_constraints()
    ok = True
    if len(qc1.conds) < 10:
        print("block1 conditions too few")
        ok = False
    if len(qc2.conds) < 10:
        print("block2 conditions too few")
        ok = False
    neg_keys = sorted([k for k in qc2.conds.keys() if k < 0])
    if neg_keys != [-2, -1]:
        print(f"block2 negative indices missing: {neg_keys}")
        ok = False
    print(f"table_load: {'PASS' if ok else 'FAIL'} (block1={len(qc1.conds)} block2={len(qc2.conds)})")
    return ok


def check_symbol_binding() -> bool:
    qc = minimal_block1_q_constraints()
    base = 3
    Q = [0] * (base + 65)
    for t, cond in qc.conds.items():
        if 46 <= t <= 63:
            head = cond.pattern[0]
            if head in ("I", "K"):
                Q[base + t] |= (1 << 31)
            elif head == "J":
                Q[base + t] &= ~(1 << 31)
    ok_consistent, _ = qc.check_all(Q, base=base, start_t=46, end_t=64)
    # flip one I-bit to force inconsistency
    for t, cond in qc.conds.items():
        if 46 <= t <= 63 and cond.pattern[0] in ("I", "K"):
            Q[base + t] &= ~(1 << 31)
            break
    ok_inconsistent, _ = qc.check_all(Q, base=base, start_t=46, end_t=64)
    ok = ok_consistent and not ok_inconsistent
    print(f"symbol_binding: {'PASS' if ok else 'FAIL'}")
    return ok


def check_block1_scaffold(trials: int, seed: int) -> bool:
    rng = random.Random(seed)
    searcher = Block1FullSearcher(rng)
    success = 0
    for _ in range(trials):
        Q = searcher._init_Q(MD5_IV)
        m = [None] * 16
        if not searcher.step1_choose_Qs(Q):
            continue
        if not searcher.step2_compute_m0_to_m15(Q, m):
            continue
        if not searcher.step3_loop_Q17_to_Q21(Q, m):
            continue
        success += 1
    ok = success > 0
    print(f"block1_step1_3: {'PASS' if ok else 'FAIL'} successes={success}/{trials}")
    return ok


def check_block2_scaffold(trials: int, seed: int) -> bool:
    rng = random.Random(seed)
    searcher = Block2FullSearcher(rng)
    sample_ihv = (0xC4DA537C, 0x1051DD8E, 0x42867DB3, 0x0D67B366)
    success = 0
    for _ in range(trials):
        Q = searcher._init_Q(sample_ihv)
        m = [None] * 16
        if not searcher.step1_choose_Q2_to_Q16(Q):
            continue
        if not searcher.step2_compute_m5_to_m15(Q, m):
            continue
        if not searcher.step3_loop_Q1_and_m0_to_m4(Q, m):
            continue
        success += 1
    ok = success > 0
    print(f"block2_step1_3: {'PASS' if ok else 'FAIL'} successes={success}/{trials}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ok = True
    ok &= check_md5_vectors()
    ok &= check_inverse()
    ok &= check_tables()
    ok &= check_symbol_binding()
    ok &= check_block1_scaffold(args.trials, args.seed)
    ok &= check_block2_scaffold(args.trials, args.seed)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
