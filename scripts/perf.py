#!/usr/bin/env python3
"""Performance micro-benchmarks for the Stevens search scaffolding."""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from md5fastcoll.core import MD5_IV
from md5fastcoll.stevens_full import Block1FullSearcher, Block2FullSearcher


def bench_block1_step1_3(trials: int, seed: int) -> None:
    rng = random.Random(seed)
    searcher = Block1FullSearcher(rng)
    start = time.time()
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
    elapsed = time.time() - start
    rate = trials / elapsed if elapsed else 0.0
    print(f"block1_step1_3: trials={trials} success={success} time={elapsed:.3f}s rate={rate:.2f}/s")


def bench_block2_step1_3(trials: int, seed: int) -> None:
    rng = random.Random(seed)
    searcher = Block2FullSearcher(rng)
    sample_ihv = (0x12345678, 0x87654321, 0x02000000, 0x00000000)
    start = time.time()
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
    elapsed = time.time() - start
    rate = trials / elapsed if elapsed else 0.0
    print(f"block2_step1_3: trials={trials} success={success} time={elapsed:.3f}s rate={rate:.2f}/s")


def bench_block1_step4(budget: int, seed: int) -> None:
    rng = random.Random(seed)
    searcher = Block1FullSearcher(rng)
    # find a configuration where step3 succeeds
    for _ in range(50):
        Q = searcher._init_Q(MD5_IV)
        m = [None] * 16
        if not searcher.step1_choose_Qs(Q):
            continue
        if not searcher.step2_compute_m0_to_m15(Q, m):
            continue
        if searcher.step3_loop_Q17_to_Q21(Q, m):
            break
    else:
        print("block1_step4: failed to reach step3")
        return
    start = time.time()
    ok = searcher.step4_q9_q10_subspace(MD5_IV, Q, m, budget=budget)
    elapsed = time.time() - start
    rate = budget / elapsed if elapsed else 0.0
    print(f"block1_step4: budget={budget} ok={ok} time={elapsed:.3f}s rate={rate:.2f} it/s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--budget", type=int, default=1 << 12)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    bench_block1_step1_3(args.trials, args.seed)
    bench_block2_step1_3(args.trials, args.seed)
    bench_block1_step4(args.budget, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
