from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

from .md5_core import MASK32, RL, RR, add32, compress_block, wt_index, reverse_step_to_w
from .conditions import minimal_block1_q_constraints


@dataclass
class SearchResult:
    block: bytes
    trace: object


class Block1Searcher:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.constraints = minimal_block1_q_constraints()

    def random_block(self) -> bytes:
        return os.urandom(64)

    def try_block(self, iv: Tuple[int, int, int, int], block: bytes):
        trace = compress_block(iv, block)
        ok = self.constraints.check(trace.Q)
        return ok, trace

    def search(self, iv: Tuple[int, int, int, int], max_trials: int = 1000):
        for i in range(max_trials):
            blk = self.random_block()
            ok, tr = self.try_block(iv, blk)
            if ok:
                return SearchResult(block=blk, trace=tr)
        return None