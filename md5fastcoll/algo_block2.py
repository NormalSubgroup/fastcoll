from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from .md5_core import compress_block
from .conditions import minimal_block2_q_constraints


@dataclass
class SearchResult:
    block: bytes
    trace: object


class Block2Searcher:
    def __init__(self):
        self.constraints = minimal_block2_q_constraints()

    def try_block(self, iv: Tuple[int, int, int, int], block: bytes):
        trace = compress_block(iv, block)
        ok = self.constraints.check(trace.Q)
        return ok, trace