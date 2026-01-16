from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import re
import pathlib

# Simple condition system for Qt bit patterns, matching the paper's symbols: '.', '0', '1', '^', '!'


@dataclass
class BitCond:
    # pattern is a 32-char string MSB..LSB consisting of . 0 1 ^ !
    pattern: str

    def __post_init__(self):
        if len(self.pattern) != 32:
            raise ValueError("pattern must be 32 chars")
        allowed = set(".01^!IJK")
        if any(c not in allowed for c in self.pattern):
            raise ValueError("invalid symbol in pattern")

    def check(self, q_curr: int, q_prev: Optional[int], symbols: Optional[Dict[str, Optional[int]]] = None) -> bool:
        for i, ch in enumerate(self.pattern):
            bit_index = 31 - i  # pattern shows [31]..[0]
            mask = 1 << bit_index
            b = 1 if (q_curr & mask) else 0
            if ch == ".":
                continue
            elif ch == "0" and b != 0:
                return False
            elif ch == "1" and b != 1:
                return False
            elif ch == "^":
                if q_prev is None:
                    return False
                pb = 1 if (q_prev & mask) else 0
                if b != pb:
                    return False
            elif ch == "!":
                if q_prev is None:
                    return False
                pb = 1 if (q_prev & mask) else 0
                if b == pb:
                    return False
            elif ch in ("I", "J", "K"):
                if symbols is None:
                    return False
                key = "I" if ch == "K" else ch
                current = symbols.get(key)
                if current is None:
                    symbols[key] = b
                elif b != current:
                    return False
        return True


@dataclass
class QConditions:
    # store per t a BitCond, and the chosen symbol bindings (I/J/K)
    conds: Dict[int, BitCond]
    symbols: Dict[str, Optional[int]] = field(default_factory=dict)

    def check_all(
        self,
        Q: List[int],
        base: int = 3,
        start_t: Optional[int] = None,
        end_t: Optional[int] = None,
    ) -> Tuple[bool, List[int]]:
        if start_t is None and end_t is None:
            keys = sorted(self.conds.keys())
        else:
            if start_t is None:
                start_t = 0
            if end_t is None:
                end_t = 64
            keys = [t for t in self.conds.keys() if start_t <= t < end_t]
        symbols = {"I": self.symbols.get("I"), "J": self.symbols.get("J")}
        bad: List[int] = []
        for t in keys:
            q_curr = Q[base + t]
            q_prev = Q[base + t - 1]
            if not self.conds[t].check(q_curr, q_prev, symbols):
                bad.append(t)
        return (len(bad) == 0), bad


# Loader for tables A-1..A-4 in a simple text format.
# Format example (spaces allowed between 8-bit chunks):
#   7: "00000011 11111110 11111000 00100000"
#   8: "0.0...00 ....1011 111....1 1...1..."
# Only symbols in {.,0,1,^,!} are allowed. Lines not in 0..64 are ignored.


def _normalize_pattern(pat: str) -> str:
    s = re.sub(r"\s+", "", pat.strip())
    if len(s) != 32:
        raise ValueError(f"pattern length != 32: {pat}")
    if not set(s) <= set(".01^!IJK"):
        raise ValueError(f"invalid symbols in pattern: {pat}")
    return s


def _merge_patterns(old: str, new: str) -> str:
    # Merge two 32-char patterns ('.','0','1','^','!'). New overrides '.' in old.
    out = []
    for oc, nc in zip(old, new):
        if nc == '.':
            out.append(oc)
        elif oc == '.':
            out.append(nc)
        else:
            # If both set and conflict (e.g., '0' vs '1'), prefer the new one
            # This is safe for our use-case where new adds specific 0/1
            out.append(nc if nc == oc else nc)
    return ''.join(out)


def load_qconditions_from_file(
    path: str | pathlib.Path,
    symbols: Optional[Dict[str, Optional[int]]] = None,
) -> QConditions:
    path = pathlib.Path(path)
    conds: Dict[int, BitCond] = {}
    if not path.exists():
        return QConditions(conds={}, symbols=symbols or {})
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        import re as _re
        m = _re.match(r"^(-?\d+)\s*:\s*\"([^\"]+)\"\s*$", line)
        if not m:
            continue
        t = int(m.group(1))
        if not (-2 <= t <= 64):
            continue
        pat = _normalize_pattern(m.group(2))
        if t in conds:
            merged = _merge_patterns(conds[t].pattern, pat)
            conds[t] = BitCond(merged)
        else:
            conds[t] = BitCond(pat)
    return QConditions(conds=conds, symbols=symbols or {})

# Minimal stubs to satisfy imports; to be filled with full Tables A-1..A-4

def _tables_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "tables"


def minimal_block1_q_constraints(symbols: Optional[Dict[str, Optional[int]]] = None) -> QConditions:
    return load_qconditions_from_file(_tables_dir() / "block1.txt", symbols=symbols)


def minimal_block2_q_constraints(symbols: Optional[Dict[str, Optional[int]]] = None) -> QConditions:
    return load_qconditions_from_file(_tables_dir() / "block2.txt", symbols=symbols)
