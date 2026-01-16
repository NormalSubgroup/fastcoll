from __future__ import annotations

import math
from typing import List, Tuple, Dict

MASK32 = 0xFFFFFFFF


def u32(x: int) -> int:
    return x & MASK32


def rl(x: int, s: int) -> int:
    x &= MASK32
    return ((x << s) | (x >> (32 - s))) & MASK32


def rr(x: int, s: int) -> int:
    x &= MASK32
    return ((x >> s) | (x << (32 - s))) & MASK32


# Rotation constants (RC_t) per MD5 specification
_RC: List[int] = (
    [7, 12, 17, 22] * 4
    + [5, 9, 14, 20] * 4
    + [4, 11, 16, 23] * 4
    + [6, 10, 15, 21] * 4
)


def _mk_AC() -> List[int]:
    # AC_t = floor(2^32 * abs(sin(t+1)))
    return [int(abs(math.sin(i + 1)) * (1 << 32)) & MASK32 for i in range(64)]


_AC: List[int] = _mk_AC()


def ft(t: int, X: int, Y: int, Z: int) -> int:
    # MD5 boolean functions as specified in the paper
    # Note: Paper uses different notation but same functions
    X, Y, Z = u32(X), u32(Y), u32(Z)
    if 0 <= t < 16:
        # F function: (X ∧ Y) ⊕ (X̄ ∧ Z) 
        return u32((X & Y) | ((~X) & Z))
    if 16 <= t < 32:
        # G function: (Z ∧ X) ⊕ (Z̄ ∧ Y)
        return u32((Z & X) | ((~Z) & Y))
    if 32 <= t < 48:
        # H function: X ⊕ Y ⊕ Z
        return u32(X ^ Y ^ Z)
    if 48 <= t < 64:
        # I function: Y ⊕ (X ∨ Z̄)
        return u32(Y ^ (X | (~Z)))
    raise ValueError("t out of range")


def wt_index(t: int) -> int:
    if 0 <= t < 16:
        return t
    if 16 <= t < 32:
        return (5 * t + 1) % 16
    if 32 <= t < 48:
        return (3 * t + 5) % 16
    if 48 <= t < 64:
        return (7 * t) % 16
    raise ValueError("t out of range")


def compress_block(
    ihv: Tuple[int, int, int, int],
    m: List[int],
) -> Tuple[Tuple[int, int, int, int], Dict[str, List[int]]]:
    """
    MD5 compression in the Stevens paper notation.
    Inputs:
      - ihv: (IV0, IV1, IV2, IV3)
      - m: 16 little-endian 32-bit words
    Returns:
      - new_ihv: tuple
      - trace: dict with arrays Q[-3..64], T[0..63], R[0..63], W[0..63]
    """
    if len(m) != 16:
        raise ValueError("m must have 16 words")

    IV0, IV1, IV2, IV3 = (u32(ihv[0]), u32(ihv[1]), u32(ihv[2]), u32(ihv[3]))

    base = 3
    # allocate enough space for indices -3..64
    Q = [0] * (base + 65)
    # Initialize Q[-3..0]
    Q[base - 3] = IV0  # Q[-3]
    Q[base - 2] = IV3  # Q[-2]
    Q[base - 1] = IV2  # Q[-1]
    Q[base + 0] = IV1  # Q[0]

    T = [0] * 64
    R = [0] * 64
    W = [0] * 64

    for t in range(64):
        # read needed Q's
        Qt = Q[base + t]
        Qt_1 = Q[base + t - 1]
        Qt_2 = Q[base + t - 2]
        Qt_3 = Q[base + t - 3]

        k = wt_index(t)
        W[t] = m[k]

        Tt = u32(ft(t, Qt, Qt_1, Qt_2) + Qt_3 + _AC[t] + W[t])
        Rt = rl(Tt, _RC[t])
        Qt1 = u32(Qt + Rt)

        T[t] = Tt
        R[t] = Rt
        Q[base + t + 1] = Qt1

    # Map back to new ihv as given in the paper
    new_IHV0 = u32(IV0 + Q[base + 61])
    new_IHV1 = u32(IV1 + Q[base + 64])
    new_IHV2 = u32(IV2 + Q[base + 63])
    new_IHV3 = u32(IV3 + Q[base + 62])
    trace = {
        "Q": Q,  # includes negative indices offset by +3
        "T": T,
        "R": R,
        "W": W,
        "RC": list(_RC),
        "AC": list(_AC),
    }
    return (new_IHV0, new_IHV1, new_IHV2, new_IHV3), trace


# MD5 initial value
MD5_IV = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476)


def md5_padding(msg_len_bytes: int) -> bytes:
    bit_len = (msg_len_bytes * 8) & ((1 << 64) - 1)
    # 0x80 then zeros then length (little-endian 64-bit)
    pad = b"\x80"
    # k such that (msg_len + 1 + k) % 64 == 56
    k = (56 - (msg_len_bytes + 1) % 64) % 64
    pad += b"\x00" * k
    pad += bit_len.to_bytes(8, "little")
    return pad


def bytes_to_words_le(block: bytes) -> List[int]:
    assert len(block) == 64
    return [int.from_bytes(block[i : i + 4], "little") for i in range(0, 64, 4)]


def words_to_bytes_le(words: List[int]) -> bytes:
    return b"".join(u32(w).to_bytes(4, "little") for w in words)