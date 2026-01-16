from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

MASK32 = 0xFFFFFFFF


def u32(x: int) -> int:
    return x & MASK32


def RL(x: int, n: int) -> int:
    n &= 31
    x &= MASK32
    return ((x << n) | (x >> (32 - n))) & MASK32


def RR(x: int, n: int) -> int:
    n &= 31
    x &= MASK32
    return ((x >> n) | (x << (32 - n))) & MASK32


def add32(*xs: int) -> int:
    s = 0
    for v in xs:
        s = (s + (v & MASK32)) & MASK32
    return s


# ACt constants (MD5 T[1..64]) from RFC 1321
AC: List[int] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
]

# RCt rotation counts (per step)
RC: List[int] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
]


def ft(t: int, X: int, Y: int, Z: int) -> int:
    # MD5 boolean functions aligned with RFC 1321 (paper uses same functions).
    X &= MASK32
    Y &= MASK32
    Z &= MASK32
    if 0 <= t < 16:
        return ((X & Y) | (~X & Z)) & MASK32
    if 16 <= t < 32:
        return ((Z & X) | (~Z & Y)) & MASK32
    if 32 <= t < 48:
        return (X ^ Y ^ Z) & MASK32
    # 48..63
    return (Y ^ (X | ~Z)) & MASK32


# Wt schedule mapping t -> m-index

def wt_index(t: int) -> int:
    if 0 <= t < 16:
        return t
    if 16 <= t < 32:
        return (1 + 5 * t) & 15
    if 32 <= t < 48:
        return (5 + 3 * t) & 15
    return (7 * t) & 15


@dataclass
class Trace:
    Q: List[int]  # Q[-3..64] stored as list offset by +3
    T: List[int]  # T[0..63]
    W: List[int]  # m[0..15]

    def q(self, t: int) -> int:
        return self.Q[t + 3]


# MD5 IV in RFC order (A,B,C,D)
IV_RFC = (0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476)


def iv_to_Q(iv: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    # Paper uses: Q0=IHV1, Q-1=IHV2, Q-2=IHV3, Q-3=IHV0
    A, B, C, D = iv
    return (B, C, D, A)


def Q_to_iv(Qm3_Q0: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    # inverse of iv_to_Q
    Q0, Qm1, Qm2, Qm3 = Qm3_Q0  # actually given as (Q0,Q-1,Q-2,Q-3)
    # But our function will pass (Q0,Q-1,Q-2,Q-3) order; convert back to iv
    return (Qm3, Q0, Qm1, Qm2)


def compress_block(iv: Tuple[int, int, int, int], block: bytes) -> Trace:
    assert len(block) == 64
    m = [int.from_bytes(block[i:i+4], "little") for i in range(0, 64, 4)]

    # Record Q and T as in the paper; compute using standard RFC MD5 state update
    Q = [0] * 68  # Q[-3..64] with +3 offset
    Tvals = [0] * 64

    A, B, C, D = iv
    a, b, c, d = A, B, C, D

    # Initialize Q mapping
    Q[0] = A  # Q-3 = a
    Q[1] = D  # Q-2 = d
    Q[2] = C  # Q-1 = c
    Q[3] = B  # Q0  = b

    for t in range(64):
        if 0 <= t < 16:
            F = (b & c) | (~b & d)
        elif 16 <= t < 32:
            F = (d & b) | (~d & c)
        elif 32 <= t < 48:
            F = b ^ c ^ d
        else:
            F = c ^ (b | ~d)
        F &= MASK32
        g = wt_index(t)
        Tt = add32(a, F, AC[t], m[g])
        Tvals[t] = Tt & MASK32
        new_b = add32(b, RL(Tt, RC[t]))
        # record Q_{t+1} = new_b
        Q[t + 4] = new_b
        # rotate state for next iteration (preserve old b in c)
        a, b, c, d = d, new_b, b, c

    return Trace(Q=Q, T=Tvals, W=m)


def reverse_step_to_w(t: int, Q: List[int]) -> int:
    """Reverse step t to recover Wt from Q[-3..64] stored with +3 offset.
    """
    Qt = Q[t + 3]
    Qtm1 = Q[t + 2]
    Qtm2 = Q[t + 1]
    Qtm3 = Q[t + 0]
    Qtp1 = Q[t + 4]
    Rt = add32(Qtp1, -Qt)
    Tt = RR(Rt, RC[t])
    wt = add32(Tt, -ft(t, Qt, Qtm1, Qtm2), -Qtm3, -AC[t])
    return wt & MASK32


def recover_message_from_Q(Q: List[int]) -> List[int]:
    m = [0] * 16
    for t in range(64):
        idx = wt_index(t)
        m[idx] = reverse_step_to_w(t, Q)
    return m


# High-level digest with padding, for verification against hashlib

def md5_digest(msg: bytes, iv: Tuple[int, int, int, int] = IV_RFC) -> bytes:
    # padding: append 0x80, then zeros, then 64-bit length (little-endian)
    ml = len(msg)
    bit_len = (ml * 8) & ((1 << 64) - 1)
    pad = b"\x80" + b"\x00" * ((56 - (ml + 1)) % 64)
    msg_padded = msg + pad + bit_len.to_bytes(8, "little")

    A, B, C, D = iv
    off = 0
    while off < len(msg_padded):
        block = msg_padded[off:off + 64]
        trace = compress_block((A, B, C, D), block)
        # After 64 steps, per paper: IHV0=IV0+Q61, IHV1=IV1+Q64, IHV2=IV2+Q63, IHV3=IV3+Q62
        A = (A + trace.Q[61 + 3]) & MASK32
        B = (B + trace.Q[64 + 3]) & MASK32
        C = (C + trace.Q[63 + 3]) & MASK32
        D = (D + trace.Q[62 + 3]) & MASK32
        off += 64

    return (A.to_bytes(4, "little") + B.to_bytes(4, "little") +
            C.to_bytes(4, "little") + D.to_bytes(4, "little"))
