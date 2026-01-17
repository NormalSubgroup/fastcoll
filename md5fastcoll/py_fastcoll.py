from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import List, Tuple

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None
else:  # pragma: no cover
    if os.getenv("MD5FASTCOLL_NO_NUMPY") == "1":
        _np = None
    else:
        # Numpy warns on expected uint32 wraparound; silence for MD5-style arithmetic.
        import warnings as _warnings

        _warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*overflow encountered.*")

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
QOFF = 3


_NUMBA_TRY_BLOCK0_Q9MASK = None


def _get_numba_try_block0_q9mask():
    global _NUMBA_TRY_BLOCK0_Q9MASK
    if _NUMBA_TRY_BLOCK0_Q9MASK is not None:
        return _NUMBA_TRY_BLOCK0_Q9MASK
    if os.getenv("MD5FASTCOLL_NO_NUMBA") == "1":
        return None
    try:
        from .numba_fastcoll import numba_available

        if not numba_available():
            return None
        from .numba_fastcoll import _try_block0_q9mask  # type: ignore[attr-defined]

        _NUMBA_TRY_BLOCK0_Q9MASK = _try_block0_q9mask
        return _NUMBA_TRY_BLOCK0_Q9MASK
    except Exception:  # pragma: no cover
        return None


def u32(x: int) -> int:
    return x & MASK32


def rol(x: int, n: int) -> int:
    x &= MASK32
    return ((x << n) | (x >> (32 - n))) & MASK32


def ror(x: int, n: int) -> int:
    x &= MASK32
    return ((x >> n) | (x << (32 - n))) & MASK32


@dataclass(slots=True)
class XorShift64:
    """HashClash `xrng64()` (32-bit output xorshift with 64-bit state)."""

    seed32_1: int
    seed32_2: int

    def next_u32(self) -> int:
        s1 = self.seed32_1 & MASK32
        s2 = self.seed32_2 & MASK32
        t = (s1 ^ ((s1 << 10) & MASK32)) & MASK32
        s1 = s2
        s2 = ((s2 ^ (s2 >> 10)) ^ (t ^ (t >> 13))) & MASK32
        self.seed32_1 = s1
        self.seed32_2 = s2
        return s1


def rng_from_seed(seed: int | None) -> XorShift64:
    if seed is None:
        s = time.time_ns() & MASK64
        seed1 = s & MASK32
        seed2 = (s >> 32) & MASK32
    else:
        seed1 = seed & MASK32
        seed2 = (seed >> 32) & MASK32
        if seed1 == 0 and seed2 == 0:
            seed2 = 0x12345678
    if seed1 == 0 and seed2 == 0:
        seed2 = 0x12345678
    return XorShift64(seed32_1=seed1, seed32_2=seed2)


def FF(b: int, c: int, d: int) -> int:
    return (d ^ (b & (c ^ d))) & MASK32


def GG(b: int, c: int, d: int) -> int:
    return (c ^ (d & (b ^ c))) & MASK32


def HH(b: int, c: int, d: int) -> int:
    return (b ^ c ^ d) & MASK32


def II(b: int, c: int, d: int) -> int:
    return (c ^ (b | (~d & MASK32))) & MASK32


# --- Minimal MD5 compression (no trace allocations) ---

_MD5_AC: Tuple[int, ...] = (
    0xD76AA478,
    0xE8C7B756,
    0x242070DB,
    0xC1BDCEEE,
    0xF57C0FAF,
    0x4787C62A,
    0xA8304613,
    0xFD469501,
    0x698098D8,
    0x8B44F7AF,
    0xFFFF5BB1,
    0x895CD7BE,
    0x6B901122,
    0xFD987193,
    0xA679438E,
    0x49B40821,
    0xF61E2562,
    0xC040B340,
    0x265E5A51,
    0xE9B6C7AA,
    0xD62F105D,
    0x02441453,
    0xD8A1E681,
    0xE7D3FBC8,
    0x21E1CDE6,
    0xC33707D6,
    0xF4D50D87,
    0x455A14ED,
    0xA9E3E905,
    0xFCEFA3F8,
    0x676F02D9,
    0x8D2A4C8A,
    0xFFFA3942,
    0x8771F681,
    0x6D9D6122,
    0xFDE5380C,
    0xA4BEEA44,
    0x4BDECFA9,
    0xF6BB4B60,
    0xBEBFBC70,
    0x289B7EC6,
    0xEAA127FA,
    0xD4EF3085,
    0x04881D05,
    0xD9D4D039,
    0xE6DB99E5,
    0x1FA27CF8,
    0xC4AC5665,
    0xF4292244,
    0x432AFF97,
    0xAB9423A7,
    0xFC93A039,
    0x655B59C3,
    0x8F0CCC92,
    0xFFEFF47D,
    0x85845DD1,
    0x6FA87E4F,
    0xFE2CE6E0,
    0xA3014314,
    0x4E0811A1,
    0xF7537E82,
    0xBD3AF235,
    0x2AD7D2BB,
    0xEB86D391,
)

_MD5_RC: Tuple[int, ...] = (
    7,
    12,
    17,
    22,
    7,
    12,
    17,
    22,
    7,
    12,
    17,
    22,
    7,
    12,
    17,
    22,
    5,
    9,
    14,
    20,
    5,
    9,
    14,
    20,
    5,
    9,
    14,
    20,
    5,
    9,
    14,
    20,
    4,
    11,
    16,
    23,
    4,
    11,
    16,
    23,
    4,
    11,
    16,
    23,
    4,
    11,
    16,
    23,
    6,
    10,
    15,
    21,
    6,
    10,
    15,
    21,
    6,
    10,
    15,
    21,
    6,
    10,
    15,
    21,
)

_MD5_G: Tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    1,
    6,
    11,
    0,
    5,
    10,
    15,
    4,
    9,
    14,
    3,
    8,
    13,
    2,
    7,
    12,
    5,
    8,
    11,
    14,
    1,
    4,
    7,
    10,
    13,
    0,
    3,
    6,
    9,
    12,
    15,
    2,
    0,
    7,
    14,
    5,
    12,
    3,
    10,
    1,
    8,
    15,
    6,
    13,
    4,
    11,
    2,
    9,
)


def md5_compress(ihv: Tuple[int, int, int, int], block: List[int]) -> Tuple[int, int, int, int]:
    """MD5 compression on exactly 1 block (16 little-endian words)."""
    if len(block) != 16:
        raise ValueError("block must have 16 words")

    a0, b0, c0, d0 = (ihv[0] & MASK32, ihv[1] & MASK32, ihv[2] & MASK32, ihv[3] & MASK32)
    a, b, c, d = a0, b0, c0, d0

    for i in range(64):
        if i < 16:
            f = d ^ (b & (c ^ d))
        elif i < 32:
            f = c ^ (d & (b ^ c))
        elif i < 48:
            f = b ^ c ^ d
        else:
            f = c ^ (b | (~d & MASK32))

        tmp = (a + f + _MD5_AC[i] + (block[_MD5_G[i]] & MASK32)) & MASK32
        tmp = ((tmp << _MD5_RC[i]) | (tmp >> (32 - _MD5_RC[i]))) & MASK32
        tmp = (tmp + b) & MASK32
        a, d, c, b = d, c, b, tmp

    return ((a0 + a) & MASK32, (b0 + b) & MASK32, (c0 + c) & MASK32, (d0 + d) & MASK32)


# --- Mask tables (precomputed once) ---


def _mk_masks(count: int, expr) -> Tuple[int, ...]:
    return tuple(u32(expr(k)) for k in range(count))


_B0_Q4MASK = _mk_masks(1 << 4, lambda k: (((k << 2) ^ (k << 26)) & 0x38000004))
_B0_Q9Q10MASK = _mk_masks(1 << 3, lambda k: (((k << 13) ^ (k << 4)) & 0x2060))
_B0_Q9MASK = _mk_masks(
    1 << 16,
    lambda k: (((k << 1) ^ (k << 2) ^ (k << 5) ^ (k << 7) ^ (k << 8) ^ (k << 10) ^ (k << 11) ^ (k << 13)) & 0x0EB94F16),
)

_W_Q4MASK = _mk_masks(1 << 6, lambda k: (((k << 13) ^ (k << 19)) & 0x01C0E000))
_W_Q9MASK = _mk_masks(1 << 5, lambda k: ((((k << 5) ^ (k << 13) ^ (k << 17) ^ (k << 24)) & 0x00084000)))
_W_Q10MASK = _mk_masks(1 << 5, lambda k: ((((k << 5) ^ (k << 13) ^ (k << 17) ^ (k << 24)) & 0x18000020)))
_W_Q9MASK2 = _mk_masks(1 << 10, lambda k: (((k << 1) ^ (k << 7) ^ (k << 14) ^ (k << 15) ^ (k << 22)) & 0x6074041C))

_S00_Q9Q10MASK = _mk_masks(1 << 3, lambda k: (((k << 5) ^ (k << 12) ^ (k << 25)) & 0x08002020))
_S00_Q9MASK = _mk_masks(1 << 9, lambda k: (((k << 1) ^ (k << 3) ^ (k << 6) ^ (k << 8) ^ (k << 11) ^ (k << 14) ^ (k << 18)) & 0x04310D12))

_S01_Q9Q10MASK = _mk_masks(1 << 5, lambda k: (((k << 4) ^ (k << 11) ^ (k << 24) ^ (k << 27)) & 0x88002030))
_S01_Q9MASK = _mk_masks(1 << 9, lambda k: (((k << 1) ^ (k << 7) ^ (k << 9) ^ (k << 12) ^ (k << 15) ^ (k << 19) ^ (k << 22)) & 0x44310D02))

_S10_Q9Q10MASK = _mk_masks(1 << 4, lambda k: (((k << 2) ^ (k << 8) ^ (k << 11) ^ (k << 25)) & 0x08004204))
_S10_Q9MASK = _mk_masks(
    1 << 10,
    lambda k: (((k << 1) ^ (k << 2) ^ (k << 3) ^ (k << 7) ^ (k << 12) ^ (k << 15) ^ (k << 18) ^ (k << 20)) & 0x2471042A),
)

_S11_Q9Q10MASK = _mk_masks(1 << 5, lambda k: (((k << 5) ^ (k << 6) ^ (k << 7) ^ (k << 24) ^ (k << 27)) & 0x880002A0))
_S11_Q9MASK = _mk_masks(1 << 9, lambda k: (((k << 1) ^ (k << 3) ^ (k << 8) ^ (k << 12) ^ (k << 15) ^ (k << 18)) & 0x04710C12))

if _np is not None:
    _NP_U32 = _np.uint32
    _B0_Q9MASK_NP = _np.array(_B0_Q9MASK, dtype=_NP_U32)
    _W_Q9MASK2_NP = _np.array(_W_Q9MASK2, dtype=_NP_U32)
    _S00_Q9MASK_NP = _np.array(_S00_Q9MASK, dtype=_NP_U32)
    _S01_Q9MASK_NP = _np.array(_S01_Q9MASK, dtype=_NP_U32)
    _S10_Q9MASK_NP = _np.array(_S10_Q9MASK, dtype=_NP_U32)
    _S11_Q9MASK_NP = _np.array(_S11_Q9MASK, dtype=_NP_U32)


def _rol_u32_np(x, n: int):
    # Works for numpy uint32 arrays/scalars; keeps uint32 wraparound semantics.
    return (x << n) | (x >> (32 - n))


def _ror_u32_np(x, n: int):
    return (x >> n) | (x << (32 - n))


def _ff_u32_np(b, c, d):
    return d ^ (b & (c ^ d))


def _gg_u32_np(b, c, d):
    return c ^ (d & (b ^ c))


def _hh_u32_np(b, c, d):
    return b ^ c ^ d


def _ii_u32_np(b, c, d):
    return c ^ (b | (~d))


def _try_block0_q9mask_numpy(
    *,
    IV0: int,
    IV1: int,
    IV2: int,
    IV3: int,
    q7: int,
    q8: int,
    q9_base: int,
    q10: int,
    q11: int,
    q12: int,
    tt8: int,
    tt9: int,
    tt12: int,
    aa: int,
    bb: int,
    cc: int,
    dd: int,
    block: List[int],
) -> List[int] | None:
    """
    Vectorized replacement for the hot `for msk in q9mask:` loop in `find_block0`.

    Only `m[8]`, `m[9]`, `m[12]` vary across the 2^16 q9 masks; everything else
    is fixed by the outer search state. We evaluate the MD5 tail conditions for
    all masks in numpy and return the first block that passes full verification.
    """
    if _np is None:  # pragma: no cover
        return None

    # Bring scalars into uint32 domain once, and silence expected uint32 wraparound.
    U32 = _NP_U32
    with _np.errstate(over="ignore"):
        q7_u = U32(q7)
        q8_u = U32(q8)
        q9_base_u = U32(q9_base)
        q10_u = U32(q10)
        q11_u = U32(q11)
        q12_u = U32(q12)
        tt8_u = U32(tt8)
        tt9_u = U32(tt9)
        tt12_u = U32(tt12)
        aa_u = U32(aa)
        bb_u = U32(bb)
        cc_u = U32(cc)
        dd_u = U32(dd)

        # Precompute varying q9 and derived message words.
        q9 = q9_base_u ^ _B0_Q9MASK_NP
        ff_q12_q11_q10 = _ff_u32_np(q12_u, q11_u, q10_u)
        m12 = tt12_u - ff_q12_q11_q10 - q9

        m8 = _ror_u32_np(q9 - q8_u, 7) - tt8_u
        m9 = _ror_u32_np(q10_u - q9, 12) - _ff_u32_np(q9, q8_u, q7_u) - tt9_u

    # MD5 tail (steps 24..63) + IHV constraints.
    b0 = U32(block[0])
    b1 = U32(block[1])
    b2 = U32(block[2])
    b3 = U32(block[3])
    b4 = U32(block[4])
    b5 = U32(block[5])
    b6 = U32(block[6])
    b7 = U32(block[7])
    b10 = U32(block[10])
    b11 = U32(block[11])
    b13 = U32(block[13])
    b14 = U32(block[14])
    b15 = U32(block[15])

    a = _rol_u32_np(aa_u + _gg_u32_np(bb_u, cc_u, dd_u) + m9 + U32(0x21E1CDE6), 5) + bb_u
    d = _rol_u32_np(dd_u + _gg_u32_np(a, bb_u, cc_u) + b14 + U32(0xC33707D6), 9) + a
    c = _rol_u32_np(cc_u + _gg_u32_np(d, a, bb_u) + b3 + U32(0xF4D50D87), 14) + d
    b = _rol_u32_np(bb_u + _gg_u32_np(c, d, a) + m8 + U32(0x455A14ED), 20) + c
    a = _rol_u32_np(a + _gg_u32_np(b, c, d) + b13 + U32(0xA9E3E905), 5) + b
    d = _rol_u32_np(d + _gg_u32_np(a, b, c) + b2 + U32(0xFCEFA3F8), 9) + a
    c = _rol_u32_np(c + _gg_u32_np(d, a, b) + b7 + U32(0x676F02D9), 14) + d
    b = _rol_u32_np(b + _gg_u32_np(c, d, a) + m12 + U32(0x8D2A4C8A), 20) + c

    a = _rol_u32_np(a + _hh_u32_np(b, c, d) + b5 + U32(0xFFFA3942), 4) + b
    d = _rol_u32_np(d + _hh_u32_np(a, b, c) + m8 + U32(0x8771F681), 11) + a
    c = c + _hh_u32_np(d, a, b) + b11 + U32(0x6D9D6122)

    # HH step34 requires bit15 == 0 for block0.
    good = (c & U32(1 << 15)) == U32(0)
    if not bool(good.any()):
        return None

    q9 = q9[good]
    m8 = m8[good]
    m9 = m9[good]
    m12 = m12[good]
    a = a[good]
    b = b[good]
    c = c[good]
    d = d[good]

    c = _rol_u32_np(c, 16) + d
    b = _rol_u32_np(b + _hh_u32_np(c, d, a) + b14 + U32(0xFDE5380C), 23) + c

    a = _rol_u32_np(a + _hh_u32_np(b, c, d) + b1 + U32(0xA4BEEA44), 4) + b
    d = _rol_u32_np(d + _hh_u32_np(a, b, c) + b4 + U32(0x4BDECFA9), 11) + a
    c = _rol_u32_np(c + _hh_u32_np(d, a, b) + b7 + U32(0xF6BB4B60), 16) + d
    b = _rol_u32_np(b + _hh_u32_np(c, d, a) + b10 + U32(0xBEBFBC70), 23) + c
    a = _rol_u32_np(a + _hh_u32_np(b, c, d) + b13 + U32(0x289B7EC6), 4) + b
    d = _rol_u32_np(d + _hh_u32_np(a, b, c) + b0 + U32(0xEAA127FA), 11) + a
    c = _rol_u32_np(c + _hh_u32_np(d, a, b) + b3 + U32(0xD4EF3085), 16) + d
    b = _rol_u32_np(b + _hh_u32_np(c, d, a) + b6 + U32(0x04881D05), 23) + c
    a = _rol_u32_np(a + _hh_u32_np(b, c, d) + m9 + U32(0xD9D4D039), 4) + b
    d = _rol_u32_np(d + _hh_u32_np(a, b, c) + m12 + U32(0xE6DB99E5), 11) + a
    c = _rol_u32_np(c + _hh_u32_np(d, a, b) + b15 + U32(0x1FA27CF8), 16) + d
    b = _rol_u32_np(b + _hh_u32_np(c, d, a) + b2 + U32(0xC4AC5665), 23) + c

    good = ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    q9 = q9[good]
    m8 = m8[good]
    m9 = m9[good]
    m12 = m12[good]
    a = a[good]
    b = b[good]
    c = c[good]
    d = d[good]

    # Step 48..63 (II): keep a predicate mask instead of repeatedly slicing arrays.
    good = _np.ones(a.shape, dtype=bool)

    a = _rol_u32_np(a + _ii_u32_np(b, c, d) + b0 + U32(0xF4292244), 6) + b
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    d = _rol_u32_np(d + _ii_u32_np(a, b, c) + b7 + U32(0x432AFF97), 10) + a
    good &= ((b ^ d) & U32(0x80000000)) != U32(0)
    if not bool(good.any()):
        return None

    c = _rol_u32_np(c + _ii_u32_np(d, a, b) + b14 + U32(0xAB9423A7), 15) + d
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    b = _rol_u32_np(b + _ii_u32_np(c, d, a) + b5 + U32(0xFC93A039), 21) + c
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    a = _rol_u32_np(a + _ii_u32_np(b, c, d) + m12 + U32(0x655B59C3), 6) + b
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    d = _rol_u32_np(d + _ii_u32_np(a, b, c) + b3 + U32(0x8F0CCC92), 10) + a
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    c = _rol_u32_np(c + _ii_u32_np(d, a, b) + b10 + U32(0xFFEFF47D), 15) + d
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    b = _rol_u32_np(b + _ii_u32_np(c, d, a) + b1 + U32(0x85845DD1), 21) + c
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    a = _rol_u32_np(a + _ii_u32_np(b, c, d) + m8 + U32(0x6FA87E4F), 6) + b
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    d = _rol_u32_np(d + _ii_u32_np(a, b, c) + b15 + U32(0xFE2CE6E0), 10) + a
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    c = _rol_u32_np(c + _ii_u32_np(d, a, b) + b6 + U32(0xA3014314), 15) + d
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    b = _rol_u32_np(b + _ii_u32_np(c, d, a) + b13 + U32(0x4E0811A1), 21) + c
    good &= ((b ^ d) & U32(0x80000000)) != U32(0)
    if not bool(good.any()):
        return None

    a = _rol_u32_np(a + _ii_u32_np(b, c, d) + b4 + U32(0xF7537E82), 6) + b
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    d = _rol_u32_np(d + _ii_u32_np(a, b, c) + b11 + U32(0xBD3AF235), 10) + a
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    c = _rol_u32_np(c + _ii_u32_np(d, a, b) + b2 + U32(0x2AD7D2BB), 15) + d
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return None

    b = _rol_u32_np(b + _ii_u32_np(c, d, a) + m9 + U32(0xEB86D391), 21) + c

    IHV1 = b + U32(IV1)
    IHV2 = c + U32(IV2)
    IHV3 = d + U32(IV3)

    wang = ((IHV2 ^ IHV1) & U32(0x86000000)) == U32(0x02000000)
    wang &= ((IHV1 ^ IHV3) & U32(0x82000000)) == U32(0)
    wang &= (IHV1 & U32(0x06000020)) == U32(0)

    stevens = ((IHV1 ^ IHV2) & U32(0x80000000)) == U32(0)
    stevens &= ((IHV1 ^ IHV3) & U32(0x80000000)) == U32(0)
    stevens &= (IHV1 & U32(1 << 25)) == U32(0)
    stevens &= (IHV2 & U32(1 << 25)) == U32(0)
    stevens &= (IHV3 & U32(1 << 25)) == U32(0)
    stevens &= ((IHV2 ^ IHV1) & U32(1)) == U32(0)

    good &= wang | stevens
    if not bool(good.any()):
        return None

    # Full compression verification on surviving candidates (usually very few).
    for idx in _np.flatnonzero(good):
        block[8] = int(m8[idx])
        block[9] = int(m9[idx])
        block[12] = int(m12[idx])

        msg2_block = apply_delta_block0(block)
        out1 = md5_compress((IV0, IV1, IV2, IV3), block)
        out2 = md5_compress((IV0, IV1, IV2, IV3), msg2_block)
        if (
            out2[0] == ((out1[0] + (1 << 31)) & MASK32)
            and out2[1] == ((out1[1] + (1 << 31) + (1 << 25)) & MASK32)
            and out2[2] == ((out1[2] + (1 << 31) + (1 << 25)) & MASK32)
            and out2[3] == ((out1[3] + (1 << 31) + (1 << 25)) & MASK32)
        ):
            return list(block)

    return None


# --- Pure-Python SWAR ("big-int SIMD") helpers ---

_SWAR_STRIDE = 64  # bits per lane (32 data + 32 spacer)
_SWAR_MASK32 = 0xFFFFFFFF

_swar_const_cache: dict[int, tuple[int, int]] = {}


def _swar_consts(lanes: int) -> tuple[int, int]:
    """
    Return `(one_mask, u32_mask)` for `lanes` packed uint32 lanes.

    Representation:
      - lane i occupies bits [i*64 .. i*64+31]
      - bits [i*64+32 .. i*64+63] are zero spacer bits
    """
    cached = _swar_const_cache.get(lanes)
    if cached is not None:
        return cached
    # 0x01 followed by 7x 0x00 per lane (little-endian) yields bit0 set for each lane.
    one = int.from_bytes(b"\x01\x00\x00\x00\x00\x00\x00\x00" * lanes, "little")
    mask32 = one * _SWAR_MASK32
    _swar_const_cache[lanes] = (one, mask32)
    return (one, mask32)


def _swar_pack_u32(values: List[int] | tuple[int, ...]) -> int:
    import struct

    # 8 bytes per lane: 4 bytes little-endian u32 value, then 4 zero spacer bytes.
    lanes = len(values)
    ba = bytearray(lanes * 8)
    for i, v in enumerate(values):
        struct.pack_into("<I", ba, i * 8, int(v) & _SWAR_MASK32)
    return int.from_bytes(ba, "little")


_SWAR_B0_Q9_LANES = 1024
_swar_b0_q9_batches: List[int] | None = None


def _swar_b0_q9mask_batches() -> List[int]:
    global _swar_b0_q9_batches
    if _swar_b0_q9_batches is not None:
        return _swar_b0_q9_batches
    masks = _B0_Q9MASK
    lanes = _SWAR_B0_Q9_LANES
    if len(masks) % lanes != 0:
        raise ValueError("unexpected _B0_Q9MASK length for SWAR batching")
    batches: List[int] = []
    for off in range(0, len(masks), lanes):
        batches.append(_swar_pack_u32(masks[off : off + lanes]))
    _swar_b0_q9_batches = batches
    return batches


def _try_block0_q9mask_swar(
    *,
    IV0: int,
    IV1: int,
    IV2: int,
    IV3: int,
    q7: int,
    q8: int,
    q9_base: int,
    q10: int,
    q11: int,
    q12: int,
    tt8: int,
    tt9: int,
    tt12: int,
    aa: int,
    bb: int,
    cc: int,
    dd: int,
    block: List[int],
) -> List[int] | None:
    """
    Pure-Python replacement for `_try_block0_q9mask_numpy` using packed big-int lanes.

    This avoids the `for msk in q9mask:` 2^16 Python loop without requiring numpy.
    """
    lanes = _SWAR_B0_Q9_LANES
    one, mask32 = _swar_consts(lanes)

    # Broadcast scalars into lanes.
    q7_p = (q7 & _SWAR_MASK32) * one
    q8_p = (q8 & _SWAR_MASK32) * one
    q9_base_p = (q9_base & _SWAR_MASK32) * one
    q10_p = (q10 & _SWAR_MASK32) * one
    q11_p = (q11 & _SWAR_MASK32) * one
    q12_p = (q12 & _SWAR_MASK32) * one
    tt8_p = (tt8 & _SWAR_MASK32) * one
    tt9_p = (tt9 & _SWAR_MASK32) * one
    tt12_p = (tt12 & _SWAR_MASK32) * one
    aa_p = (aa & _SWAR_MASK32) * one
    bb_p = (bb & _SWAR_MASK32) * one
    cc_p = (cc & _SWAR_MASK32) * one
    dd_p = (dd & _SWAR_MASK32) * one

    # Fixed block words (broadcast once).
    b0_p = (block[0] & _SWAR_MASK32) * one
    b1_p = (block[1] & _SWAR_MASK32) * one
    b2_p = (block[2] & _SWAR_MASK32) * one
    b3_p = (block[3] & _SWAR_MASK32) * one
    b4_p = (block[4] & _SWAR_MASK32) * one
    b5_p = (block[5] & _SWAR_MASK32) * one
    b6_p = (block[6] & _SWAR_MASK32) * one
    b7_p = (block[7] & _SWAR_MASK32) * one
    b10_p = (block[10] & _SWAR_MASK32) * one
    b11_p = (block[11] & _SWAR_MASK32) * one
    b13_p = (block[13] & _SWAR_MASK32) * one
    b14_p = (block[14] & _SWAR_MASK32) * one
    b15_p = (block[15] & _SWAR_MASK32) * one

    # Local helpers (inline-ish).
    def _add(x: int, y: int) -> int:
        return (x + y) & mask32

    def _not(x: int) -> int:
        return x ^ mask32

    def _neg(x: int) -> int:
        return _add(_not(x), one)

    def _sub(x: int, y: int) -> int:
        return _add(x, _neg(y))

    def _rol(x: int, n: int) -> int:
        return ((x << n) & mask32) | ((x >> (32 - n)) & mask32)

    def _ror(x: int, n: int) -> int:
        return ((x >> n) & mask32) | ((x << (32 - n)) & mask32)

    def _ff(b: int, c: int, d: int) -> int:
        return d ^ (b & (c ^ d))

    def _gg(b: int, c: int, d: int) -> int:
        return c ^ (d & (b ^ c))

    def _hh(b: int, c: int, d: int) -> int:
        return b ^ c ^ d

    def _ii(b: int, c: int, d: int) -> int:
        return c ^ (b | _not(d))

    # Precompute fixed FF(q12,q11,q10).
    ff_q12_q11_q10_p = _ff(q12_p, q11_p, q10_p)

    # Iterate over packed q9 mask batches.
    for batch_idx, q9mask_p in enumerate(_swar_b0_q9mask_batches()):
        q9_p = q9_base_p ^ q9mask_p

        m12_p = _sub(_sub(tt12_p, ff_q12_q11_q10_p), q9_p)
        m8_p = _sub(_ror(_sub(q9_p, q8_p), 7), tt8_p)
        m9_p = _sub(_sub(_ror(_sub(q10_p, q9_p), 12), _ff(q9_p, q8_p, q7_p)), tt9_p)

        # MD5 tail (steps 24..63) + IHV constraints.
        a = _add(_rol(_add(_add(_add(aa_p, _gg(bb_p, cc_p, dd_p)), m9_p), (0x21E1CDE6 * one)), 5), bb_p)
        d = _add(_rol(_add(_add(_add(dd_p, _gg(a, bb_p, cc_p)), b14_p), (0xC33707D6 * one)), 9), a)
        c = _add(_rol(_add(_add(_add(cc_p, _gg(d, a, bb_p)), b3_p), (0xF4D50D87 * one)), 14), d)
        b = _add(_rol(_add(_add(_add(bb_p, _gg(c, d, a)), m8_p), (0x455A14ED * one)), 20), c)
        a = _add(_rol(_add(_add(_add(a, _gg(b, c, d)), b13_p), (0xA9E3E905 * one)), 5), b)
        d = _add(_rol(_add(_add(_add(d, _gg(a, b, c)), b2_p), (0xFCEFA3F8 * one)), 9), a)
        c = _add(_rol(_add(_add(_add(c, _gg(d, a, b)), b7_p), (0x676F02D9 * one)), 14), d)
        b = _add(_rol(_add(_add(_add(b, _gg(c, d, a)), m12_p), (0x8D2A4C8A * one)), 20), c)

        a = _add(_rol(_add(_add(_add(a, _hh(b, c, d)), b5_p), (0xFFFA3942 * one)), 4), b)
        d = _add(_rol(_add(_add(_add(d, _hh(a, b, c)), m8_p), (0x8771F681 * one)), 11), a)
        c = _add(_add(_add(c, _hh(d, a, b)), b11_p), (0x6D9D6122 * one))

        # HH step34 requires bit15 == 0 for block0.
        good = (((c >> 15) & one) ^ one)
        if good == 0:
            continue

        c = _add(_rol(c, 16), d)
        b = _add(_rol(_add(_add(_add(b, _hh(c, d, a)), b14_p), (0xFDE5380C * one)), 23), c)

        a = _add(_rol(_add(_add(_add(a, _hh(b, c, d)), b1_p), (0xA4BEEA44 * one)), 4), b)
        d = _add(_rol(_add(_add(_add(d, _hh(a, b, c)), b4_p), (0x4BDECFA9 * one)), 11), a)
        c = _add(_rol(_add(_add(_add(c, _hh(d, a, b)), b7_p), (0xF6BB4B60 * one)), 16), d)
        b = _add(_rol(_add(_add(_add(b, _hh(c, d, a)), b10_p), (0xBEBFBC70 * one)), 23), c)
        a = _add(_rol(_add(_add(_add(a, _hh(b, c, d)), b13_p), (0x289B7EC6 * one)), 4), b)
        d = _add(_rol(_add(_add(_add(d, _hh(a, b, c)), b0_p), (0xEAA127FA * one)), 11), a)
        c = _add(_rol(_add(_add(_add(c, _hh(d, a, b)), b3_p), (0xD4EF3085 * one)), 16), d)
        b = _add(_rol(_add(_add(_add(b, _hh(c, d, a)), b6_p), (0x04881D05 * one)), 23), c)
        a = _add(_rol(_add(_add(_add(a, _hh(b, c, d)), m9_p), (0xD9D4D039 * one)), 4), b)
        d = _add(_rol(_add(_add(_add(d, _hh(a, b, c)), m12_p), (0xE6DB99E5 * one)), 11), a)
        c = _add(_rol(_add(_add(_add(c, _hh(d, a, b)), b15_p), (0x1FA27CF8 * one)), 16), d)
        b = _add(_rol(_add(_add(_add(b, _hh(c, d, a)), b2_p), (0xC4AC5665 * one)), 23), c)

        good &= (((b ^ d) >> 31) & one) ^ one
        if good == 0:
            continue

        a = _add(_rol(_add(_add(_add(a, _ii(b, c, d)), b0_p), (0xF4292244 * one)), 6), b)
        good &= (((a ^ c) >> 31) & one) ^ one
        if good == 0:
            continue
        d = _add(_rol(_add(_add(_add(d, _ii(a, b, c)), b7_p), (0x432AFF97 * one)), 10), a)
        good &= ((b ^ d) >> 31) & one
        if good == 0:
            continue
        c = _add(_rol(_add(_add(_add(c, _ii(d, a, b)), b14_p), (0xAB9423A7 * one)), 15), d)
        good &= (((a ^ c) >> 31) & one) ^ one
        if good == 0:
            continue
        b = _add(_rol(_add(_add(_add(b, _ii(c, d, a)), b5_p), (0xFC93A039 * one)), 21), c)
        good &= (((b ^ d) >> 31) & one) ^ one
        if good == 0:
            continue
        a = _add(_rol(_add(_add(_add(a, _ii(b, c, d)), m12_p), (0x655B59C3 * one)), 6), b)
        good &= (((a ^ c) >> 31) & one) ^ one
        if good == 0:
            continue
        d = _add(_rol(_add(_add(_add(d, _ii(a, b, c)), b3_p), (0x8F0CCC92 * one)), 10), a)
        good &= (((b ^ d) >> 31) & one) ^ one
        if good == 0:
            continue
        c = _add(_rol(_add(_add(_add(c, _ii(d, a, b)), b10_p), (0xFFEFF47D * one)), 15), d)
        good &= (((a ^ c) >> 31) & one) ^ one
        if good == 0:
            continue
        b = _add(_rol(_add(_add(_add(b, _ii(c, d, a)), b1_p), (0x85845DD1 * one)), 21), c)
        good &= (((b ^ d) >> 31) & one) ^ one
        if good == 0:
            continue
        a = _add(_rol(_add(_add(_add(a, _ii(b, c, d)), m8_p), (0x6FA87E4F * one)), 6), b)
        good &= (((a ^ c) >> 31) & one) ^ one
        if good == 0:
            continue
        d = _add(_rol(_add(_add(_add(d, _ii(a, b, c)), b15_p), (0xFE2CE6E0 * one)), 10), a)
        good &= (((b ^ d) >> 31) & one) ^ one
        if good == 0:
            continue
        c = _add(_rol(_add(_add(_add(c, _ii(d, a, b)), b6_p), (0xA3014314 * one)), 15), d)
        good &= (((a ^ c) >> 31) & one) ^ one
        if good == 0:
            continue
        b = _add(_rol(_add(_add(_add(b, _ii(c, d, a)), b13_p), (0x4E0811A1 * one)), 21), c)
        good &= ((b ^ d) >> 31) & one
        if good == 0:
            continue
        a = _add(_rol(_add(_add(_add(a, _ii(b, c, d)), b4_p), (0xF7537E82 * one)), 6), b)
        good &= (((a ^ c) >> 31) & one) ^ one
        if good == 0:
            continue
        d = _add(_rol(_add(_add(_add(d, _ii(a, b, c)), b11_p), (0xBD3AF235 * one)), 10), a)
        good &= (((b ^ d) >> 31) & one) ^ one
        if good == 0:
            continue
        c = _add(_rol(_add(_add(_add(c, _ii(d, a, b)), b2_p), (0x2AD7D2BB * one)), 15), d)
        good &= (((a ^ c) >> 31) & one) ^ one
        if good == 0:
            continue
        b = _add(_rol(_add(_add(_add(b, _ii(c, d, a)), m9_p), (0xEB86D391 * one)), 21), c)

        IHV1_p = _add(b, (IV1 & _SWAR_MASK32) * one)
        IHV2_p = _add(c, (IV2 & _SWAR_MASK32) * one)
        IHV3_p = _add(d, (IV3 & _SWAR_MASK32) * one)

        # Wang conditions expressed as bit tests.
        t21 = IHV2_p ^ IHV1_p
        wang = ((t21 >> 25) & one)  # bit25 == 1
        wang &= (((t21 >> 26) & one) ^ one)  # bit26 == 0
        wang &= (((t21 >> 31) & one) ^ one)  # bit31 == 0

        t13 = IHV1_p ^ IHV3_p
        wang &= (((t13 >> 25) & one) ^ one)
        wang &= (((t13 >> 31) & one) ^ one)

        wang &= (((IHV1_p >> 5) & one) ^ one)  # bit5 == 0
        wang &= (((IHV1_p >> 25) & one) ^ one)
        wang &= (((IHV1_p >> 26) & one) ^ one)

        # Stevens conditions expressed as bit tests.
        stevens = ((((IHV1_p ^ IHV2_p) >> 31) & one) ^ one)
        stevens &= ((((IHV1_p ^ IHV3_p) >> 31) & one) ^ one)
        stevens &= (((IHV1_p >> 25) & one) ^ one)
        stevens &= (((IHV2_p >> 25) & one) ^ one)
        stevens &= (((IHV3_p >> 25) & one) ^ one)
        stevens &= ((t21 & one) ^ one)  # bit0 == 0

        good &= wang | stevens
        if good == 0:
            continue

        # Full compression verification on surviving candidates.
        batch_base = batch_idx * lanes
        good_bits = good
        while good_bits:
            lsb = good_bits & -good_bits
            bitpos = lsb.bit_length() - 1
            lane = bitpos // _SWAR_STRIDE
            shift = lane * _SWAR_STRIDE
            m8 = (m8_p >> shift) & _SWAR_MASK32
            m9 = (m9_p >> shift) & _SWAR_MASK32
            m12 = (m12_p >> shift) & _SWAR_MASK32

            block[8] = int(m8)
            block[9] = int(m9)
            block[12] = int(m12)

            msg2_block = apply_delta_block0(block)
            out1 = md5_compress((IV0, IV1, IV2, IV3), block)
            out2 = md5_compress((IV0, IV1, IV2, IV3), msg2_block)
            if (
                out2[0] == ((out1[0] + (1 << 31)) & MASK32)
                and out2[1] == ((out1[1] + (1 << 31) + (1 << 25)) & MASK32)
                and out2[2] == ((out1[2] + (1 << 31) + (1 << 25)) & MASK32)
                and out2[3] == ((out1[3] + (1 << 31) + (1 << 25)) & MASK32)
            ):
                return list(block)

            good_bits ^= lsb

    return None


def _tail_rounds_common_mask_numpy(
    a0: int,
    b0: int,
    c0: int,
    d0: int,
    block: List[int],
    *,
    m8,
    m9,
    m12,
    hh_step34_bit15_required: int,
):
    """Vectorized `_tail_rounds_common` predicate over candidate arrays."""
    if _np is None:  # pragma: no cover
        return None

    U32 = _NP_U32
    a0_u = U32(a0)
    b0_u = U32(b0)
    c0_u = U32(c0)
    d0_u = U32(d0)

    w0 = U32(block[0])
    w1 = U32(block[1])
    w2 = U32(block[2])
    w3 = U32(block[3])
    w4 = U32(block[4])
    w5 = U32(block[5])
    w6 = U32(block[6])
    w7 = U32(block[7])
    w10 = U32(block[10])
    w11 = U32(block[11])
    w13 = U32(block[13])
    w14 = U32(block[14])
    w15 = U32(block[15])

    # Step 24..31 (GG)
    a = _rol_u32_np(a0_u + _gg_u32_np(b0_u, c0_u, d0_u) + m9 + U32(0x21E1CDE6), 5) + b0_u
    d = _rol_u32_np(d0_u + _gg_u32_np(a, b0_u, c0_u) + w14 + U32(0xC33707D6), 9) + a
    c = _rol_u32_np(c0_u + _gg_u32_np(d, a, b0_u) + w3 + U32(0xF4D50D87), 14) + d
    b = _rol_u32_np(b0_u + _gg_u32_np(c, d, a) + m8 + U32(0x455A14ED), 20) + c
    a = _rol_u32_np(a + _gg_u32_np(b, c, d) + w13 + U32(0xA9E3E905), 5) + b
    d = _rol_u32_np(d + _gg_u32_np(a, b, c) + w2 + U32(0xFCEFA3F8), 9) + a
    c = _rol_u32_np(c + _gg_u32_np(d, a, b) + w7 + U32(0x676F02D9), 14) + d
    b = _rol_u32_np(b + _gg_u32_np(c, d, a) + m12 + U32(0x8D2A4C8A), 20) + c

    # Step 32..47 (HH)
    a = _rol_u32_np(a + _hh_u32_np(b, c, d) + w5 + U32(0xFFFA3942), 4) + b
    d = _rol_u32_np(d + _hh_u32_np(a, b, c) + m8 + U32(0x8771F681), 11) + a
    c = c + _hh_u32_np(d, a, b) + w11 + U32(0x6D9D6122)

    need_bit = U32(hh_step34_bit15_required & 1)
    good = ((c >> 15) & U32(1)) == need_bit
    if not bool(good.any()):
        return good

    c = _rol_u32_np(c, 16) + d
    b = _rol_u32_np(b + _hh_u32_np(c, d, a) + w14 + U32(0xFDE5380C), 23) + c

    a = _rol_u32_np(a + _hh_u32_np(b, c, d) + w1 + U32(0xA4BEEA44), 4) + b
    d = _rol_u32_np(d + _hh_u32_np(a, b, c) + w4 + U32(0x4BDECFA9), 11) + a
    c = _rol_u32_np(c + _hh_u32_np(d, a, b) + w7 + U32(0xF6BB4B60), 16) + d
    b = _rol_u32_np(b + _hh_u32_np(c, d, a) + w10 + U32(0xBEBFBC70), 23) + c
    a = _rol_u32_np(a + _hh_u32_np(b, c, d) + w13 + U32(0x289B7EC6), 4) + b
    d = _rol_u32_np(d + _hh_u32_np(a, b, c) + w0 + U32(0xEAA127FA), 11) + a
    c = _rol_u32_np(c + _hh_u32_np(d, a, b) + w3 + U32(0xD4EF3085), 16) + d
    b = _rol_u32_np(b + _hh_u32_np(c, d, a) + w6 + U32(0x04881D05), 23) + c
    a = _rol_u32_np(a + _hh_u32_np(b, c, d) + m9 + U32(0xD9D4D039), 4) + b
    d = _rol_u32_np(d + _hh_u32_np(a, b, c) + m12 + U32(0xE6DB99E5), 11) + a
    c = _rol_u32_np(c + _hh_u32_np(d, a, b) + w15 + U32(0x1FA27CF8), 16) + d
    b = _rol_u32_np(b + _hh_u32_np(c, d, a) + w2 + U32(0xC4AC5665), 23) + c

    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good

    # Step 48..63 (II)
    a = _rol_u32_np(a + _ii_u32_np(b, c, d) + w0 + U32(0xF4292244), 6) + b
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    d = _rol_u32_np(d + _ii_u32_np(a, b, c) + w7 + U32(0x432AFF97), 10) + a
    good &= ((b ^ d) & U32(0x80000000)) != U32(0)
    if not bool(good.any()):
        return good
    c = _rol_u32_np(c + _ii_u32_np(d, a, b) + w14 + U32(0xAB9423A7), 15) + d
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    b = _rol_u32_np(b + _ii_u32_np(c, d, a) + w5 + U32(0xFC93A039), 21) + c
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    a = _rol_u32_np(a + _ii_u32_np(b, c, d) + m12 + U32(0x655B59C3), 6) + b
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    d = _rol_u32_np(d + _ii_u32_np(a, b, c) + w3 + U32(0x8F0CCC92), 10) + a
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    c = _rol_u32_np(c + _ii_u32_np(d, a, b) + w10 + U32(0xFFEFF47D), 15) + d
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    b = _rol_u32_np(b + _ii_u32_np(c, d, a) + w1 + U32(0x85845DD1), 21) + c
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    a = _rol_u32_np(a + _ii_u32_np(b, c, d) + m8 + U32(0x6FA87E4F), 6) + b
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    d = _rol_u32_np(d + _ii_u32_np(a, b, c) + w15 + U32(0xFE2CE6E0), 10) + a
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    c = _rol_u32_np(c + _ii_u32_np(d, a, b) + w6 + U32(0xA3014314), 15) + d
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    b = _rol_u32_np(b + _ii_u32_np(c, d, a) + w13 + U32(0x4E0811A1), 21) + c
    good &= ((b ^ d) & U32(0x80000000)) != U32(0)
    if not bool(good.any()):
        return good
    a = _rol_u32_np(a + _ii_u32_np(b, c, d) + w4 + U32(0xF7537E82), 6) + b
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    d = _rol_u32_np(d + _ii_u32_np(a, b, c) + w11 + U32(0xBD3AF235), 10) + a
    good &= ((b ^ d) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    c = _rol_u32_np(c + _ii_u32_np(d, a, b) + w2 + U32(0x2AD7D2BB), 15) + d
    good &= ((a ^ c) & U32(0x80000000)) == U32(0)
    if not bool(good.any()):
        return good
    b = _rol_u32_np(b + _ii_u32_np(c, d, a) + m9 + U32(0xEB86D391), 21) + c
    return good


def _try_block1_q9_variants_numpy(
    *,
    q9_base: int,
    q9_masks_np,
    q5: int,
    q6: int,
    q7: int,
    q8: int,
    q10: int,
    q11: int,
    q12: int,
    q13: int,
    a0: int,
    b0: int,
    c0: int,
    d0: int,
    block: List[int],
    hh_step34_bit15_required: int,
    iv1: Tuple[int, int, int, int],
    iv2: Tuple[int, int, int, int],
    delta_apply,
) -> List[int] | None:
    """Vectorized q9-mask loop used in block1 (Wang/Stevens) generators."""
    if _np is None:  # pragma: no cover
        return None

    U32 = _NP_U32
    q9 = U32(q9_base) ^ q9_masks_np
    q5_u = U32(q5)
    q6_u = U32(q6)
    q7_u = U32(q7)
    q8_u = U32(q8)
    q10_u = U32(q10)
    q11_u = U32(q11)
    q12_u = U32(q12)
    q13_u = U32(q13)

    # reverse_step_first_round for t=8,9,12 (only q9 varies)
    m8 = _ror_u32_np(q9 - q8_u, 7) - (_ff_u32_np(q8_u, q7_u, q6_u) + q5_u + U32(0x698098D8))
    m9 = _ror_u32_np(q10_u - q9, 12) - (_ff_u32_np(q9, q8_u, q7_u) + q6_u + U32(0x8B44F7AF))
    t12 = _ror_u32_np(q13_u - q12_u, 7) - (_ff_u32_np(q12_u, q11_u, q10_u) + U32(0x6B901122))
    m12 = t12 - q9

    good = _tail_rounds_common_mask_numpy(
        a0,
        b0,
        c0,
        d0,
        block,
        m8=m8,
        m9=m9,
        m12=m12,
        hh_step34_bit15_required=hh_step34_bit15_required,
    )
    if good is None or not bool(good.any()):
        return None

    for idx in _np.flatnonzero(good):
        block[8] = int(m8[idx])
        block[9] = int(m9[idx])
        block[12] = int(m12[idx])

        out1 = md5_compress(iv1, block)
        out2 = md5_compress(iv2, delta_apply(block))
        if out1 == out2:
            return list(block)
    return None


def _batch_find_q1_chain_numpy(
    rng: XorShift64,
    *,
    q1a: int,
    q1_mask: int,
    q2: int,
    q0: int,
    q_minus1: int,
    tt1: int,
    q16: int,
    q15: int,
    tt17: int,
    tt18: int,
    tt19: int,
    tt0: int,
    q17_xor_mask: int,
    q17_xor_expect: int,
    q17_forbid_mask: int,
    q18_xor_mask: int,
    q18_xor_expect: int,
    q19_and_mask: int,
    q19_and_expect: int,
    q20_xor_mask: int,
    q20_xor_expect: int,
    tries: int = 1 << 12,
    want_q21: bool = False,
    block5: int = 0,
    q21_xor_mask: int = 0,
    q21_xor_expect: int = 0,
):
    """Vectorized replacement for the q1 search loops (batch over `tries` RNG outputs)."""
    if _np is None:  # pragma: no cover
        return None

    U32 = _NP_U32

    q2_u = U32(q2)
    q0_u = U32(q0)
    qm1_u = U32(q_minus1)
    tt1_u = U32(tt1)
    q16_u = U32(q16)
    q15_u = U32(q15)
    tt17_u = U32(tt17)
    tt18_u = U32(tt18)
    tt19_u = U32(tt19)
    tt0_u = U32(tt0)

    q1a_u = U32(q1a)
    q1_mask_u = U32(q1_mask)

    # Process RNG outputs in chunks, so we can stop early like the scalar loop.
    chunk = 256
    while True:
        if int(tries) <= 0:
            return None
        n = min(chunk, int(tries))
        tries = int(tries) - n

        r = _np.array([rng.next_u32() for _ in range(n)], dtype=U32)
        q1 = (q1a_u | (r & q1_mask_u)) & U32(MASK32)

        # m1 + GG chain to q20 (mirrors the inner `while counter < 1<<12` loops)
        m1 = _ror_u32_np(q2_u - q1, 12) - _ff_u32_np(q1, q0_u, qm1_u) - tt1_u
        q17 = _rol_u32_np(tt17_u + m1, 5) + q16_u

        ok = ((q17 ^ q16_u) & U32(q17_xor_mask)) == U32(q17_xor_expect)
        ok &= (q17 & U32(q17_forbid_mask)) == U32(0)
        if not bool(ok.any()):
            continue

        q18 = _rol_u32_np(_gg_u32_np(q17, q16_u, q15_u) + tt18_u, 9) + q17
        ok &= ((q18 ^ q17) & U32(q18_xor_mask)) == U32(q18_xor_expect)
        if not bool(ok.any()):
            continue

        q19 = _rol_u32_np(_gg_u32_np(q18, q17, q16_u) + tt19_u, 14) + q18
        ok &= (q19 & U32(q19_and_mask)) == U32(q19_and_expect)
        if not bool(ok.any()):
            continue

        m0 = _ror_u32_np(q1 - q0_u, 7) - tt0_u
        q20 = _rol_u32_np(_gg_u32_np(q19, q18, q17) + q16_u + U32(0xE9B6C7AA) + m0, 20) + q19
        ok &= ((q20 ^ q19) & U32(q20_xor_mask)) == U32(q20_xor_expect)
        if not bool(ok.any()):
            continue

        if want_q21:
            q21 = _rol_u32_np(_gg_u32_np(q20, q19, q18) + q17 + U32(0xD62F105D) + U32(block5), 5) + q20
            ok &= ((q21 ^ q20) & U32(q21_xor_mask)) == U32(q21_xor_expect)
            if not bool(ok.any()):
                continue

        idx = int(_np.flatnonzero(ok)[0])
        if want_q21:
            return (
                int(q1[idx]),
                int(m0[idx]),
                int(m1[idx]),
                int(q17[idx]),
                int(q18[idx]),
                int(q19[idx]),
                int(q20[idx]),
                int(q21[idx]),
            )
        return (
            int(q1[idx]),
            int(m0[idx]),
            int(m1[idx]),
            int(q17[idx]),
            int(q18[idx]),
            int(q19[idx]),
            int(q20[idx]),
        )


def reverse_step_first_round(Q: List[int], t: int, ac: int, rc: int) -> int:
    """HashClash `MD5_REVERSE_STEP` macro, for round-1 steps (t=0..15)."""
    Rt = (Q[QOFF + t + 1] - Q[QOFF + t]) & MASK32
    Tt = ror(Rt, rc)
    ff = FF(Q[QOFF + t], Q[QOFF + t - 1], Q[QOFF + t - 2])
    return (Tt - ff - Q[QOFF + t - 3] - (ac & MASK32)) & MASK32


def apply_delta_block0(block: List[int]) -> List[int]:
    out = list(block)
    out[4] = (out[4] + (1 << 31)) & MASK32
    out[11] = (out[11] + (1 << 15)) & MASK32
    out[14] = (out[14] + (1 << 31)) & MASK32
    return out


def apply_delta_block1(block: List[int]) -> List[int]:
    out = list(block)
    out[4] = (out[4] + (1 << 31)) & MASK32
    out[11] = (out[11] - (1 << 15)) & MASK32
    out[14] = (out[14] + (1 << 31)) & MASK32
    return out


def find_block0(rng: XorShift64, IV: Tuple[int, int, int, int]) -> List[int]:
    """HashClash `find_block0`."""
    IV0, IV1, IV2, IV3 = (u32(IV[0]), u32(IV[1]), u32(IV[2]), u32(IV[3]))
    Q = [0] * 68
    Q[0], Q[1], Q[2], Q[3] = IV0, IV3, IV2, IV1
    block = [0] * 16

    q4mask = _B0_Q4MASK
    q9q10mask = _B0_Q9Q10MASK
    q9mask = _B0_Q9MASK
    xrng = rng.next_u32

    # q9mask engine selection:
    # - "numpy" (default): vectorized NumPy implementation (fast, deterministic).
    # - "swar": pure-Python SWAR batching (no NumPy needed).
    # - "numba": experimental JIT scan loop (may change which valid block is found first).
    q9_engine = os.getenv("MD5FASTCOLL_BLOCK0_Q9_ENGINE", "numpy").strip().lower()
    if q9_engine not in ("numpy", "swar", "numba"):
        q9_engine = "numpy"

    try_block0_q9mask_numba = None
    iv_u32_np = None
    block_u32_np = None
    if q9_engine == "numba" and _np is not None:
        try_block0_q9mask_numba = _get_numba_try_block0_q9mask()
        if try_block0_q9mask_numba is not None:
            iv_u32_np = _np.array([IV0, IV1, IV2, IV3], dtype=_NP_U32)
            block_u32_np = _np.zeros(16, dtype=_NP_U32)

    while True:
        Q[QOFF + 1] = xrng()
        Q[QOFF + 3] = (xrng() & 0xFE87BC3F) | 0x017841C0
        Q[QOFF + 4] = (xrng() & 0x44000033) | 0x000002C0 | (Q[QOFF + 3] & 0x0287BC00)
        Q[QOFF + 5] = 0x41FFFFC8 | (Q[QOFF + 4] & 0x04000033)
        Q[QOFF + 6] = 0xB84B82D6
        Q[QOFF + 7] = (xrng() & 0x68000084) | 0x02401B43
        Q[QOFF + 8] = (xrng() & 0x2B8F6E04) | 0x005090D3 | ((~Q[QOFF + 7]) & 0x40000000)
        Q[QOFF + 9] = 0x20040068 | (Q[QOFF + 8] & 0x00020000) | ((~Q[QOFF + 8]) & 0x40000000)
        Q[QOFF + 10] = (xrng() & 0x40000000) | 0x1040B089
        Q[QOFF + 11] = (xrng() & 0x10408008) | 0x0FBB7F16 | ((~Q[QOFF + 10]) & 0x40000000)
        Q[QOFF + 12] = (xrng() & 0x1ED9DF7F) | 0x00022080 | ((~Q[QOFF + 11]) & 0x40200000)
        Q[QOFF + 13] = (xrng() & 0x5EFB4F77) | 0x20049008
        Q[QOFF + 14] = (xrng() & 0x1FFF5F77) | 0x0000A088 | ((~Q[QOFF + 13]) & 0x40000000)
        Q[QOFF + 15] = (xrng() & 0x5EFE7FF7) | 0x80008000 | ((~Q[QOFF + 14]) & 0x00010000)
        Q[QOFF + 16] = (xrng() & 0x1FFDFFFF) | 0xA0000000 | ((~Q[QOFF + 15]) & 0x40020000)

        block[0] = reverse_step_first_round(Q, 0, 0xD76AA478, 7)
        block[6] = reverse_step_first_round(Q, 6, 0xA8304613, 17)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)
        block[11] = reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
        block[14] = reverse_step_first_round(Q, 14, 0xA679438E, 17)
        block[15] = reverse_step_first_round(Q, 15, 0x49B40821, 22)

        tt1 = (FF(Q[QOFF + 1], Q[QOFF + 0], Q[QOFF - 1]) + Q[QOFF - 2] + 0xE8C7B756) & MASK32
        tt17 = (GG(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + 0xF61E2562) & MASK32
        tt18 = (Q[QOFF + 14] + 0xC040B340 + block[6]) & MASK32
        tt19 = (Q[QOFF + 15] + 0x265E5A51 + block[11]) & MASK32
        tt20 = (Q[QOFF + 16] + 0xE9B6C7AA + block[0]) & MASK32

        tt5 = (ror((Q[QOFF + 6] - Q[QOFF + 5]) & MASK32, 12) - FF(Q[QOFF + 5], Q[QOFF + 4], Q[QOFF + 3]) - 0x4787C62A) & MASK32

        counter = 0
        while counter < (1 << 7):
            q16 = Q[QOFF + 16]
            q17 = (((xrng() & 0x3FFD7FF7) | (q16 & 0xC0008008)) ^ 0x40000000) & MASK32
            counter += 1

            q18 = (GG(q17, q16, Q[QOFF + 15]) + tt18) & MASK32
            q18 = (rol(q18, 9) + q17) & MASK32
            if 0x00020000 != ((q18 ^ q17) & 0xA0020000):
                continue

            q19 = (GG(q18, q17, q16) + tt19) & MASK32
            q19 = (rol(q19, 14) + q18) & MASK32
            if 0x80000000 != (q19 & 0x80020000):
                continue

            q20 = (GG(q19, q18, q17) + tt20) & MASK32
            q20 = (rol(q20, 20) + q19) & MASK32
            if 0x00040000 != ((q20 ^ q19) & 0x80040000):
                continue

            block[1] = (ror((q17 - q16) & MASK32, 5) - tt17) & MASK32
            q2 = (rol((block[1] + tt1) & MASK32, 12) + Q[QOFF + 1]) & MASK32
            block[5] = (tt5 - q2) & MASK32

            Q[QOFF + 2] = q2
            Q[QOFF + 17] = q17
            Q[QOFF + 18] = q18
            Q[QOFF + 19] = q19
            Q[QOFF + 20] = q20
            block[2] = reverse_step_first_round(Q, 2, 0x242070DB, 17)
            counter = 0
            break

        if counter != 0:
            continue

        q4 = Q[QOFF + 4]
        q9backup = Q[QOFF + 9]
        tt21 = (GG(Q[QOFF + 20], Q[QOFF + 19], Q[QOFF + 18]) + Q[QOFF + 17] + 0xD62F105D) & MASK32

        counter2 = 0
        while counter2 < (1 << 4):
            Q[QOFF + 4] = (q4 ^ q4mask[counter2]) & MASK32
            counter2 += 1

            block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
            q21 = (rol((tt21 + block[5]) & MASK32, 5) + Q[QOFF + 20]) & MASK32
            if 0 != ((q21 ^ Q[QOFF + 20]) & 0x80020000):
                continue

            Q[QOFF + 21] = q21
            block[3] = reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
            block[4] = reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
            block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)

            tt22 = (GG(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + 0x02441453) & MASK32
            tt23 = (Q[QOFF + 19] + 0xD8A1E681 + block[15]) & MASK32
            tt24 = (Q[QOFF + 20] + 0xE7D3FBC8 + block[4]) & MASK32

            tt9 = (Q[QOFF + 6] + 0x8B44F7AF) & MASK32
            tt10 = (Q[QOFF + 7] + 0xFFFF5BB1) & MASK32
            tt8 = (FF(Q[QOFF + 8], Q[QOFF + 7], Q[QOFF + 6]) + Q[QOFF + 5] + 0x698098D8) & MASK32
            tt12 = (ror((Q[QOFF + 13] - Q[QOFF + 12]) & MASK32, 7) - 0x6B901122) & MASK32
            tt13 = (ror((Q[QOFF + 14] - Q[QOFF + 13]) & MASK32, 12) - FF(Q[QOFF + 13], Q[QOFF + 12], Q[QOFF + 11]) - 0xFD987193) & MASK32

            for counter3 in range(1 << 3):
                q10 = (Q[QOFF + 10] ^ (q9q10mask[counter3] & 0x60)) & MASK32
                Q[QOFF + 9] = (q9backup ^ (q9q10mask[counter3] & 0x2000)) & MASK32

                m10 = ror((Q[QOFF + 11] - q10) & MASK32, 17)
                m10 = (m10 - FF(q10, Q[QOFF + 9], Q[QOFF + 8]) - tt10) & MASK32

                aa = Q[QOFF + 21]
                dd = (rol((tt22 + m10) & MASK32, 9) + aa) & MASK32
                if 0x80000000 != (dd & 0x80000000):
                    continue

                bb = Q[QOFF + 20]
                cc = (tt23 + GG(dd, aa, bb)) & MASK32
                if 0 != (cc & 0x20000):
                    continue
                cc = (rol(cc, 14) + dd) & MASK32
                if 0 != (cc & 0x80000000):
                    continue

                bb = (tt24 + GG(cc, dd, aa)) & MASK32
                bb = (rol(bb, 20) + cc) & MASK32
                if 0 == (bb & 0x80000000):
                    continue

                block[10] = m10
                block[13] = (tt13 - q10) & MASK32

                if iv_u32_np is not None and block_u32_np is not None:
                    # Keep Python control flow (RNG/state), but run the 2^16 q9mask scan in JIT.
                    block_u32_np[:] = block
                    ok = try_block0_q9mask_numba(
                        iv_u32_np,
                        _NP_U32(Q[QOFF + 7]),
                        _NP_U32(Q[QOFF + 8]),
                        _NP_U32(Q[QOFF + 9]),
                        _NP_U32(q10),
                        _NP_U32(Q[QOFF + 11]),
                        _NP_U32(Q[QOFF + 12]),
                        _NP_U32(tt8),
                        _NP_U32(tt9),
                        _NP_U32(tt12),
                        _NP_U32(aa),
                        _NP_U32(bb),
                        _NP_U32(cc),
                        _NP_U32(dd),
                        block_u32_np,
                    )
                    maybe = [int(x) for x in block_u32_np.tolist()] if ok else None
                elif _np is not None and q9_engine != "swar":
                    maybe = _try_block0_q9mask_numpy(
                        IV0=IV0,
                        IV1=IV1,
                        IV2=IV2,
                        IV3=IV3,
                        q7=Q[QOFF + 7],
                        q8=Q[QOFF + 8],
                        q9_base=Q[QOFF + 9],
                        q10=q10,
                        q11=Q[QOFF + 11],
                        q12=Q[QOFF + 12],
                        tt8=tt8,
                        tt9=tt9,
                        tt12=tt12,
                        aa=aa,
                        bb=bb,
                        cc=cc,
                        dd=dd,
                        block=block,
                    )
                else:
                    maybe = _try_block0_q9mask_swar(
                        IV0=IV0,
                        IV1=IV1,
                        IV2=IV2,
                        IV3=IV3,
                        q7=Q[QOFF + 7],
                        q8=Q[QOFF + 8],
                        q9_base=Q[QOFF + 9],
                        q10=q10,
                        q11=Q[QOFF + 11],
                        q12=Q[QOFF + 12],
                        tt8=tt8,
                        tt9=tt9,
                        tt12=tt12,
                        aa=aa,
                        bb=bb,
                        cc=cc,
                        dd=dd,
                        block=block,
                    )
                if maybe is not None:
                    return maybe
                continue


def _tail_rounds_common(
    a: int,
    b: int,
    c: int,
    d: int,
    block: List[int],
    *,
    hh_step34_bit15_required: int,
) -> Tuple[int, int, int, int] | None:
    """
    Shared MD5 tail for block1 generators (steps 24..63).
    `hh_step34_bit15_required`:
      - 0: require bit15 == 0 (Stevens)
      - 1: require bit15 == 1 (Wang)
    """
    # Step 24..31 (GG)
    a = (rol((a + GG(b, c, d) + block[9] + 0x21E1CDE6) & MASK32, 5) + b) & MASK32
    d = (rol((d + GG(a, b, c) + block[14] + 0xC33707D6) & MASK32, 9) + a) & MASK32
    c = (rol((c + GG(d, a, b) + block[3] + 0xF4D50D87) & MASK32, 14) + d) & MASK32
    b = (rol((b + GG(c, d, a) + block[8] + 0x455A14ED) & MASK32, 20) + c) & MASK32
    a = (rol((a + GG(b, c, d) + block[13] + 0xA9E3E905) & MASK32, 5) + b) & MASK32
    d = (rol((d + GG(a, b, c) + block[2] + 0xFCEFA3F8) & MASK32, 9) + a) & MASK32
    c = (rol((c + GG(d, a, b) + block[7] + 0x676F02D9) & MASK32, 14) + d) & MASK32
    b = (rol((b + GG(c, d, a) + block[12] + 0x8D2A4C8A) & MASK32, 20) + c) & MASK32

    # Step 32..47 (HH)
    a = (rol((a + HH(b, c, d) + block[5] + 0xFFFA3942) & MASK32, 4) + b) & MASK32
    d = (rol((d + HH(a, b, c) + block[8] + 0x8771F681) & MASK32, 11) + a) & MASK32
    c = (c + HH(d, a, b) + block[11] + 0x6D9D6122) & MASK32
    if ((c >> 15) & 1) != (hh_step34_bit15_required & 1):
        return None
    c = (rol(c, 16) + d) & MASK32
    b = (rol((b + HH(c, d, a) + block[14] + 0xFDE5380C) & MASK32, 23) + c) & MASK32

    a = (rol((a + HH(b, c, d) + block[1] + 0xA4BEEA44) & MASK32, 4) + b) & MASK32
    d = (rol((d + HH(a, b, c) + block[4] + 0x4BDECFA9) & MASK32, 11) + a) & MASK32
    c = (rol((c + HH(d, a, b) + block[7] + 0xF6BB4B60) & MASK32, 16) + d) & MASK32
    b = (rol((b + HH(c, d, a) + block[10] + 0xBEBFBC70) & MASK32, 23) + c) & MASK32
    a = (rol((a + HH(b, c, d) + block[13] + 0x289B7EC6) & MASK32, 4) + b) & MASK32
    d = (rol((d + HH(a, b, c) + block[0] + 0xEAA127FA) & MASK32, 11) + a) & MASK32
    c = (rol((c + HH(d, a, b) + block[3] + 0xD4EF3085) & MASK32, 16) + d) & MASK32
    b = (rol((b + HH(c, d, a) + block[6] + 0x04881D05) & MASK32, 23) + c) & MASK32
    a = (rol((a + HH(b, c, d) + block[9] + 0xD9D4D039) & MASK32, 4) + b) & MASK32
    d = (rol((d + HH(a, b, c) + block[12] + 0xE6DB99E5) & MASK32, 11) + a) & MASK32
    c = (rol((c + HH(d, a, b) + block[15] + 0x1FA27CF8) & MASK32, 16) + d) & MASK32
    b = (rol((b + HH(c, d, a) + block[2] + 0xC4AC5665) & MASK32, 23) + c) & MASK32
    if 0 != ((b ^ d) & 0x80000000):
        return None

    # Step 48..63 (II)
    a = (rol((a + II(b, c, d) + block[0] + 0xF4292244) & MASK32, 6) + b) & MASK32
    if 0 != ((a ^ c) & 0x80000000):
        return None
    d = (rol((d + II(a, b, c) + block[7] + 0x432AFF97) & MASK32, 10) + a) & MASK32
    if 0 == ((b ^ d) & 0x80000000):
        return None
    c = (rol((c + II(d, a, b) + block[14] + 0xAB9423A7) & MASK32, 15) + d) & MASK32
    if 0 != ((a ^ c) & 0x80000000):
        return None
    b = (rol((b + II(c, d, a) + block[5] + 0xFC93A039) & MASK32, 21) + c) & MASK32
    if 0 != ((b ^ d) & 0x80000000):
        return None
    a = (rol((a + II(b, c, d) + block[12] + 0x655B59C3) & MASK32, 6) + b) & MASK32
    if 0 != ((a ^ c) & 0x80000000):
        return None
    d = (rol((d + II(a, b, c) + block[3] + 0x8F0CCC92) & MASK32, 10) + a) & MASK32
    if 0 != ((b ^ d) & 0x80000000):
        return None
    c = (rol((c + II(d, a, b) + block[10] + 0xFFEFF47D) & MASK32, 15) + d) & MASK32
    if 0 != ((a ^ c) & 0x80000000):
        return None
    b = (rol((b + II(c, d, a) + block[1] + 0x85845DD1) & MASK32, 21) + c) & MASK32
    if 0 != ((b ^ d) & 0x80000000):
        return None
    a = (rol((a + II(b, c, d) + block[8] + 0x6FA87E4F) & MASK32, 6) + b) & MASK32
    if 0 != ((a ^ c) & 0x80000000):
        return None
    d = (rol((d + II(a, b, c) + block[15] + 0xFE2CE6E0) & MASK32, 10) + a) & MASK32
    if 0 != ((b ^ d) & 0x80000000):
        return None
    c = (rol((c + II(d, a, b) + block[6] + 0xA3014314) & MASK32, 15) + d) & MASK32
    if 0 != ((a ^ c) & 0x80000000):
        return None
    b = (rol((b + II(c, d, a) + block[13] + 0x4E0811A1) & MASK32, 21) + c) & MASK32
    if 0 == ((b ^ d) & 0x80000000):
        return None
    a = (rol((a + II(b, c, d) + block[4] + 0xF7537E82) & MASK32, 6) + b) & MASK32
    if 0 != ((a ^ c) & 0x80000000):
        return None
    d = (rol((d + II(a, b, c) + block[11] + 0xBD3AF235) & MASK32, 10) + a) & MASK32
    if 0 != ((b ^ d) & 0x80000000):
        return None
    c = (rol((c + II(d, a, b) + block[2] + 0x2AD7D2BB) & MASK32, 15) + d) & MASK32
    if 0 != ((a ^ c) & 0x80000000):
        return None
    b = (rol((b + II(c, d, a) + block[9] + 0xEB86D391) & MASK32, 21) + c) & MASK32
    return (a, b, c, d)


def find_block1_wang(rng: XorShift64, IV: Tuple[int, int, int, int]) -> List[int]:
    """HashClash `find_block1_wang`."""
    IV0, IV1, IV2, IV3 = (u32(IV[0]), u32(IV[1]), u32(IV[2]), u32(IV[3]))
    Q = [0] * 68
    Q[0], Q[1], Q[2], Q[3] = IV0, IV3, IV2, IV1
    block = [0] * 16

    q4mask = _W_Q4MASK
    q9mask = _W_Q9MASK
    q10mask = _W_Q10MASK
    q9mask2 = _W_Q9MASK2
    xrng = rng.next_u32

    while True:
        aa = Q[QOFF] & 0x80000000
        bb = 0x80000000 ^ aa

        Q[QOFF + 2] = (xrng() & 0x71DE7799) | 0x0C008840 | bb
        Q[QOFF + 3] = (xrng() & 0x01C06601) | 0x3E1F0966 | (Q[QOFF + 2] & 0x80000018)
        Q[QOFF + 4] = 0x3A040010 | (Q[QOFF + 3] & 0x80000601)
        Q[QOFF + 5] = (xrng() & 0x03C0E000) | 0x482F0E50 | aa
        Q[QOFF + 6] = (xrng() & 0x600C0000) | 0x05E2EC56 | aa
        Q[QOFF + 7] = (xrng() & 0x604C203E) | 0x16819E01 | bb | (Q[QOFF + 6] & 0x01000000)
        Q[QOFF + 8] = (xrng() & 0x604C7C1C) | 0x043283E0 | (Q[QOFF + 7] & 0x80000002)
        Q[QOFF + 9] = (xrng() & 0x00002800) | 0x1C0101C1 | (Q[QOFF + 8] & 0x80001000)
        Q[QOFF + 10] = 0x078BCBC0 | bb
        Q[QOFF + 11] = (xrng() & 0x07800000) | 0x607DC7DF | bb
        Q[QOFF + 12] = (xrng() & 0x00F00F7F) | 0x00081080 | (Q[QOFF + 11] & 0xE7000000)
        Q[QOFF + 13] = (xrng() & 0x00701F77) | 0x3F0FE008 | aa
        Q[QOFF + 14] = (xrng() & 0x00701F77) | 0x408BE088 | aa
        Q[QOFF + 15] = (xrng() & 0x00FF3FF7) | 0x7D000000
        Q[QOFF + 16] = (xrng() & 0x4FFDFFFF) | 0x20000000 | ((~Q[QOFF + 15]) & 0x00020000)

        block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
        block[6] = reverse_step_first_round(Q, 6, 0xA8304613, 17)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)
        block[11] = reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
        block[14] = reverse_step_first_round(Q, 14, 0xA679438E, 17)
        block[15] = reverse_step_first_round(Q, 15, 0x49B40821, 22)

        tt17 = (GG(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + 0xF61E2562) & MASK32
        tt18 = (Q[QOFF + 14] + 0xC040B340 + block[6]) & MASK32
        tt19 = (Q[QOFF + 15] + 0x265E5A51 + block[11]) & MASK32

        tt0 = (FF(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + 0xD76AA478) & MASK32
        tt1 = (Q[QOFF - 2] + 0xE8C7B756) & MASK32
        q1a = (0x04200040 | (Q[QOFF + 2] & 0xF01E1080)) & MASK32

        if _np is not None:
            res = _batch_find_q1_chain_numpy(
                rng,
                q1a=q1a,
                q1_mask=0x01C0E71F,
                q2=Q[QOFF + 2],
                q0=Q[QOFF + 0],
                q_minus1=Q[QOFF - 1],
                tt1=tt1,
                q16=Q[QOFF + 16],
                q15=Q[QOFF + 15],
                tt17=tt17,
                tt18=tt18,
                tt19=tt19,
                tt0=tt0,
                q17_xor_mask=0xC0008008,
                q17_xor_expect=0x40000000,
                q17_forbid_mask=0x00020000,
                q18_xor_mask=0xA0020000,
                q18_xor_expect=0x00020000,
                q19_and_mask=0x80020000,
                q19_and_expect=0x00000000,
                q20_xor_mask=0x80040000,
                q20_xor_expect=0x00040000,
            )
            if res is None:
                continue
            q1, m0, m1, q17, q18, q19, q20 = res
            Q[QOFF + 1] = q1
            Q[QOFF + 17] = q17
            Q[QOFF + 18] = q18
            Q[QOFF + 19] = q19
            Q[QOFF + 20] = q20
            block[0] = m0
            block[1] = m1
            block[2] = reverse_step_first_round(Q, 2, 0x242070DB, 17)
        else:
            counter = 0
            while counter < (1 << 12):
                counter += 1
                q1 = (q1a | (xrng() & 0x01C0E71F)) & MASK32
                m1 = (Q[QOFF + 2] - q1) & MASK32
                m1 = (ror(m1, 12) - FF(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1) & MASK32

                q16 = Q[QOFF + 16]
                q17 = (rol((tt17 + m1) & MASK32, 5) + q16) & MASK32
                if 0x40000000 != ((q17 ^ q16) & 0xC0008008):
                    continue
                if 0 != (q17 & 0x00020000):
                    continue

                q18 = (rol((GG(q17, q16, Q[QOFF + 15]) + tt18) & MASK32, 9) + q17) & MASK32
                if 0x00020000 != ((q18 ^ q17) & 0xA0020000):
                    continue

                q19 = (rol((GG(q18, q17, q16) + tt19) & MASK32, 14) + q18) & MASK32
                if 0 != (q19 & 0x80020000):
                    continue

                m0 = (q1 - Q[QOFF + 0]) & MASK32
                m0 = (ror(m0, 7) - tt0) & MASK32

                q20 = (rol((GG(q19, q18, q17) + q16 + 0xE9B6C7AA + m0) & MASK32, 20) + q19) & MASK32
                if 0x00040000 != ((q20 ^ q19) & 0x80040000):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1
                block[2] = reverse_step_first_round(Q, 2, 0x242070DB, 17)
                counter = 0
                break

            if counter != 0:
                continue

        q4b = Q[QOFF + 4]
        q9b = Q[QOFF + 9]
        q10b = Q[QOFF + 10]
        tt21 = (GG(Q[QOFF + 20], Q[QOFF + 19], Q[QOFF + 18]) + Q[QOFF + 17] + 0xD62F105D) & MASK32

        counter = 0
        while counter < (1 << 6):
            Q[QOFF + 4] = (q4b ^ q4mask[counter]) & MASK32
            counter += 1

            block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
            q21 = (rol((tt21 + block[5]) & MASK32, 5) + Q[QOFF + 20]) & MASK32
            if 0 != ((q21 ^ Q[QOFF + 20]) & 0x80020000):
                continue

            Q[QOFF + 21] = q21
            block[3] = reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
            block[4] = reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
            block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)

            tt10 = (Q[QOFF + 7] + 0xFFFF5BB1) & MASK32
            tt22 = (GG(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + 0x02441453) & MASK32
            tt23 = (Q[QOFF + 19] + 0xD8A1E681 + block[15]) & MASK32
            tt24 = (Q[QOFF + 20] + 0xE7D3FBC8 + block[4]) & MASK32

            counter2 = 0
            while counter2 < (1 << 5):
                q10 = (q10b ^ q10mask[counter2]) & MASK32
                q9 = (q9b ^ q9mask[counter2]) & MASK32
                counter2 += 1

                m10 = ror((Q[QOFF + 11] - q10) & MASK32, 17)
                m10 = (m10 - FF(q10, q9, Q[QOFF + 8]) - tt10) & MASK32

                aa = Q[QOFF + 21]
                dd = (rol((tt22 + m10) & MASK32, 9) + aa) & MASK32
                if 0 != (dd & 0x80000000):
                    continue

                bb = Q[QOFF + 20]
                cc = (tt23 + GG(dd, aa, bb)) & MASK32
                if 0 != (cc & 0x20000):
                    continue
                cc = (rol(cc, 14) + dd) & MASK32
                if 0 != (cc & 0x80000000):
                    continue

                bb = (rol((tt24 + GG(cc, dd, aa)) & MASK32, 20) + cc) & MASK32
                if 0 == (bb & 0x80000000):
                    continue

                block[10] = m10
                Q[QOFF + 9] = q9
                Q[QOFF + 10] = q10
                block[13] = reverse_step_first_round(Q, 13, 0xFD987193, 12)

                iv1 = (IV0, IV1, IV2, IV3)
                iv2 = (
                    (IV0 + (1 << 31)) & MASK32,
                    (IV1 + (1 << 31) + (1 << 25)) & MASK32,
                    (IV2 + (1 << 31) + (1 << 25)) & MASK32,
                    (IV3 + (1 << 31) + (1 << 25)) & MASK32,
                )

                if _np is not None:
                    maybe = _try_block1_q9_variants_numpy(
                        q9_base=q9,
                        q9_masks_np=_W_Q9MASK2_NP,
                        q5=Q[QOFF + 5],
                        q6=Q[QOFF + 6],
                        q7=Q[QOFF + 7],
                        q8=Q[QOFF + 8],
                        q10=q10,
                        q11=Q[QOFF + 11],
                        q12=Q[QOFF + 12],
                        q13=Q[QOFF + 13],
                        a0=Q[QOFF + 21],
                        b0=bb,
                        c0=cc,
                        d0=dd,
                        block=block,
                        hh_step34_bit15_required=1,
                        iv1=iv1,
                        iv2=iv2,
                        delta_apply=apply_delta_block1,
                    )
                    if maybe is not None:
                        return maybe
                    continue

                for msk in q9mask2:
                    Q[QOFF + 9] = (q9 ^ msk) & MASK32
                    block[8] = reverse_step_first_round(Q, 8, 0x698098D8, 7)
                    block[9] = reverse_step_first_round(Q, 9, 0x8B44F7AF, 12)
                    block[12] = reverse_step_first_round(Q, 12, 0x6B901122, 7)

                    tail = _tail_rounds_common(
                        Q[QOFF + 21],
                        bb,
                        cc,
                        dd,
                        block,
                        hh_step34_bit15_required=1,
                    )
                    if tail is None:
                        continue
                    a, b, c, d = tail

                    # Verify collision for the Wang differential.
                    out1 = md5_compress(iv1, block)
                    out2 = md5_compress(iv2, apply_delta_block1(block))
                    if out1 == out2:
                        return list(block)


def find_block1_stevens_00(rng: XorShift64, IV: Tuple[int, int, int, int]) -> List[int]:
    """HashClash `find_block1_stevens_00`: returns msg2block1 words."""
    IV0, IV1, IV2, IV3 = (u32(IV[0]), u32(IV[1]), u32(IV[2]), u32(IV[3]))
    Q = [0] * 68
    Q[0], Q[1], Q[2], Q[3] = IV0, IV3, IV2, IV1
    block = [0] * 16
    xrng = rng.next_u32
    q9q10mask = _S00_Q9Q10MASK
    q9mask = _S00_Q9MASK

    while True:
        aa = Q[QOFF] & 0x80000000

        Q[QOFF + 2] = (xrng() & 0x49A0E73E) | 0x221F00C1 | aa
        Q[QOFF + 3] = (xrng() & 0x0000040C) | 0x3FCE1A71 | (Q[QOFF + 2] & 0x8000E000)
        Q[QOFF + 4] = (xrng() & 0x00000004) | (0xA5F281A2 ^ (Q[QOFF + 3] & 0x80000008))
        Q[QOFF + 5] = (xrng() & 0x00000004) | 0x67FD823B
        Q[QOFF + 6] = (xrng() & 0x00001044) | 0x15E5829A
        Q[QOFF + 7] = (xrng() & 0x00200806) | 0x950430B0
        Q[QOFF + 8] = (xrng() & 0x60050110) | 0x1BD29CA2 | (Q[QOFF + 7] & 0x00000004)
        Q[QOFF + 9] = (xrng() & 0x40044000) | 0xB8820004
        Q[QOFF + 10] = 0xF288B209 | (Q[QOFF + 9] & 0x00044000)
        Q[QOFF + 11] = (xrng() & 0x12888008) | 0x85712F57
        Q[QOFF + 12] = (xrng() & 0x1ED98D7F) | 0xC0023080 | ((~Q[QOFF + 11]) & 0x00200000)
        Q[QOFF + 13] = (xrng() & 0x0EFB1D77) | 0x1000C008
        Q[QOFF + 14] = (xrng() & 0x0FFF5D77) | 0xA000A288
        Q[QOFF + 15] = (xrng() & 0x0EFE7FF7) | 0xE0008000 | ((~Q[QOFF + 14]) & 0x00010000)
        Q[QOFF + 16] = (xrng() & 0x0FFDFFFF) | 0xF0000000 | ((~Q[QOFF + 15]) & 0x00020000)

        block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
        block[6] = reverse_step_first_round(Q, 6, 0xA8304613, 17)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)
        block[11] = reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
        block[14] = reverse_step_first_round(Q, 14, 0xA679438E, 17)
        block[15] = reverse_step_first_round(Q, 15, 0x49B40821, 22)

        tt17 = (GG(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + 0xF61E2562) & MASK32
        tt18 = (Q[QOFF + 14] + 0xC040B340 + block[6]) & MASK32
        tt19 = (Q[QOFF + 15] + 0x265E5A51 + block[11]) & MASK32

        tt0 = (FF(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + 0xD76AA478) & MASK32
        tt1 = (Q[QOFF - 2] + 0xE8C7B756) & MASK32
        q1a = (0x02020801 | (Q[QOFF + 0] & 0x80000000)) & MASK32

        if _np is not None:
            res = _batch_find_q1_chain_numpy(
                rng,
                q1a=q1a,
                q1_mask=0x7DFDF7BE,
                q2=Q[QOFF + 2],
                q0=Q[QOFF + 0],
                q_minus1=Q[QOFF - 1],
                tt1=tt1,
                q16=Q[QOFF + 16],
                q15=Q[QOFF + 15],
                tt17=tt17,
                tt18=tt18,
                tt19=tt19,
                tt0=tt0,
                q17_xor_mask=0x80008008,
                q17_xor_expect=0x80000000,
                q17_forbid_mask=0x00020000,
                q18_xor_mask=0xA0020000,
                q18_xor_expect=0x80020000,
                q19_and_mask=0x80020000,
                q19_and_expect=0x80000000,
                q20_xor_mask=0x80040000,
                q20_xor_expect=0x00040000,
                want_q21=True,
                block5=block[5],
                q21_xor_mask=0x80020000,
                q21_xor_expect=0x00000000,
            )
            if res is None:
                continue
            q1, m0, m1, q17, q18, q19, q20, q21 = res
            Q[QOFF + 1] = q1
            Q[QOFF + 17] = q17
            Q[QOFF + 18] = q18
            Q[QOFF + 19] = q19
            Q[QOFF + 20] = q20
            Q[QOFF + 21] = q21
            block[0] = m0
            block[1] = m1
        else:
            counter = 0
            while counter < (1 << 12):
                counter += 1

                q1 = (q1a | (xrng() & 0x7DFDF7BE)) & MASK32
                m1 = (Q[QOFF + 2] - q1) & MASK32
                m1 = (ror(m1, 12) - FF(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1) & MASK32

                q16 = Q[QOFF + 16]
                q17 = (rol((tt17 + m1) & MASK32, 5) + q16) & MASK32
                if 0x80000000 != ((q17 ^ q16) & 0x80008008):
                    continue
                if 0 != (q17 & 0x00020000):
                    continue

                q18 = (rol((GG(q17, q16, Q[QOFF + 15]) + tt18) & MASK32, 9) + q17) & MASK32
                if 0x80020000 != ((q18 ^ q17) & 0xA0020000):
                    continue

                q19 = (rol((GG(q18, q17, q16) + tt19) & MASK32, 14) + q18) & MASK32
                if 0x80000000 != (q19 & 0x80020000):
                    continue

                m0 = (q1 - Q[QOFF + 0]) & MASK32
                m0 = (ror(m0, 7) - tt0) & MASK32

                q20 = (rol((GG(q19, q18, q17) + q16 + 0xE9B6C7AA + m0) & MASK32, 20) + q19) & MASK32
                if 0x00040000 != ((q20 ^ q19) & 0x80040000):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1

                block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = (rol((GG(q20, q19, q18) + q17 + 0xD62F105D + block[5]) & MASK32, 5) + q20) & MASK32
                if 0 != ((q21 ^ q20) & 0x80020000):
                    continue
                Q[QOFF + 21] = q21

                counter = 0
                break

            if counter != 0:
                continue

        q9b = Q[QOFF + 9]
        q10b = Q[QOFF + 10]

        block[2] = reverse_step_first_round(Q, 2, 0x242070DB, 17)
        block[3] = reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
        block[4] = reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)

        tt10 = (Q[QOFF + 7] + 0xFFFF5BB1) & MASK32
        tt22 = (GG(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + 0x02441453) & MASK32
        tt23 = (Q[QOFF + 19] + 0xD8A1E681 + block[15]) & MASK32
        tt24 = (Q[QOFF + 20] + 0xE7D3FBC8 + block[4]) & MASK32

        for k10 in range(1 << 3):
            mask = q9q10mask[k10]
            q10 = (q10b | (mask & 0x08000020)) & MASK32
            q9 = (q9b | (mask & 0x00002000)) & MASK32

            m10 = ror((Q[QOFF + 11] - q10) & MASK32, 17)
            m10 = (m10 - FF(q10, q9, Q[QOFF + 8]) - tt10) & MASK32

            aa = Q[QOFF + 21]
            dd = (rol((tt22 + m10) & MASK32, 9) + aa) & MASK32
            if 0 == (dd & 0x80000000):
                continue

            bb = Q[QOFF + 20]
            cc = (tt23 + GG(dd, aa, bb)) & MASK32
            if 0 != (cc & 0x20000):
                continue
            cc = (rol(cc, 14) + dd) & MASK32
            if 0 != (cc & 0x80000000):
                continue

            bb = (rol((tt24 + GG(cc, dd, aa)) & MASK32, 20) + cc) & MASK32
            if 0 == (bb & 0x80000000):
                continue

            block[10] = m10
            Q[QOFF + 9] = q9
            Q[QOFF + 10] = q10
            block[13] = reverse_step_first_round(Q, 13, 0xFD987193, 12)

            iv1 = (IV0, IV1, IV2, IV3)
            iv2 = (
                (IV0 + (1 << 31)) & MASK32,
                (IV1 + (1 << 31) - (1 << 25)) & MASK32,
                (IV2 + (1 << 31) - (1 << 25)) & MASK32,
                (IV3 + (1 << 31) - (1 << 25)) & MASK32,
            )

            if _np is not None:
                maybe = _try_block1_q9_variants_numpy(
                    q9_base=q9,
                    q9_masks_np=_S00_Q9MASK_NP,
                    q5=Q[QOFF + 5],
                    q6=Q[QOFF + 6],
                    q7=Q[QOFF + 7],
                    q8=Q[QOFF + 8],
                    q10=q10,
                    q11=Q[QOFF + 11],
                    q12=Q[QOFF + 12],
                    q13=Q[QOFF + 13],
                    a0=aa,
                    b0=bb,
                    c0=cc,
                    d0=dd,
                    block=block,
                    hh_step34_bit15_required=0,
                    iv1=iv1,
                    iv2=iv2,
                    delta_apply=apply_delta_block0,
                )
                if maybe is not None:
                    return maybe
                continue

            for k9 in range(1 << 9):
                a, b, c, d = aa, bb, cc, dd
                Q[QOFF + 9] = (q9 ^ q9mask[k9]) & MASK32
                block[8] = reverse_step_first_round(Q, 8, 0x698098D8, 7)
                block[9] = reverse_step_first_round(Q, 9, 0x8B44F7AF, 12)
                block[12] = reverse_step_first_round(Q, 12, 0x6B901122, 7)

                if _tail_rounds_common(a, b, c, d, block, hh_step34_bit15_required=0) is None:
                    continue

                out1 = md5_compress(iv1, block)
                out2 = md5_compress(iv2, apply_delta_block0(block))
                if out1 == out2:
                    return list(block)


def find_block1_stevens_01(rng: XorShift64, IV: Tuple[int, int, int, int]) -> List[int]:
    """HashClash `find_block1_stevens_01`: returns msg2block1 words."""
    IV0, IV1, IV2, IV3 = (u32(IV[0]), u32(IV[1]), u32(IV[2]), u32(IV[3]))
    Q = [0] * 68
    Q[0], Q[1], Q[2], Q[3] = IV0, IV3, IV2, IV1
    block = [0] * 16
    xrng = rng.next_u32
    q9q10mask = _S01_Q9Q10MASK
    q9mask = _S01_Q9MASK

    while True:
        aa = Q[QOFF] & 0x80000000

        Q[QOFF + 2] = (xrng() & 0x4DB0E03E) | 0x32460441 | aa
        Q[QOFF + 3] = (xrng() & 0x0C000008) | 0x123C3AF1 | (Q[QOFF + 2] & 0x80800002)
        Q[QOFF + 4] = 0xE398F812 ^ (Q[QOFF + 3] & 0x88000000)
        Q[QOFF + 5] = (xrng() & 0x82000000) | 0x4C66E99E
        Q[QOFF + 6] = (xrng() & 0x80000000) | 0x27180590
        Q[QOFF + 7] = (xrng() & 0x00010130) | 0x51EA9E47
        Q[QOFF + 8] = (xrng() & 0x40200800) | 0xB7C291E5
        Q[QOFF + 9] = (xrng() & 0x00044000) | 0x380002B4
        Q[QOFF + 10] = 0xB282B208 | (Q[QOFF + 9] & 0x00044000)
        Q[QOFF + 11] = (xrng() & 0x12808008) | 0xC5712F47
        Q[QOFF + 12] = (xrng() & 0x1EF18D7F) | 0x000A3080
        Q[QOFF + 13] = (xrng() & 0x1EFB1D77) | 0x4004C008
        Q[QOFF + 14] = (xrng() & 0x1FFF5D77) | 0x6000A288
        Q[QOFF + 15] = (xrng() & 0x1EFE7FF7) | 0xA0008000 | ((~Q[QOFF + 14]) & 0x00010000)
        Q[QOFF + 16] = (xrng() & 0x1FFDFFFF) | 0x20000000 | ((~Q[QOFF + 15]) & 0x00020000)

        block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
        block[6] = reverse_step_first_round(Q, 6, 0xA8304613, 17)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)
        block[11] = reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
        block[14] = reverse_step_first_round(Q, 14, 0xA679438E, 17)
        block[15] = reverse_step_first_round(Q, 15, 0x49B40821, 22)

        tt17 = (GG(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + 0xF61E2562) & MASK32
        tt18 = (Q[QOFF + 14] + 0xC040B340 + block[6]) & MASK32
        tt19 = (Q[QOFF + 15] + 0x265E5A51 + block[11]) & MASK32

        tt0 = (FF(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + 0xD76AA478) & MASK32
        tt1 = (Q[QOFF - 2] + 0xE8C7B756) & MASK32
        q1a = (0x02000021 ^ (Q[QOFF + 0] & 0x80000020)) & MASK32

        if _np is not None:
            res = _batch_find_q1_chain_numpy(
                rng,
                q1a=q1a,
                q1_mask=0x7DFFF39E,
                q2=Q[QOFF + 2],
                q0=Q[QOFF + 0],
                q_minus1=Q[QOFF - 1],
                tt1=tt1,
                q16=Q[QOFF + 16],
                q15=Q[QOFF + 15],
                tt17=tt17,
                tt18=tt18,
                tt19=tt19,
                tt0=tt0,
                q17_xor_mask=0x80008008,
                q17_xor_expect=0x80000000,
                q17_forbid_mask=0x00020000,
                q18_xor_mask=0xA0020000,
                q18_xor_expect=0x80020000,
                q19_and_mask=0x80020000,
                q19_and_expect=0x00000000,
                q20_xor_mask=0x80040000,
                q20_xor_expect=0x00040000,
                want_q21=True,
                block5=block[5],
                q21_xor_mask=0x80020000,
                q21_xor_expect=0x00000000,
            )
            if res is None:
                continue
            q1, m0, m1, q17, q18, q19, q20, q21 = res
            Q[QOFF + 1] = q1
            Q[QOFF + 17] = q17
            Q[QOFF + 18] = q18
            Q[QOFF + 19] = q19
            Q[QOFF + 20] = q20
            Q[QOFF + 21] = q21
            block[0] = m0
            block[1] = m1
        else:
            counter = 0
            while counter < (1 << 12):
                counter += 1

                q1 = (q1a | (xrng() & 0x7DFFF39E)) & MASK32
                m1 = (Q[QOFF + 2] - q1) & MASK32
                m1 = (ror(m1, 12) - FF(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1) & MASK32

                q16 = Q[QOFF + 16]
                q17 = (rol((tt17 + m1) & MASK32, 5) + q16) & MASK32
                if 0x80000000 != ((q17 ^ q16) & 0x80008008):
                    continue
                if 0 != (q17 & 0x00020000):
                    continue

                q18 = (rol((GG(q17, q16, Q[QOFF + 15]) + tt18) & MASK32, 9) + q17) & MASK32
                if 0x80020000 != ((q18 ^ q17) & 0xA0020000):
                    continue

                q19 = (rol((GG(q18, q17, q16) + tt19) & MASK32, 14) + q18) & MASK32
                if 0 != (q19 & 0x80020000):
                    continue

                m0 = (q1 - Q[QOFF + 0]) & MASK32
                m0 = (ror(m0, 7) - tt0) & MASK32

                q20 = (rol((GG(q19, q18, q17) + q16 + 0xE9B6C7AA + m0) & MASK32, 20) + q19) & MASK32
                if 0x00040000 != ((q20 ^ q19) & 0x80040000):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1

                block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = (rol((GG(q20, q19, q18) + q17 + 0xD62F105D + block[5]) & MASK32, 5) + q20) & MASK32
                if 0 != ((q21 ^ q20) & 0x80020000):
                    continue
                Q[QOFF + 21] = q21

                counter = 0
                break

            if counter != 0:
                continue

        q9b = Q[QOFF + 9]
        q10b = Q[QOFF + 10]

        block[2] = reverse_step_first_round(Q, 2, 0x242070DB, 17)
        block[3] = reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
        block[4] = reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)

        tt10 = (Q[QOFF + 7] + 0xFFFF5BB1) & MASK32
        tt22 = (GG(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + 0x02441453) & MASK32
        tt23 = (Q[QOFF + 19] + 0xD8A1E681 + block[15]) & MASK32
        tt24 = (Q[QOFF + 20] + 0xE7D3FBC8 + block[4]) & MASK32

        for k10 in range(1 << 5):
            mask = q9q10mask[k10]
            q10 = (q10b | (mask & 0x08000030)) & MASK32
            q9 = (q9b | (mask & 0x80002000)) & MASK32

            m10 = ror((Q[QOFF + 11] - q10) & MASK32, 17)
            m10 = (m10 - FF(q10, q9, Q[QOFF + 8]) - tt10) & MASK32

            aa = Q[QOFF + 21]
            dd = (rol((tt22 + m10) & MASK32, 9) + aa) & MASK32
            if 0 != (dd & 0x80000000):
                continue

            bb = Q[QOFF + 20]
            cc = (tt23 + GG(dd, aa, bb)) & MASK32
            if 0 != (cc & 0x20000):
                continue
            cc = (rol(cc, 14) + dd) & MASK32
            if 0 != (cc & 0x80000000):
                continue

            bb = (rol((tt24 + GG(cc, dd, aa)) & MASK32, 20) + cc) & MASK32
            if 0 == (bb & 0x80000000):
                continue

            block[10] = m10
            Q[QOFF + 9] = q9
            Q[QOFF + 10] = q10
            block[13] = reverse_step_first_round(Q, 13, 0xFD987193, 12)

            iv1 = (IV0, IV1, IV2, IV3)
            iv2 = (
                (IV0 + (1 << 31)) & MASK32,
                (IV1 + (1 << 31) - (1 << 25)) & MASK32,
                (IV2 + (1 << 31) - (1 << 25)) & MASK32,
                (IV3 + (1 << 31) - (1 << 25)) & MASK32,
            )

            if _np is not None:
                maybe = _try_block1_q9_variants_numpy(
                    q9_base=q9,
                    q9_masks_np=_S01_Q9MASK_NP,
                    q5=Q[QOFF + 5],
                    q6=Q[QOFF + 6],
                    q7=Q[QOFF + 7],
                    q8=Q[QOFF + 8],
                    q10=q10,
                    q11=Q[QOFF + 11],
                    q12=Q[QOFF + 12],
                    q13=Q[QOFF + 13],
                    a0=aa,
                    b0=bb,
                    c0=cc,
                    d0=dd,
                    block=block,
                    hh_step34_bit15_required=0,
                    iv1=iv1,
                    iv2=iv2,
                    delta_apply=apply_delta_block0,
                )
                if maybe is not None:
                    return maybe
                continue

            for k9 in range(1 << 9):
                a, b, c, d = aa, bb, cc, dd
                Q[QOFF + 9] = (q9 ^ q9mask[k9]) & MASK32
                block[8] = reverse_step_first_round(Q, 8, 0x698098D8, 7)
                block[9] = reverse_step_first_round(Q, 9, 0x8B44F7AF, 12)
                block[12] = reverse_step_first_round(Q, 12, 0x6B901122, 7)

                if _tail_rounds_common(a, b, c, d, block, hh_step34_bit15_required=0) is None:
                    continue

                out1 = md5_compress(iv1, block)
                out2 = md5_compress(iv2, apply_delta_block0(block))
                if out1 == out2:
                    return list(block)


def find_block1_stevens_10(rng: XorShift64, IV: Tuple[int, int, int, int]) -> List[int]:
    """HashClash `find_block1_stevens_10`: returns msg2block1 words."""
    IV0, IV1, IV2, IV3 = (u32(IV[0]), u32(IV[1]), u32(IV[2]), u32(IV[3]))
    Q = [0] * 68
    Q[0], Q[1], Q[2], Q[3] = IV0, IV3, IV2, IV1
    block = [0] * 16
    xrng = rng.next_u32
    q9q10mask = _S10_Q9Q10MASK
    q9mask = _S10_Q9MASK

    while True:
        aa = Q[QOFF] & 0x80000000

        Q[QOFF + 2] = (xrng() & 0x79B0C6BA) | 0x024C3841 | aa
        Q[QOFF + 3] = (xrng() & 0x19300210) | 0x2603096D | (Q[QOFF + 2] & 0x80000082)
        Q[QOFF + 4] = (xrng() & 0x10300000) | 0xE4CAE30C | (Q[QOFF + 3] & 0x01000030)
        Q[QOFF + 5] = (xrng() & 0x10000000) | 0x63494061 | (Q[QOFF + 4] & 0x00300000)
        Q[QOFF + 6] = 0x7DEAFF68
        Q[QOFF + 7] = (xrng() & 0x20444000) | 0x09091EE0
        Q[QOFF + 8] = (xrng() & 0x09040000) | 0xB2529F6D
        Q[QOFF + 9] = (xrng() & 0x00040000) | 0x10885184
        Q[QOFF + 10] = (xrng() & 0x00000080) | 0x428AFB11 | (Q[QOFF + 9] & 0x00040000)
        Q[QOFF + 11] = (xrng() & 0x128A8110) | 0x6571266B | (Q[QOFF + 10] & 0x00000080)
        Q[QOFF + 12] = (xrng() & 0x3EF38D7F) | 0x00003080 | ((~Q[QOFF + 11]) & 0x00080000)
        Q[QOFF + 13] = (xrng() & 0x3EFB1D77) | 0x0004C008
        Q[QOFF + 14] = (xrng() & 0x5FFF5D77) | 0x8000A288
        Q[QOFF + 15] = (xrng() & 0x1EFE7FF7) | 0xE0008000 | ((~Q[QOFF + 14]) & 0x00010000)
        Q[QOFF + 16] = (xrng() & 0x5FFDFFFF) | 0x20000000 | ((~Q[QOFF + 15]) & 0x00020000)

        block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
        block[6] = reverse_step_first_round(Q, 6, 0xA8304613, 17)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)
        block[11] = reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
        block[14] = reverse_step_first_round(Q, 14, 0xA679438E, 17)
        block[15] = reverse_step_first_round(Q, 15, 0x49B40821, 22)

        tt17 = (GG(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + 0xF61E2562) & MASK32
        tt18 = (Q[QOFF + 14] + 0xC040B340 + block[6]) & MASK32
        tt19 = (Q[QOFF + 15] + 0x265E5A51 + block[11]) & MASK32

        tt0 = (FF(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + 0xD76AA478) & MASK32
        tt1 = (Q[QOFF - 2] + 0xE8C7B756) & MASK32
        q1a = (0x02000941 ^ (Q[QOFF + 0] & 0x80000000)) & MASK32

        if _np is not None:
            res = _batch_find_q1_chain_numpy(
                rng,
                q1a=q1a,
                q1_mask=0x7DFDF6BE,
                q2=Q[QOFF + 2],
                q0=Q[QOFF + 0],
                q_minus1=Q[QOFF - 1],
                tt1=tt1,
                q16=Q[QOFF + 16],
                q15=Q[QOFF + 15],
                tt17=tt17,
                tt18=tt18,
                tt19=tt19,
                tt0=tt0,
                q17_xor_mask=0x80008008,
                q17_xor_expect=0x80000000,
                q17_forbid_mask=0x00020000,
                q18_xor_mask=0xA0020000,
                q18_xor_expect=0x80020000,
                q19_and_mask=0x80020000,
                q19_and_expect=0x00000000,
                q20_xor_mask=0x80040000,
                q20_xor_expect=0x00040000,
                want_q21=True,
                block5=block[5],
                q21_xor_mask=0x80020000,
                q21_xor_expect=0x00000000,
            )
            if res is None:
                continue
            q1, m0, m1, q17, q18, q19, q20, q21 = res
            Q[QOFF + 1] = q1
            Q[QOFF + 17] = q17
            Q[QOFF + 18] = q18
            Q[QOFF + 19] = q19
            Q[QOFF + 20] = q20
            Q[QOFF + 21] = q21
            block[0] = m0
            block[1] = m1
        else:
            counter = 0
            while counter < (1 << 12):
                counter += 1
                q1 = (q1a | (xrng() & 0x7DFDF6BE)) & MASK32
                m1 = (Q[QOFF + 2] - q1) & MASK32
                m1 = (ror(m1, 12) - FF(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1) & MASK32

                q16 = Q[QOFF + 16]
                q17 = (rol((tt17 + m1) & MASK32, 5) + q16) & MASK32
                if 0x80000000 != ((q17 ^ q16) & 0x80008008):
                    continue
                if 0 != (q17 & 0x00020000):
                    continue

                q18 = (rol((GG(q17, q16, Q[QOFF + 15]) + tt18) & MASK32, 9) + q17) & MASK32
                if 0x80020000 != ((q18 ^ q17) & 0xA0020000):
                    continue

                q19 = (rol((GG(q18, q17, q16) + tt19) & MASK32, 14) + q18) & MASK32
                if 0 != (q19 & 0x80020000):
                    continue

                m0 = (q1 - Q[QOFF + 0]) & MASK32
                m0 = (ror(m0, 7) - tt0) & MASK32

                q20 = (rol((GG(q19, q18, q17) + q16 + 0xE9B6C7AA + m0) & MASK32, 20) + q19) & MASK32
                if 0x00040000 != ((q20 ^ q19) & 0x80040000):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1

                block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = (rol((GG(q20, q19, q18) + q17 + 0xD62F105D + block[5]) & MASK32, 5) + q20) & MASK32
                if 0 != ((q21 ^ q20) & 0x80020000):
                    continue
                Q[QOFF + 21] = q21
                counter = 0
                break

            if counter != 0:
                continue

        q9b = Q[QOFF + 9]
        q10b = Q[QOFF + 10]

        block[2] = reverse_step_first_round(Q, 2, 0x242070DB, 17)
        block[3] = reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
        block[4] = reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)

        tt10 = (Q[QOFF + 7] + 0xFFFF5BB1) & MASK32
        tt22 = (GG(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + 0x02441453) & MASK32
        tt23 = (Q[QOFF + 19] + 0xD8A1E681 + block[15]) & MASK32
        tt24 = (Q[QOFF + 20] + 0xE7D3FBC8 + block[4]) & MASK32

        for k10 in range(1 << 4):
            mask = q9q10mask[k10]
            q10 = (q10b | (mask & 0x08000004)) & MASK32
            q9 = (q9b | (mask & 0x00004200)) & MASK32

            m10 = ror((Q[QOFF + 11] - q10) & MASK32, 17)
            m10 = (m10 - FF(q10, q9, Q[QOFF + 8]) - tt10) & MASK32

            aa = Q[QOFF + 21]
            dd = (rol((tt22 + m10) & MASK32, 9) + aa) & MASK32
            if 0 != (dd & 0x80000000):
                continue

            bb = Q[QOFF + 20]
            cc = (tt23 + GG(dd, aa, bb)) & MASK32
            if 0 != (cc & 0x20000):
                continue
            cc = (rol(cc, 14) + dd) & MASK32
            if 0 != (cc & 0x80000000):
                continue

            bb = (rol((tt24 + GG(cc, dd, aa)) & MASK32, 20) + cc) & MASK32
            if 0 == (bb & 0x80000000):
                continue

            block[10] = m10
            Q[QOFF + 9] = q9
            Q[QOFF + 10] = q10
            block[13] = reverse_step_first_round(Q, 13, 0xFD987193, 12)

            iv1 = (IV0, IV1, IV2, IV3)
            iv2 = (
                (IV0 + (1 << 31)) & MASK32,
                (IV1 + (1 << 31) - (1 << 25)) & MASK32,
                (IV2 + (1 << 31) - (1 << 25)) & MASK32,
                (IV3 + (1 << 31) - (1 << 25)) & MASK32,
            )

            if _np is not None:
                maybe = _try_block1_q9_variants_numpy(
                    q9_base=q9,
                    q9_masks_np=_S10_Q9MASK_NP,
                    q5=Q[QOFF + 5],
                    q6=Q[QOFF + 6],
                    q7=Q[QOFF + 7],
                    q8=Q[QOFF + 8],
                    q10=q10,
                    q11=Q[QOFF + 11],
                    q12=Q[QOFF + 12],
                    q13=Q[QOFF + 13],
                    a0=aa,
                    b0=bb,
                    c0=cc,
                    d0=dd,
                    block=block,
                    hh_step34_bit15_required=0,
                    iv1=iv1,
                    iv2=iv2,
                    delta_apply=apply_delta_block0,
                )
                if maybe is not None:
                    return maybe
                continue

            for k9 in range(1 << 10):
                a, b, c, d = aa, bb, cc, dd
                Q[QOFF + 9] = (q9 ^ q9mask[k9]) & MASK32
                block[8] = reverse_step_first_round(Q, 8, 0x698098D8, 7)
                block[9] = reverse_step_first_round(Q, 9, 0x8B44F7AF, 12)
                block[12] = reverse_step_first_round(Q, 12, 0x6B901122, 7)

                if _tail_rounds_common(a, b, c, d, block, hh_step34_bit15_required=0) is None:
                    continue

                out1 = md5_compress(iv1, block)
                out2 = md5_compress(iv2, apply_delta_block0(block))
                if out1 == out2:
                    return list(block)


def find_block1_stevens_11(rng: XorShift64, IV: Tuple[int, int, int, int]) -> List[int]:
    """HashClash `find_block1_stevens_11`: returns msg2block1 words."""
    IV0, IV1, IV2, IV3 = (u32(IV[0]), u32(IV[1]), u32(IV[2]), u32(IV[3]))
    Q = [0] * 68
    Q[0], Q[1], Q[2], Q[3] = IV0, IV3, IV2, IV1
    block = [0] * 16
    xrng = rng.next_u32
    q9q10mask = _S11_Q9Q10MASK
    q9mask = _S11_Q9MASK

    while True:
        aa = Q[QOFF] & 0x80000000

        Q[QOFF + 2] = (xrng() & 0x75BEF63E) | 0x0A410041 | aa
        Q[QOFF + 3] = (xrng() & 0x10345614) | 0x0202A9E1 | (Q[QOFF + 2] & 0x84000002)
        Q[QOFF + 4] = (xrng() & 0x00145400) | 0xE84BA909 | (Q[QOFF + 3] & 0x00000014)
        Q[QOFF + 5] = (xrng() & 0x80000000) | 0x75E90B1D | (Q[QOFF + 4] & 0x00145400)
        Q[QOFF + 6] = 0x7C23FF5A | (Q[QOFF + 5] & 0x80000000)
        Q[QOFF + 7] = (xrng() & 0x40000880) | 0x114BF41A
        Q[QOFF + 8] = (xrng() & 0x00002090) | 0xB352DD01
        Q[QOFF + 9] = (xrng() & 0x00044000) | 0x7A803124
        Q[QOFF + 10] = (xrng() & 0x00002000) | 0xF28A92C9 | (Q[QOFF + 9] & 0x00044000)
        Q[QOFF + 11] = (xrng() & 0x128A8108) | 0xC5710ED7 | (Q[QOFF + 10] & 0x00002000)
        Q[QOFF + 12] = (xrng() & 0x9EDB8D7F) | 0x20003080 | ((~Q[QOFF + 11]) & 0x00200000)
        Q[QOFF + 13] = (xrng() & 0x3EFB1D77) | 0x4004C008 | (Q[QOFF + 12] & 0x80000000)
        Q[QOFF + 14] = (xrng() & 0x1FFF5D77) | 0x0000A288
        Q[QOFF + 15] = (xrng() & 0x1EFE7FF7) | 0x20008000 | ((~Q[QOFF + 14]) & 0x00010000)
        Q[QOFF + 16] = (xrng() & 0x1FFDFFFF) | 0x20000000 | ((~Q[QOFF + 15]) & 0x40020000)

        block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
        block[6] = reverse_step_first_round(Q, 6, 0xA8304613, 17)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)
        block[11] = reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
        block[14] = reverse_step_first_round(Q, 14, 0xA679438E, 17)
        block[15] = reverse_step_first_round(Q, 15, 0x49B40821, 22)

        tt17 = (GG(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + 0xF61E2562) & MASK32
        tt18 = (Q[QOFF + 14] + 0xC040B340 + block[6]) & MASK32
        tt19 = (Q[QOFF + 15] + 0x265E5A51 + block[11]) & MASK32

        tt0 = (FF(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + 0xD76AA478) & MASK32
        tt1 = (Q[QOFF - 2] + 0xE8C7B756) & MASK32
        q1a = (0x02000861 ^ (Q[QOFF + 0] & 0x80000020)) & MASK32

        if _np is not None:
            res = _batch_find_q1_chain_numpy(
                rng,
                q1a=q1a,
                q1_mask=0x7DFFF79E,
                q2=Q[QOFF + 2],
                q0=Q[QOFF + 0],
                q_minus1=Q[QOFF - 1],
                tt1=tt1,
                q16=Q[QOFF + 16],
                q15=Q[QOFF + 15],
                tt17=tt17,
                tt18=tt18,
                tt19=tt19,
                tt0=tt0,
                q17_xor_mask=0xC0008008,
                q17_xor_expect=0x40000000,
                q17_forbid_mask=0x00020000,
                q18_xor_mask=0xA0020000,
                q18_xor_expect=0x80020000,
                q19_and_mask=0x80020000,
                q19_and_expect=0x80000000,
                q20_xor_mask=0x80040000,
                q20_xor_expect=0x00040000,
                want_q21=True,
                block5=block[5],
                q21_xor_mask=0x80020000,
                q21_xor_expect=0x00000000,
            )
            if res is None:
                continue
            q1, m0, m1, q17, q18, q19, q20, q21 = res
            Q[QOFF + 1] = q1
            Q[QOFF + 17] = q17
            Q[QOFF + 18] = q18
            Q[QOFF + 19] = q19
            Q[QOFF + 20] = q20
            Q[QOFF + 21] = q21
            block[0] = m0
            block[1] = m1
        else:
            counter = 0
            while counter < (1 << 12):
                counter += 1
                q1 = (q1a | (xrng() & 0x7DFFF79E)) & MASK32
                m1 = (Q[QOFF + 2] - q1) & MASK32
                m1 = (ror(m1, 12) - FF(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1) & MASK32

                q16 = Q[QOFF + 16]
                q17 = (rol((tt17 + m1) & MASK32, 5) + q16) & MASK32
                if 0x40000000 != ((q17 ^ q16) & 0xC0008008):
                    continue
                if 0 != (q17 & 0x00020000):
                    continue

                q18 = (rol((GG(q17, q16, Q[QOFF + 15]) + tt18) & MASK32, 9) + q17) & MASK32
                if 0x80020000 != ((q18 ^ q17) & 0xA0020000):
                    continue

                q19 = (rol((GG(q18, q17, q16) + tt19) & MASK32, 14) + q18) & MASK32
                if 0x80000000 != (q19 & 0x80020000):
                    continue

                m0 = (q1 - Q[QOFF + 0]) & MASK32
                m0 = (ror(m0, 7) - tt0) & MASK32

                q20 = (rol((GG(q19, q18, q17) + q16 + 0xE9B6C7AA + m0) & MASK32, 20) + q19) & MASK32
                if 0x00040000 != ((q20 ^ q19) & 0x80040000):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1

                block[5] = reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = (rol((GG(q20, q19, q18) + q17 + 0xD62F105D + block[5]) & MASK32, 5) + q20) & MASK32
                if 0 != ((q21 ^ q20) & 0x80020000):
                    continue
                Q[QOFF + 21] = q21
                counter = 0
                break

            if counter != 0:
                continue

        q9b = Q[QOFF + 9]
        q10b = Q[QOFF + 10]

        block[2] = reverse_step_first_round(Q, 2, 0x242070DB, 17)
        block[3] = reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
        block[4] = reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
        block[7] = reverse_step_first_round(Q, 7, 0xFD469501, 22)

        tt10 = (Q[QOFF + 7] + 0xFFFF5BB1) & MASK32
        tt22 = (GG(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + 0x02441453) & MASK32
        tt23 = (Q[QOFF + 19] + 0xD8A1E681 + block[15]) & MASK32
        tt24 = (Q[QOFF + 20] + 0xE7D3FBC8 + block[4]) & MASK32

        for k10 in range(1 << 5):
            mask = q9q10mask[k10]
            q10 = (q10b | (mask & 0x08000040)) & MASK32
            q9 = (q9b | (mask & 0x80000280)) & MASK32

            m10 = ror((Q[QOFF + 11] - q10) & MASK32, 17)
            m10 = (m10 - FF(q10, q9, Q[QOFF + 8]) - tt10) & MASK32

            aa = Q[QOFF + 21]
            dd = (rol((tt22 + m10) & MASK32, 9) + aa) & MASK32
            if 0 == (dd & 0x80000000):
                continue

            bb = Q[QOFF + 20]
            cc = (tt23 + GG(dd, aa, bb)) & MASK32
            if 0 != (cc & 0x20000):
                continue
            cc = (rol(cc, 14) + dd) & MASK32
            if 0 != (cc & 0x80000000):
                continue

            bb = (rol((tt24 + GG(cc, dd, aa)) & MASK32, 20) + cc) & MASK32
            if 0 == (bb & 0x80000000):
                continue

            block[10] = m10
            Q[QOFF + 9] = q9
            Q[QOFF + 10] = q10
            block[13] = reverse_step_first_round(Q, 13, 0xFD987193, 12)

            iv1 = (IV0, IV1, IV2, IV3)
            iv2 = (
                (IV0 + (1 << 31)) & MASK32,
                (IV1 + (1 << 31) - (1 << 25)) & MASK32,
                (IV2 + (1 << 31) - (1 << 25)) & MASK32,
                (IV3 + (1 << 31) - (1 << 25)) & MASK32,
            )

            if _np is not None:
                maybe = _try_block1_q9_variants_numpy(
                    q9_base=q9,
                    q9_masks_np=_S11_Q9MASK_NP,
                    q5=Q[QOFF + 5],
                    q6=Q[QOFF + 6],
                    q7=Q[QOFF + 7],
                    q8=Q[QOFF + 8],
                    q10=q10,
                    q11=Q[QOFF + 11],
                    q12=Q[QOFF + 12],
                    q13=Q[QOFF + 13],
                    a0=aa,
                    b0=bb,
                    c0=cc,
                    d0=dd,
                    block=block,
                    hh_step34_bit15_required=0,
                    iv1=iv1,
                    iv2=iv2,
                    delta_apply=apply_delta_block0,
                )
                if maybe is not None:
                    return maybe
                continue

            for k9 in range(1 << 9):
                a, b, c, d = aa, bb, cc, dd
                Q[QOFF + 9] = (q9 ^ q9mask[k9]) & MASK32
                block[8] = reverse_step_first_round(Q, 8, 0x698098D8, 7)
                block[9] = reverse_step_first_round(Q, 9, 0x8B44F7AF, 12)
                block[12] = reverse_step_first_round(Q, 12, 0x6B901122, 7)

                if _tail_rounds_common(a, b, c, d, block, hh_step34_bit15_required=0) is None:
                    continue

                out1 = md5_compress(iv1, block)
                out2 = md5_compress(iv2, apply_delta_block0(block))
                if out1 == out2:
                    return list(block)


def find_block1(rng: XorShift64, IV: Tuple[int, int, int, int]) -> List[int]:
    """HashClash `find_block1`: returns msg1block1 words."""
    IV = tuple(u32(x) for x in IV)
    if (
        ((IV[1] ^ IV[2]) & (1 << 31)) == 0
        and ((IV[1] ^ IV[3]) & (1 << 31)) == 0
        and (IV[3] & (1 << 25)) == 0
        and (IV[2] & (1 << 25)) == 0
        and (IV[1] & (1 << 25)) == 0
        and ((IV[2] ^ IV[1]) & 1) == 0
    ):
        IV2 = (
            (IV[0] + (1 << 31)) & MASK32,
            (IV[1] + (1 << 31) + (1 << 25)) & MASK32,
            (IV[2] + (1 << 31) + (1 << 25)) & MASK32,
            (IV[3] + (1 << 31) + (1 << 25)) & MASK32,
        )
        if (IV[1] & (1 << 6)) != 0 and (IV[1] & 1) != 0:
            msg2_block = find_block1_stevens_11(rng, IV2)
        elif (IV[1] & (1 << 6)) != 0 and (IV[1] & 1) == 0:
            msg2_block = find_block1_stevens_10(rng, IV2)
        elif (IV[1] & (1 << 6)) == 0 and (IV[1] & 1) != 0:
            msg2_block = find_block1_stevens_01(rng, IV2)
        else:
            msg2_block = find_block1_stevens_00(rng, IV2)
        # Convert msg2block1 -> msg1block1 (as in block1.cpp)
        msg1_block = apply_delta_block0(msg2_block)
        return msg1_block

    return find_block1_wang(rng, IV)


def find_collision_blocks(
    ihv: Tuple[int, int, int, int],
    *,
    seed: int | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Generate the two 64-byte blocks for msg1 (fastcoll-compatible).
    msg2 is obtained by applying the standard fastcoll deltas.
    """
    rng = rng_from_seed(seed)
    b0 = find_block0(rng, ihv)
    ihv_after_b0 = md5_compress(ihv, b0)
    b1 = find_block1(rng, ihv_after_b0)
    return b0, b1


def _worker_find_collision_blocks(args: Tuple[Tuple[int, int, int, int], int]) -> Tuple[List[int], List[int]]:
    ihv, seed = args
    return find_collision_blocks(ihv, seed=seed)


def find_collision_blocks_parallel(
    ihv: Tuple[int, int, int, int],
    *,
    seed: int | None = None,
    jobs: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    Run multiple independent searches in parallel and return the first result.

    Notes:
    - Uses `fork` when available (faster start + COW sharing of mask tables).
    - `jobs<=1` falls back to single-process search.
    """
    jobs = int(jobs)
    if jobs <= 1:
        return find_collision_blocks(ihv, seed=seed)

    import multiprocessing as mp

    methods = mp.get_all_start_methods()
    ctx = mp.get_context("fork" if "fork" in methods else None)

    base_seed = seed if seed is not None else (time.time_ns() & MASK64)
    base_seed &= MASK64

    # SplitMix64 to decorrelate worker RNG streams.
    def _splitmix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & MASK64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    tasks = [(ihv, _splitmix64((base_seed + i) & MASK64)) for i in range(jobs)]

    pool = ctx.Pool(processes=jobs)
    try:
        it = pool.imap_unordered(_worker_find_collision_blocks, tasks, chunksize=1)
        result = next(it)
        return result
    finally:
        pool.terminate()
        pool.join()
