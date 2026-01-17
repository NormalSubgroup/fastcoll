"""
HashClash md5_fastcoll compatible implementation using Numba JIT.

Sources
- HashClash repository https://github.com/cr-marcstevens/hashclash
- Code under src/md5fastcoll such as block1wang.cpp and block1stevens*.cpp
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np

try:
    if os.getenv("MD5FASTCOLL_NO_NUMBA") == "1":
        raise ImportError("MD5FASTCOLL_NO_NUMBA=1")
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


MASK32_U32 = np.uint32(0xFFFFFFFF)
MASK64 = 0xFFFFFFFFFFFFFFFF
QOFF = 3


def numba_available() -> bool:
    return njit is not None


def _seed_to_u32_pair(seed: int | None) -> tuple[int, int]:
    if seed is None:
        s = time.time_ns() & MASK64
        seed1 = int(s & 0xFFFFFFFF)
        seed2 = int((s >> 32) & 0xFFFFFFFF)
    else:
        seed1 = int(seed & 0xFFFFFFFF)
        seed2 = int((seed >> 32) & 0xFFFFFFFF)
        if seed1 == 0 and seed2 == 0:
            seed2 = 0x12345678
    if seed1 == 0 and seed2 == 0:
        seed2 = 0x12345678
    return seed1, seed2


_MD5_AC = np.array(
    (
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
    ),
    dtype=np.uint32,
)

_MD5_RC = np.array(
    (
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
    ),
    dtype=np.int64,
)

_MD5_G = np.array(
    (
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
    ),
    dtype=np.int64,
)

# Numba sometimes behaves unexpectedly when indexing numpy global arrays inside `@njit`
# functions on some platforms. Keep tuple-based copies for deterministic typing.
_MD5_AC_T = tuple(int(x) for x in _MD5_AC.tolist())
_MD5_RC_T = tuple(int(x) for x in _MD5_RC.tolist())
_MD5_G_T = tuple(int(x) for x in _MD5_G.tolist())


def _mk_masks(count: int, expr) -> np.ndarray:
    return np.array([expr(k) & 0xFFFFFFFF for k in range(count)], dtype=np.uint32)


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
_S01_Q9MASK = _mk_masks(
    1 << 9,
    lambda k: (((k << 1) ^ (k << 7) ^ (k << 9) ^ (k << 12) ^ (k << 15) ^ (k << 19) ^ (k << 22)) & 0x44310D02),
)

_S10_Q9Q10MASK = _mk_masks(1 << 4, lambda k: (((k << 2) ^ (k << 8) ^ (k << 11) ^ (k << 25)) & 0x08004204))
_S10_Q9MASK = _mk_masks(
    1 << 10,
    lambda k: (((k << 1) ^ (k << 2) ^ (k << 3) ^ (k << 7) ^ (k << 12) ^ (k << 15) ^ (k << 18) ^ (k << 20)) & 0x2471042A),
)

_S11_Q9Q10MASK = _mk_masks(1 << 5, lambda k: (((k << 5) ^ (k << 6) ^ (k << 7) ^ (k << 24) ^ (k << 27)) & 0x880002A0))
_S11_Q9MASK = _mk_masks(1 << 9, lambda k: (((k << 1) ^ (k << 3) ^ (k << 8) ^ (k << 12) ^ (k << 15) ^ (k << 18)) & 0x04710C12))


if njit is not None:

    @njit(cache=True, inline="always")
    def _rol_u32(x: np.uint32, n: int) -> np.uint32:
        y = np.uint32(x)
        return (y << n) | (y >> (32 - n))

    @njit(cache=True, inline="always")
    def _ror_u32(x: np.uint32, n: int) -> np.uint32:
        y = np.uint32(x)
        return (y >> n) | (y << (32 - n))

    @njit(cache=True, inline="always")
    def _ff(b: np.uint32, c: np.uint32, d: np.uint32) -> np.uint32:
        bb = np.uint32(b)
        cc = np.uint32(c)
        dd = np.uint32(d)
        return dd ^ (bb & (cc ^ dd))

    @njit(cache=True, inline="always")
    def _gg(b: np.uint32, c: np.uint32, d: np.uint32) -> np.uint32:
        bb = np.uint32(b)
        cc = np.uint32(c)
        dd = np.uint32(d)
        return cc ^ (dd & (bb ^ cc))

    @njit(cache=True, inline="always")
    def _hh(b: np.uint32, c: np.uint32, d: np.uint32) -> np.uint32:
        bb = np.uint32(b)
        cc = np.uint32(c)
        dd = np.uint32(d)
        return bb ^ cc ^ dd

    @njit(cache=True, inline="always")
    def _ii(b: np.uint32, c: np.uint32, d: np.uint32) -> np.uint32:
        bb = np.uint32(b)
        cc = np.uint32(c)
        dd = np.uint32(d)
        not_d = np.uint32(~dd)
        return cc ^ (bb | not_d)

    @njit(cache=True, inline="always")
    def _rng_next_u32(state: np.ndarray) -> np.uint32:
        s1 = state[0]
        s2 = state[1]
        t = np.uint32(s1 ^ np.uint32(s1 << 10))
        s1 = s2
        s2 = np.uint32((s2 ^ (s2 >> 10)) ^ (t ^ (t >> 13)))
        state[0] = s1
        state[1] = s2
        return s1

    @njit(cache=True, inline="always")
    def _reverse_step_first_round(Q: np.ndarray, t: int, ac: int, rc: int) -> np.uint32:
        Rt = Q[QOFF + t + 1] - Q[QOFF + t]
        Tt = _ror_u32(Rt, rc)
        f = _ff(Q[QOFF + t], Q[QOFF + t - 1], Q[QOFF + t - 2])
        return Tt - f - Q[QOFF + t - 3] - np.uint32(ac & 0xFFFFFFFF)

    @njit(cache=True)
    def md5_compress_u32(ihv: np.ndarray, block: np.ndarray) -> tuple[np.uint32, np.uint32, np.uint32, np.uint32]:
        a0 = ihv[0]
        b0 = ihv[1]
        c0 = ihv[2]
        d0 = ihv[3]
        a = a0
        b = b0
        c = c0
        d = d0
        for i in range(64):
            if i < 16:
                f = d ^ (b & (c ^ d))
            elif i < 32:
                f = c ^ (d & (b ^ c))
            elif i < 48:
                f = b ^ c ^ d
            else:
                f = c ^ (b | np.uint32(~d))

            g = _MD5_G_T[i]
            tmp = np.uint32(a + f + np.uint32(_MD5_AC_T[i]) + block[g])
            tmp = _rol_u32(tmp, _MD5_RC_T[i])
            tmp = np.uint32(tmp + b)
            a, d, c, b = d, c, b, tmp
        return (np.uint32(a0 + a), np.uint32(b0 + b), np.uint32(c0 + c), np.uint32(d0 + d))

    @njit(cache=True)
    def _tail_rounds_common(
        a: np.uint32,
        b: np.uint32,
        c: np.uint32,
        d: np.uint32,
        block: np.ndarray,
        hh_step34_bit15_required: int,
    ) -> tuple[bool, np.uint32, np.uint32, np.uint32, np.uint32]:
        # Step 24..31 (GG)
        a = np.uint32(_rol_u32(a + _gg(b, c, d) + block[9] + np.uint32(0x21E1CDE6), 5) + b)
        d = np.uint32(_rol_u32(d + _gg(a, b, c) + block[14] + np.uint32(0xC33707D6), 9) + a)
        c = np.uint32(_rol_u32(c + _gg(d, a, b) + block[3] + np.uint32(0xF4D50D87), 14) + d)
        b = np.uint32(_rol_u32(b + _gg(c, d, a) + block[8] + np.uint32(0x455A14ED), 20) + c)
        a = np.uint32(_rol_u32(a + _gg(b, c, d) + block[13] + np.uint32(0xA9E3E905), 5) + b)
        d = np.uint32(_rol_u32(d + _gg(a, b, c) + block[2] + np.uint32(0xFCEFA3F8), 9) + a)
        c = np.uint32(_rol_u32(c + _gg(d, a, b) + block[7] + np.uint32(0x676F02D9), 14) + d)
        b = np.uint32(_rol_u32(b + _gg(c, d, a) + block[12] + np.uint32(0x8D2A4C8A), 20) + c)

        # Step 32..47 (HH)
        a = np.uint32(_rol_u32(a + _hh(b, c, d) + block[5] + np.uint32(0xFFFA3942), 4) + b)
        d = np.uint32(_rol_u32(d + _hh(a, b, c) + block[8] + np.uint32(0x8771F681), 11) + a)
        c = np.uint32(c + _hh(d, a, b) + block[11] + np.uint32(0x6D9D6122))
        if ((c >> 15) & np.uint32(1)) != np.uint32(hh_step34_bit15_required & 1):
            return (False, a, b, c, d)
        c = np.uint32(_rol_u32(c, 16) + d)
        b = np.uint32(_rol_u32(b + _hh(c, d, a) + block[14] + np.uint32(0xFDE5380C), 23) + c)

        a = np.uint32(_rol_u32(a + _hh(b, c, d) + block[1] + np.uint32(0xA4BEEA44), 4) + b)
        d = np.uint32(_rol_u32(d + _hh(a, b, c) + block[4] + np.uint32(0x4BDECFA9), 11) + a)
        c = np.uint32(_rol_u32(c + _hh(d, a, b) + block[7] + np.uint32(0xF6BB4B60), 16) + d)
        b = np.uint32(_rol_u32(b + _hh(c, d, a) + block[10] + np.uint32(0xBEBFBC70), 23) + c)
        a = np.uint32(_rol_u32(a + _hh(b, c, d) + block[13] + np.uint32(0x289B7EC6), 4) + b)
        d = np.uint32(_rol_u32(d + _hh(a, b, c) + block[0] + np.uint32(0xEAA127FA), 11) + a)
        c = np.uint32(_rol_u32(c + _hh(d, a, b) + block[3] + np.uint32(0xD4EF3085), 16) + d)
        b = np.uint32(_rol_u32(b + _hh(c, d, a) + block[6] + np.uint32(0x04881D05), 23) + c)
        a = np.uint32(_rol_u32(a + _hh(b, c, d) + block[9] + np.uint32(0xD9D4D039), 4) + b)
        d = np.uint32(_rol_u32(d + _hh(a, b, c) + block[12] + np.uint32(0xE6DB99E5), 11) + a)
        c = np.uint32(_rol_u32(c + _hh(d, a, b) + block[15] + np.uint32(0x1FA27CF8), 16) + d)
        b = np.uint32(_rol_u32(b + _hh(c, d, a) + block[2] + np.uint32(0xC4AC5665), 23) + c)
        if (b ^ d) & np.uint32(0x80000000):
            return (False, a, b, c, d)

        # Step 48..63 (II)
        a = np.uint32(_rol_u32(a + _ii(b, c, d) + block[0] + np.uint32(0xF4292244), 6) + b)
        if (a ^ c) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        d = np.uint32(_rol_u32(d + _ii(a, b, c) + block[7] + np.uint32(0x432AFF97), 10) + a)
        if ((b ^ d) & np.uint32(0x80000000)) == np.uint32(0):
            return (False, a, b, c, d)
        c = np.uint32(_rol_u32(c + _ii(d, a, b) + block[14] + np.uint32(0xAB9423A7), 15) + d)
        if (a ^ c) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        b = np.uint32(_rol_u32(b + _ii(c, d, a) + block[5] + np.uint32(0xFC93A039), 21) + c)
        if (b ^ d) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        a = np.uint32(_rol_u32(a + _ii(b, c, d) + block[12] + np.uint32(0x655B59C3), 6) + b)
        if (a ^ c) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        d = np.uint32(_rol_u32(d + _ii(a, b, c) + block[3] + np.uint32(0x8F0CCC92), 10) + a)
        if (b ^ d) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        c = np.uint32(_rol_u32(c + _ii(d, a, b) + block[10] + np.uint32(0xFFEFF47D), 15) + d)
        if (a ^ c) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        b = np.uint32(_rol_u32(b + _ii(c, d, a) + block[1] + np.uint32(0x85845DD1), 21) + c)
        if (b ^ d) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        a = np.uint32(_rol_u32(a + _ii(b, c, d) + block[8] + np.uint32(0x6FA87E4F), 6) + b)
        if (a ^ c) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        d = np.uint32(_rol_u32(d + _ii(a, b, c) + block[15] + np.uint32(0xFE2CE6E0), 10) + a)
        if (b ^ d) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        c = np.uint32(_rol_u32(c + _ii(d, a, b) + block[6] + np.uint32(0xA3014314), 15) + d)
        if (a ^ c) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        b = np.uint32(_rol_u32(b + _ii(c, d, a) + block[13] + np.uint32(0x4E0811A1), 21) + c)
        if ((b ^ d) & np.uint32(0x80000000)) == np.uint32(0):
            return (False, a, b, c, d)
        a = np.uint32(_rol_u32(a + _ii(b, c, d) + block[4] + np.uint32(0xF7537E82), 6) + b)
        if (a ^ c) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        d = np.uint32(_rol_u32(d + _ii(a, b, c) + block[11] + np.uint32(0xBD3AF235), 10) + a)
        if (b ^ d) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        c = np.uint32(_rol_u32(c + _ii(d, a, b) + block[2] + np.uint32(0x2AD7D2BB), 15) + d)
        if (a ^ c) & np.uint32(0x80000000):
            return (False, a, b, c, d)
        b = np.uint32(_rol_u32(b + _ii(c, d, a) + block[9] + np.uint32(0xEB86D391), 21) + c)
        return (True, a, b, c, d)

    @njit(cache=True, inline="always")
    def _apply_delta_block0_inplace(block: np.ndarray) -> None:
        block[4] = block[4] + np.uint32(1 << 31)
        block[11] = block[11] + np.uint32(1 << 15)
        block[14] = block[14] + np.uint32(1 << 31)

    @njit(cache=True, inline="always")
    def _apply_delta_block1_inplace(block: np.ndarray) -> None:
        block[4] = block[4] + np.uint32(1 << 31)
        block[11] = block[11] - np.uint32(1 << 15)
        block[14] = block[14] + np.uint32(1 << 31)

    @njit(cache=True)
    def _try_block0_q9mask(
        IV: np.ndarray,
        q7: np.uint32,
        q8: np.uint32,
        q9_base: np.uint32,
        q10: np.uint32,
        q11: np.uint32,
        q12: np.uint32,
        tt8: np.uint32,
        tt9: np.uint32,
        tt12: np.uint32,
        aa: np.uint32,
        bb: np.uint32,
        cc: np.uint32,
        dd: np.uint32,
        block: np.ndarray,
    ) -> bool:
        for k in range(_B0_Q9MASK.shape[0]):
            q9 = q9_base ^ _B0_Q9MASK[k]
            block[12] = tt12 - _ff(q12, q11, q10) - q9
            block[8] = _ror_u32(q9 - q8, 7) - tt8
            block[9] = _ror_u32(q10 - q9, 12) - _ff(q9, q8, q7) - tt9

            ok, a, b, c, d = _tail_rounds_common(aa, bb, cc, dd, block, 0)
            if not ok:
                continue

            IHV1 = b + IV[1]
            IHV2 = c + IV[2]
            IHV3 = d + IV[3]

            wang = True
            if np.uint32(0x02000000) != ((IHV2 ^ IHV1) & np.uint32(0x86000000)):
                wang = False
            if np.uint32(0) != ((IHV1 ^ IHV3) & np.uint32(0x82000000)):
                wang = False
            if np.uint32(0) != (IHV1 & np.uint32(0x06000020)):
                wang = False

            stevens = True
            if ((IHV1 ^ IHV2) & np.uint32(0x80000000)) != np.uint32(0) or ((IHV1 ^ IHV3) & np.uint32(0x80000000)) != np.uint32(0):
                stevens = False
            if (
                (IHV3 & np.uint32(1 << 25)) != np.uint32(0)
                or (IHV2 & np.uint32(1 << 25)) != np.uint32(0)
                or (IHV1 & np.uint32(1 << 25)) != np.uint32(0)
                or ((IHV2 ^ IHV1) & np.uint32(1)) != np.uint32(0)
            ):
                stevens = False

            if not (wang or stevens):
                continue

            out1 = md5_compress_u32(IV, block)

            block2 = block.copy()
            _apply_delta_block0_inplace(block2)
            out2 = md5_compress_u32(IV, block2)

            if (
                out2[0] == out1[0] + np.uint32(1 << 31)
                and out2[1] == out1[1] + np.uint32((1 << 31) + (1 << 25))
                and out2[2] == out1[2] + np.uint32((1 << 31) + (1 << 25))
                and out2[3] == out1[3] + np.uint32((1 << 31) + (1 << 25))
            ):
                return True

        return False

    @njit(cache=True)
    def find_block0_numba(rng_state: np.ndarray, IV: np.ndarray) -> np.ndarray:
        Q = np.zeros(68, dtype=np.uint32)
        Q[0] = IV[0]
        Q[1] = IV[3]
        Q[2] = IV[2]
        Q[3] = IV[1]
        block = np.zeros(16, dtype=np.uint32)

        while True:
            Q[QOFF + 1] = _rng_next_u32(rng_state)
            Q[QOFF + 3] = (_rng_next_u32(rng_state) & np.uint32(0xFE87BC3F)) | np.uint32(0x017841C0)
            Q[QOFF + 4] = (
                (_rng_next_u32(rng_state) & np.uint32(0x44000033))
                | np.uint32(0x000002C0)
                | (Q[QOFF + 3] & np.uint32(0x0287BC00))
            )
            Q[QOFF + 5] = np.uint32(0x41FFFFC8) | (Q[QOFF + 4] & np.uint32(0x04000033))
            Q[QOFF + 6] = np.uint32(0xB84B82D6)
            Q[QOFF + 7] = (_rng_next_u32(rng_state) & np.uint32(0x68000084)) | np.uint32(0x02401B43)
            Q[QOFF + 8] = (
                (_rng_next_u32(rng_state) & np.uint32(0x2B8F6E04))
                | np.uint32(0x005090D3)
                | (np.uint32(~Q[QOFF + 7]) & np.uint32(0x40000000))
            )
            Q[QOFF + 9] = np.uint32(0x20040068) | (Q[QOFF + 8] & np.uint32(0x00020000)) | (np.uint32(~Q[QOFF + 8]) & np.uint32(0x40000000))
            Q[QOFF + 10] = (_rng_next_u32(rng_state) & np.uint32(0x40000000)) | np.uint32(0x1040B089)
            Q[QOFF + 11] = (
                (_rng_next_u32(rng_state) & np.uint32(0x10408008))
                | np.uint32(0x0FBB7F16)
                | (np.uint32(~Q[QOFF + 10]) & np.uint32(0x40000000))
            )
            Q[QOFF + 12] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1ED9DF7F))
                | np.uint32(0x00022080)
                | (np.uint32(~Q[QOFF + 11]) & np.uint32(0x40200000))
            )
            Q[QOFF + 13] = (_rng_next_u32(rng_state) & np.uint32(0x5EFB4F77)) | np.uint32(0x20049008)
            Q[QOFF + 14] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1FFF5F77))
                | np.uint32(0x0000A088)
                | (np.uint32(~Q[QOFF + 13]) & np.uint32(0x40000000))
            )
            Q[QOFF + 15] = (
                (_rng_next_u32(rng_state) & np.uint32(0x5EFE7FF7))
                | np.uint32(0x80008000)
                | (np.uint32(~Q[QOFF + 14]) & np.uint32(0x00010000))
            )
            Q[QOFF + 16] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1FFDFFFF))
                | np.uint32(0xA0000000)
                | (np.uint32(~Q[QOFF + 15]) & np.uint32(0x40020000))
            )

            block[0] = _reverse_step_first_round(Q, 0, 0xD76AA478, 7)
            block[6] = _reverse_step_first_round(Q, 6, 0xA8304613, 17)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)
            block[11] = _reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
            block[14] = _reverse_step_first_round(Q, 14, 0xA679438E, 17)
            block[15] = _reverse_step_first_round(Q, 15, 0x49B40821, 22)

            tt1 = _ff(Q[QOFF + 1], Q[QOFF + 0], Q[QOFF - 1]) + Q[QOFF - 2] + np.uint32(0xE8C7B756)
            tt17 = _gg(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + np.uint32(0xF61E2562)
            tt18 = Q[QOFF + 14] + np.uint32(0xC040B340) + block[6]
            tt19 = Q[QOFF + 15] + np.uint32(0x265E5A51) + block[11]
            tt20 = Q[QOFF + 16] + np.uint32(0xE9B6C7AA) + block[0]
            tt5 = _ror_u32(Q[QOFF + 6] - Q[QOFF + 5], 12) - _ff(Q[QOFF + 5], Q[QOFF + 4], Q[QOFF + 3]) - np.uint32(0x4787C62A)

            counter = 0
            while counter < (1 << 7):
                q16 = Q[QOFF + 16]
                q17 = ((_rng_next_u32(rng_state) & np.uint32(0x3FFD7FF7)) | (q16 & np.uint32(0xC0008008))) ^ np.uint32(0x40000000)
                counter += 1

                q18 = _rol_u32(_gg(q17, q16, Q[QOFF + 15]) + tt18, 9) + q17
                if np.uint32(0x00020000) != ((q18 ^ q17) & np.uint32(0xA0020000)):
                    continue

                q19 = _rol_u32(_gg(q18, q17, q16) + tt19, 14) + q18
                if np.uint32(0x80000000) != (q19 & np.uint32(0x80020000)):
                    continue

                q20 = _rol_u32(_gg(q19, q18, q17) + tt20, 20) + q19
                if np.uint32(0x00040000) != ((q20 ^ q19) & np.uint32(0x80040000)):
                    continue

                block[1] = _ror_u32(q17 - q16, 5) - tt17
                q2 = _rol_u32(block[1] + tt1, 12) + Q[QOFF + 1]
                block[5] = tt5 - q2

                Q[QOFF + 2] = q2
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[2] = _reverse_step_first_round(Q, 2, 0x242070DB, 17)
                counter = 0
                break

            if counter != 0:
                continue

            q4 = Q[QOFF + 4]
            q9backup = Q[QOFF + 9]
            tt21 = _gg(Q[QOFF + 20], Q[QOFF + 19], Q[QOFF + 18]) + Q[QOFF + 17] + np.uint32(0xD62F105D)

            counter2 = 0
            while counter2 < (1 << 4):
                Q[QOFF + 4] = q4 ^ _B0_Q4MASK[counter2]
                counter2 += 1

                block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = _rol_u32(tt21 + block[5], 5) + Q[QOFF + 20]
                if (q21 ^ Q[QOFF + 20]) & np.uint32(0x80020000):
                    continue

                Q[QOFF + 21] = q21
                block[3] = _reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
                block[4] = _reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
                block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)

                tt22 = _gg(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + np.uint32(0x02441453)
                tt23 = Q[QOFF + 19] + np.uint32(0xD8A1E681) + block[15]
                tt24 = Q[QOFF + 20] + np.uint32(0xE7D3FBC8) + block[4]

                tt9 = Q[QOFF + 6] + np.uint32(0x8B44F7AF)
                tt10 = Q[QOFF + 7] + np.uint32(0xFFFF5BB1)
                tt8 = _ff(Q[QOFF + 8], Q[QOFF + 7], Q[QOFF + 6]) + Q[QOFF + 5] + np.uint32(0x698098D8)
                tt12 = _ror_u32(Q[QOFF + 13] - Q[QOFF + 12], 7) - np.uint32(0x6B901122)
                tt13 = _ror_u32(Q[QOFF + 14] - Q[QOFF + 13], 12) - _ff(Q[QOFF + 13], Q[QOFF + 12], Q[QOFF + 11]) - np.uint32(0xFD987193)

                for counter3 in range(1 << 3):
                    q10 = Q[QOFF + 10] ^ (_B0_Q9Q10MASK[counter3] & np.uint32(0x60))
                    Q[QOFF + 9] = q9backup ^ (_B0_Q9Q10MASK[counter3] & np.uint32(0x2000))

                    m10 = _ror_u32(Q[QOFF + 11] - q10, 17)
                    m10 = m10 - _ff(q10, Q[QOFF + 9], Q[QOFF + 8]) - tt10

                    aa = Q[QOFF + 21]
                    dd = _rol_u32(tt22 + m10, 9) + aa
                    if np.uint32(0x80000000) != (dd & np.uint32(0x80000000)):
                        continue

                    bb = Q[QOFF + 20]
                    cc = tt23 + _gg(dd, aa, bb)
                    if cc & np.uint32(0x20000):
                        continue
                    cc = _rol_u32(cc, 14) + dd
                    if cc & np.uint32(0x80000000):
                        continue

                    bb = _rol_u32(tt24 + _gg(cc, dd, aa), 20) + cc
                    if (bb & np.uint32(0x80000000)) == np.uint32(0):
                        continue

                    block[10] = m10
                    block[13] = tt13 - q10

                    if _try_block0_q9mask(
                        IV,
                        Q[QOFF + 7],
                        Q[QOFF + 8],
                        Q[QOFF + 9],
                        q10,
                        Q[QOFF + 11],
                        Q[QOFF + 12],
                        tt8,
                        tt9,
                        tt12,
                        aa,
                        bb,
                        cc,
                        dd,
                        block,
                    ):
                        return block.copy()

        # unreachable

    @njit(cache=True)
    def _try_block1_q9_loop(
        q9_base: np.uint32,
        q9mask: np.ndarray,
        Q: np.ndarray,
        q5: np.uint32,
        q6: np.uint32,
        q7: np.uint32,
        q8: np.uint32,
        q10: np.uint32,
        q11: np.uint32,
        q12: np.uint32,
        q13: np.uint32,
        a0: np.uint32,
        b0: np.uint32,
        c0: np.uint32,
        d0: np.uint32,
        block: np.ndarray,
        hh_step34_bit15_required: int,
        iv1: np.ndarray,
        iv2: np.ndarray,
        delta_block1: bool,
    ) -> bool:
        for k9 in range(q9mask.shape[0]):
            Q[QOFF + 9] = q9_base ^ q9mask[k9]
            # reverse steps for m8,m9,m12
            block[8] = _reverse_step_first_round(Q, 8, 0x698098D8, 7)
            block[9] = _reverse_step_first_round(Q, 9, 0x8B44F7AF, 12)
            block[12] = _reverse_step_first_round(Q, 12, 0x6B901122, 7)

            ok, a, b, c, d = _tail_rounds_common(a0, b0, c0, d0, block, hh_step34_bit15_required)
            if not ok:
                continue

            out1 = md5_compress_u32(iv1, block)
            block2 = block.copy()
            if delta_block1:
                _apply_delta_block1_inplace(block2)
            else:
                _apply_delta_block0_inplace(block2)
            out2 = md5_compress_u32(iv2, block2)
            if out1[0] == out2[0] and out1[1] == out2[1] and out1[2] == out2[2] and out1[3] == out2[3]:
                return True
        return False

    @njit(cache=True)
    def find_block1_wang_numba(rng_state: np.ndarray, IV: np.ndarray) -> np.ndarray:
        Q = np.zeros(68, dtype=np.uint32)
        Q[0] = IV[0]
        Q[1] = IV[3]
        Q[2] = IV[2]
        Q[3] = IV[1]
        block = np.zeros(16, dtype=np.uint32)

        while True:
            aa = Q[QOFF] & np.uint32(0x80000000)
            bb = np.uint32(0x80000000) ^ aa

            Q[QOFF + 2] = (_rng_next_u32(rng_state) & np.uint32(0x71DE7799)) | np.uint32(0x0C008840) | bb
            Q[QOFF + 3] = (_rng_next_u32(rng_state) & np.uint32(0x01C06601)) | np.uint32(0x3E1F0966) | (Q[QOFF + 2] & np.uint32(0x80000018))
            Q[QOFF + 4] = np.uint32(0x3A040010) | (Q[QOFF + 3] & np.uint32(0x80000601))
            Q[QOFF + 5] = (_rng_next_u32(rng_state) & np.uint32(0x03C0E000)) | np.uint32(0x482F0E50) | aa
            Q[QOFF + 6] = (_rng_next_u32(rng_state) & np.uint32(0x600C0000)) | np.uint32(0x05E2EC56) | aa
            Q[QOFF + 7] = (
                (_rng_next_u32(rng_state) & np.uint32(0x604C203E))
                | np.uint32(0x16819E01)
                | bb
                | (Q[QOFF + 6] & np.uint32(0x01000000))
            )
            Q[QOFF + 8] = (_rng_next_u32(rng_state) & np.uint32(0x604C7C1C)) | np.uint32(0x043283E0) | (Q[QOFF + 7] & np.uint32(0x80000002))
            Q[QOFF + 9] = (_rng_next_u32(rng_state) & np.uint32(0x00002800)) | np.uint32(0x1C0101C1) | (Q[QOFF + 8] & np.uint32(0x80001000))
            Q[QOFF + 10] = np.uint32(0x078BCBC0) | bb
            Q[QOFF + 11] = (_rng_next_u32(rng_state) & np.uint32(0x07800000)) | np.uint32(0x607DC7DF) | bb
            Q[QOFF + 12] = (_rng_next_u32(rng_state) & np.uint32(0x00F00F7F)) | np.uint32(0x00081080) | (Q[QOFF + 11] & np.uint32(0xE7000000))
            Q[QOFF + 13] = (_rng_next_u32(rng_state) & np.uint32(0x00701F77)) | np.uint32(0x3F0FE008) | aa
            Q[QOFF + 14] = (_rng_next_u32(rng_state) & np.uint32(0x00701F77)) | np.uint32(0x408BE088) | aa
            Q[QOFF + 15] = (_rng_next_u32(rng_state) & np.uint32(0x00FF3FF7)) | np.uint32(0x7D000000)
            Q[QOFF + 16] = (
                (_rng_next_u32(rng_state) & np.uint32(0x4FFDFFFF))
                | np.uint32(0x20000000)
                | (np.uint32(~Q[QOFF + 15]) & np.uint32(0x00020000))
            )

            block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
            block[6] = _reverse_step_first_round(Q, 6, 0xA8304613, 17)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)
            block[11] = _reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
            block[14] = _reverse_step_first_round(Q, 14, 0xA679438E, 17)
            block[15] = _reverse_step_first_round(Q, 15, 0x49B40821, 22)

            tt17 = _gg(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + np.uint32(0xF61E2562)
            tt18 = Q[QOFF + 14] + np.uint32(0xC040B340) + block[6]
            tt19 = Q[QOFF + 15] + np.uint32(0x265E5A51) + block[11]

            tt0 = _ff(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + np.uint32(0xD76AA478)
            tt1 = Q[QOFF - 2] + np.uint32(0xE8C7B756)
            q1a = np.uint32(0x04200040) | (Q[QOFF + 2] & np.uint32(0xF01E1080))

            counter = 0
            while counter < (1 << 12):
                counter += 1
                q1 = q1a | (_rng_next_u32(rng_state) & np.uint32(0x01C0E71F))
                m1 = _ror_u32(Q[QOFF + 2] - q1, 12) - _ff(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1

                q16 = Q[QOFF + 16]
                q17 = _rol_u32(tt17 + m1, 5) + q16
                if np.uint32(0x40000000) != ((q17 ^ q16) & np.uint32(0xC0008008)):
                    continue
                if q17 & np.uint32(0x00020000):
                    continue

                q18 = _rol_u32(_gg(q17, q16, Q[QOFF + 15]) + tt18, 9) + q17
                if np.uint32(0x00020000) != ((q18 ^ q17) & np.uint32(0xA0020000)):
                    continue

                q19 = _rol_u32(_gg(q18, q17, q16) + tt19, 14) + q18
                if q19 & np.uint32(0x80020000):
                    continue

                m0 = _ror_u32(q1 - Q[QOFF + 0], 7) - tt0
                q20 = _rol_u32(_gg(q19, q18, q17) + q16 + np.uint32(0xE9B6C7AA) + m0, 20) + q19
                if np.uint32(0x00040000) != ((q20 ^ q19) & np.uint32(0x80040000)):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1
                block[2] = _reverse_step_first_round(Q, 2, 0x242070DB, 17)
                counter = 0
                break

            if counter != 0:
                continue

            q4b = Q[QOFF + 4]
            q9b = Q[QOFF + 9]
            q10b = Q[QOFF + 10]
            tt21 = _gg(Q[QOFF + 20], Q[QOFF + 19], Q[QOFF + 18]) + Q[QOFF + 17] + np.uint32(0xD62F105D)

            counter = 0
            while counter < (1 << 6):
                Q[QOFF + 4] = q4b ^ _W_Q4MASK[counter]
                counter += 1

                block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = _rol_u32(tt21 + block[5], 5) + Q[QOFF + 20]
                if (q21 ^ Q[QOFF + 20]) & np.uint32(0x80020000):
                    continue

                Q[QOFF + 21] = q21
                block[3] = _reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
                block[4] = _reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
                block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)

                tt10 = Q[QOFF + 7] + np.uint32(0xFFFF5BB1)
                tt22 = _gg(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + np.uint32(0x02441453)
                tt23 = Q[QOFF + 19] + np.uint32(0xD8A1E681) + block[15]
                tt24 = Q[QOFF + 20] + np.uint32(0xE7D3FBC8) + block[4]

                counter2 = 0
                while counter2 < (1 << 5):
                    q10 = q10b ^ _W_Q10MASK[counter2]
                    q9 = q9b ^ _W_Q9MASK[counter2]
                    counter2 += 1

                    m10 = _ror_u32(Q[QOFF + 11] - q10, 17)
                    m10 = m10 - _ff(q10, q9, Q[QOFF + 8]) - tt10

                    aa = Q[QOFF + 21]
                    dd = _rol_u32(tt22 + m10, 9) + aa
                    if dd & np.uint32(0x80000000):
                        continue

                    bb = Q[QOFF + 20]
                    cc = tt23 + _gg(dd, aa, bb)
                    if cc & np.uint32(0x20000):
                        continue
                    cc = _rol_u32(cc, 14) + dd
                    if cc & np.uint32(0x80000000):
                        continue

                    bb = _rol_u32(tt24 + _gg(cc, dd, aa), 20) + cc
                    if (bb & np.uint32(0x80000000)) == np.uint32(0):
                        continue

                    block[10] = m10
                    Q[QOFF + 9] = q9
                    Q[QOFF + 10] = q10
                    block[13] = _reverse_step_first_round(Q, 13, 0xFD987193, 12)

                    iv1 = IV.copy()
                    iv2 = np.empty(4, dtype=np.uint32)
                    iv2[0] = IV[0] + np.uint32(1 << 31)
                    iv2[1] = IV[1] + np.uint32((1 << 31) + (1 << 25))
                    iv2[2] = IV[2] + np.uint32((1 << 31) + (1 << 25))
                    iv2[3] = IV[3] + np.uint32((1 << 31) + (1 << 25))

                    if _try_block1_q9_loop(
                        q9,
                        _W_Q9MASK2,
                        Q,
                        Q[QOFF + 5],
                        Q[QOFF + 6],
                        Q[QOFF + 7],
                        Q[QOFF + 8],
                        q10,
                        Q[QOFF + 11],
                        Q[QOFF + 12],
                        Q[QOFF + 13],
                        Q[QOFF + 21],
                        bb,
                        cc,
                        dd,
                        block,
                        1,
                        iv1,
                        iv2,
                        True,
                    ):
                        return block.copy()

        # unreachable

    @njit(cache=True)
    def find_block1_stevens_00_numba(rng_state: np.ndarray, IV: np.ndarray) -> np.ndarray:
        Q = np.zeros(68, dtype=np.uint32)
        Q[0] = IV[0]
        Q[1] = IV[3]
        Q[2] = IV[2]
        Q[3] = IV[1]
        block = np.zeros(16, dtype=np.uint32)

        while True:
            aa = Q[QOFF] & np.uint32(0x80000000)

            Q[QOFF + 2] = (_rng_next_u32(rng_state) & np.uint32(0x49A0E73E)) | np.uint32(0x221F00C1) | aa
            Q[QOFF + 3] = (_rng_next_u32(rng_state) & np.uint32(0x0000040C)) | np.uint32(0x3FCE1A71) | (Q[QOFF + 2] & np.uint32(0x8000E000))
            Q[QOFF + 4] = (_rng_next_u32(rng_state) & np.uint32(0x00000004)) | (np.uint32(0xA5F281A2) ^ (Q[QOFF + 3] & np.uint32(0x80000008)))
            Q[QOFF + 5] = (_rng_next_u32(rng_state) & np.uint32(0x00000004)) | np.uint32(0x67FD823B)
            Q[QOFF + 6] = (_rng_next_u32(rng_state) & np.uint32(0x00001044)) | np.uint32(0x15E5829A)
            Q[QOFF + 7] = (_rng_next_u32(rng_state) & np.uint32(0x00200806)) | np.uint32(0x950430B0)
            Q[QOFF + 8] = (
                (_rng_next_u32(rng_state) & np.uint32(0x60050110))
                | np.uint32(0x1BD29CA2)
                | (Q[QOFF + 7] & np.uint32(0x00000004))
            )
            Q[QOFF + 9] = (_rng_next_u32(rng_state) & np.uint32(0x40044000)) | np.uint32(0xB8820004)
            Q[QOFF + 10] = np.uint32(0xF288B209) | (Q[QOFF + 9] & np.uint32(0x00044000))
            Q[QOFF + 11] = (_rng_next_u32(rng_state) & np.uint32(0x12888008)) | np.uint32(0x85712F57)
            Q[QOFF + 12] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1ED98D7F))
                | np.uint32(0xC0023080)
                | (np.uint32(~Q[QOFF + 11]) & np.uint32(0x00200000))
            )
            Q[QOFF + 13] = (_rng_next_u32(rng_state) & np.uint32(0x0EFB1D77)) | np.uint32(0x1000C008)
            Q[QOFF + 14] = (_rng_next_u32(rng_state) & np.uint32(0x0FFF5D77)) | np.uint32(0xA000A288)
            Q[QOFF + 15] = (
                (_rng_next_u32(rng_state) & np.uint32(0x0EFE7FF7))
                | np.uint32(0xE0008000)
                | (np.uint32(~Q[QOFF + 14]) & np.uint32(0x00010000))
            )
            Q[QOFF + 16] = (
                (_rng_next_u32(rng_state) & np.uint32(0x0FFDFFFF))
                | np.uint32(0xF0000000)
                | (np.uint32(~Q[QOFF + 15]) & np.uint32(0x00020000))
            )

            block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
            block[6] = _reverse_step_first_round(Q, 6, 0xA8304613, 17)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)
            block[11] = _reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
            block[14] = _reverse_step_first_round(Q, 14, 0xA679438E, 17)
            block[15] = _reverse_step_first_round(Q, 15, 0x49B40821, 22)

            tt17 = _gg(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + np.uint32(0xF61E2562)
            tt18 = Q[QOFF + 14] + np.uint32(0xC040B340) + block[6]
            tt19 = Q[QOFF + 15] + np.uint32(0x265E5A51) + block[11]

            tt0 = _ff(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + np.uint32(0xD76AA478)
            tt1 = Q[QOFF - 2] + np.uint32(0xE8C7B756)
            q1a = np.uint32(0x02020801) | (Q[QOFF + 0] & np.uint32(0x80000000))

            counter = 0
            while counter < (1 << 12):
                counter += 1
                q1 = q1a | (_rng_next_u32(rng_state) & np.uint32(0x7DFDF7BE))
                m1 = _ror_u32(Q[QOFF + 2] - q1, 12) - _ff(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1

                q16 = Q[QOFF + 16]
                q17 = _rol_u32(tt17 + m1, 5) + q16
                if np.uint32(0x80000000) != ((q17 ^ q16) & np.uint32(0x80008008)):
                    continue
                if q17 & np.uint32(0x00020000):
                    continue

                q18 = _rol_u32(_gg(q17, q16, Q[QOFF + 15]) + tt18, 9) + q17
                if np.uint32(0x80020000) != ((q18 ^ q17) & np.uint32(0xA0020000)):
                    continue

                q19 = _rol_u32(_gg(q18, q17, q16) + tt19, 14) + q18
                if np.uint32(0x80000000) != (q19 & np.uint32(0x80020000)):
                    continue

                m0 = _ror_u32(q1 - Q[QOFF + 0], 7) - tt0
                q20 = _rol_u32(_gg(q19, q18, q17) + q16 + np.uint32(0xE9B6C7AA) + m0, 20) + q19
                if np.uint32(0x00040000) != ((q20 ^ q19) & np.uint32(0x80040000)):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1

                block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = _rol_u32(_gg(q20, q19, q18) + q17 + np.uint32(0xD62F105D) + block[5], 5) + q20
                if (q21 ^ q20) & np.uint32(0x80020000):
                    continue
                Q[QOFF + 21] = q21
                counter = 0
                break

            if counter != 0:
                continue

            q9b = Q[QOFF + 9]
            q10b = Q[QOFF + 10]

            block[2] = _reverse_step_first_round(Q, 2, 0x242070DB, 17)
            block[3] = _reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
            block[4] = _reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)

            tt10 = Q[QOFF + 7] + np.uint32(0xFFFF5BB1)
            tt22 = _gg(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + np.uint32(0x02441453)
            tt23 = Q[QOFF + 19] + np.uint32(0xD8A1E681) + block[15]
            tt24 = Q[QOFF + 20] + np.uint32(0xE7D3FBC8) + block[4]

            for k10 in range(_S00_Q9Q10MASK.shape[0]):
                mask = _S00_Q9Q10MASK[k10]
                q10 = q10b | (mask & np.uint32(0x08000020))
                q9 = q9b | (mask & np.uint32(0x00002000))

                m10 = _ror_u32(Q[QOFF + 11] - q10, 17)
                m10 = m10 - _ff(q10, q9, Q[QOFF + 8]) - tt10

                aa = Q[QOFF + 21]
                dd = _rol_u32(tt22 + m10, 9) + aa
                if (dd & np.uint32(0x80000000)) == np.uint32(0):
                    continue

                bb = Q[QOFF + 20]
                cc = tt23 + _gg(dd, aa, bb)
                if cc & np.uint32(0x20000):
                    continue
                cc = _rol_u32(cc, 14) + dd
                if cc & np.uint32(0x80000000):
                    continue

                bb = _rol_u32(tt24 + _gg(cc, dd, aa), 20) + cc
                if (bb & np.uint32(0x80000000)) == np.uint32(0):
                    continue

                block[10] = m10
                Q[QOFF + 9] = q9
                Q[QOFF + 10] = q10
                block[13] = _reverse_step_first_round(Q, 13, 0xFD987193, 12)

                iv1 = IV.copy()
                iv2 = np.empty(4, dtype=np.uint32)
                iv2[0] = IV[0] + np.uint32(1 << 31)
                iv2[1] = IV[1] + np.uint32(1 << 31) - np.uint32(1 << 25)
                iv2[2] = IV[2] + np.uint32(1 << 31) - np.uint32(1 << 25)
                iv2[3] = IV[3] + np.uint32(1 << 31) - np.uint32(1 << 25)

                if _try_block1_q9_loop(
                    q9,
                    _S00_Q9MASK,
                    Q,
                    Q[QOFF + 5],
                    Q[QOFF + 6],
                    Q[QOFF + 7],
                    Q[QOFF + 8],
                    q10,
                    Q[QOFF + 11],
                    Q[QOFF + 12],
                    Q[QOFF + 13],
                    aa,
                    bb,
                    cc,
                    dd,
                    block,
                    0,
                    iv1,
                    iv2,
                    False,
                ):
                    return block.copy()

        # unreachable

    # Stevens 01/10/11 are implemented by directly porting the Python code;
    # keep them separate to minimize branching in the hot loops.

    @njit(cache=True)
    def find_block1_stevens_01_numba(rng_state: np.ndarray, IV: np.ndarray) -> np.ndarray:
        Q = np.zeros(68, dtype=np.uint32)
        Q[0] = IV[0]
        Q[1] = IV[3]
        Q[2] = IV[2]
        Q[3] = IV[1]
        block = np.zeros(16, dtype=np.uint32)

        while True:
            aa_bit = Q[QOFF] & np.uint32(0x80000000)

            Q[QOFF + 2] = (_rng_next_u32(rng_state) & np.uint32(0x4DB0E03E)) | np.uint32(0x32460441) | aa_bit
            Q[QOFF + 3] = (
                (_rng_next_u32(rng_state) & np.uint32(0x0C000008))
                | np.uint32(0x123C3AF1)
                | (Q[QOFF + 2] & np.uint32(0x80800002))
            )
            Q[QOFF + 4] = np.uint32(0xE398F812) ^ (Q[QOFF + 3] & np.uint32(0x88000000))
            Q[QOFF + 5] = (_rng_next_u32(rng_state) & np.uint32(0x82000000)) | np.uint32(0x4C66E99E)
            Q[QOFF + 6] = (_rng_next_u32(rng_state) & np.uint32(0x80000000)) | np.uint32(0x27180590)
            Q[QOFF + 7] = (_rng_next_u32(rng_state) & np.uint32(0x00010130)) | np.uint32(0x51EA9E47)
            Q[QOFF + 8] = (_rng_next_u32(rng_state) & np.uint32(0x40200800)) | np.uint32(0xB7C291E5)
            Q[QOFF + 9] = (_rng_next_u32(rng_state) & np.uint32(0x00044000)) | np.uint32(0x380002B4)
            Q[QOFF + 10] = np.uint32(0xB282B208) | (Q[QOFF + 9] & np.uint32(0x00044000))
            Q[QOFF + 11] = (_rng_next_u32(rng_state) & np.uint32(0x12808008)) | np.uint32(0xC5712F47)
            Q[QOFF + 12] = (_rng_next_u32(rng_state) & np.uint32(0x1EF18D7F)) | np.uint32(0x000A3080)
            Q[QOFF + 13] = (_rng_next_u32(rng_state) & np.uint32(0x1EFB1D77)) | np.uint32(0x4004C008)
            Q[QOFF + 14] = (_rng_next_u32(rng_state) & np.uint32(0x1FFF5D77)) | np.uint32(0x6000A288)
            Q[QOFF + 15] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1EFE7FF7))
                | np.uint32(0xA0008000)
                | (np.uint32(~Q[QOFF + 14]) & np.uint32(0x00010000))
            )
            Q[QOFF + 16] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1FFDFFFF))
                | np.uint32(0x20000000)
                | (np.uint32(~Q[QOFF + 15]) & np.uint32(0x00020000))
            )

            block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
            block[6] = _reverse_step_first_round(Q, 6, 0xA8304613, 17)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)
            block[11] = _reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
            block[14] = _reverse_step_first_round(Q, 14, 0xA679438E, 17)
            block[15] = _reverse_step_first_round(Q, 15, 0x49B40821, 22)

            tt17 = _gg(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + np.uint32(0xF61E2562)
            tt18 = Q[QOFF + 14] + np.uint32(0xC040B340) + block[6]
            tt19 = Q[QOFF + 15] + np.uint32(0x265E5A51) + block[11]

            tt0 = _ff(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + np.uint32(0xD76AA478)
            tt1 = Q[QOFF - 2] + np.uint32(0xE8C7B756)
            q1a = np.uint32(0x02000021) ^ (Q[QOFF + 0] & np.uint32(0x80000020))

            counter = 0
            while counter < (1 << 12):
                counter += 1
                q1 = q1a | (_rng_next_u32(rng_state) & np.uint32(0x7DFFF39E))
                m1 = _ror_u32(Q[QOFF + 2] - q1, 12) - _ff(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1

                q16 = Q[QOFF + 16]
                q17 = _rol_u32(tt17 + m1, 5) + q16
                if np.uint32(0x80000000) != ((q17 ^ q16) & np.uint32(0x80008008)):
                    continue
                if q17 & np.uint32(0x00020000):
                    continue

                q18 = _rol_u32(_gg(q17, q16, Q[QOFF + 15]) + tt18, 9) + q17
                if np.uint32(0x80020000) != ((q18 ^ q17) & np.uint32(0xA0020000)):
                    continue

                q19 = _rol_u32(_gg(q18, q17, q16) + tt19, 14) + q18
                if np.uint32(0x00000000) != (q19 & np.uint32(0x80020000)):
                    continue

                m0 = _ror_u32(q1 - Q[QOFF + 0], 7) - tt0
                q20 = _rol_u32(_gg(q19, q18, q17) + q16 + np.uint32(0xE9B6C7AA) + m0, 20) + q19
                if np.uint32(0x00040000) != ((q20 ^ q19) & np.uint32(0x80040000)):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1

                block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = _rol_u32(_gg(q20, q19, q18) + q17 + np.uint32(0xD62F105D) + block[5], 5) + q20
                if (q21 ^ q20) & np.uint32(0x80020000):
                    continue
                Q[QOFF + 21] = q21
                counter = 0
                break

            if counter != 0:
                continue

            q9b = Q[QOFF + 9]
            q10b = Q[QOFF + 10]

            block[2] = _reverse_step_first_round(Q, 2, 0x242070DB, 17)
            block[3] = _reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
            block[4] = _reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)

            tt10 = Q[QOFF + 7] + np.uint32(0xFFFF5BB1)
            tt22 = _gg(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + np.uint32(0x02441453)
            tt23 = Q[QOFF + 19] + np.uint32(0xD8A1E681) + block[15]
            tt24 = Q[QOFF + 20] + np.uint32(0xE7D3FBC8) + block[4]

            for k10 in range(_S01_Q9Q10MASK.shape[0]):
                mask = _S01_Q9Q10MASK[k10]
                q10 = q10b | (mask & np.uint32(0x08000030))
                q9 = q9b | (mask & np.uint32(0x80002000))

                m10 = _ror_u32(Q[QOFF + 11] - q10, 17)
                m10 = m10 - _ff(q10, q9, Q[QOFF + 8]) - tt10

                aa = Q[QOFF + 21]
                dd = _rol_u32(tt22 + m10, 9) + aa
                if dd & np.uint32(0x80000000):
                    continue

                bb = Q[QOFF + 20]
                cc = tt23 + _gg(dd, aa, bb)
                if cc & np.uint32(0x20000):
                    continue
                cc = _rol_u32(cc, 14) + dd
                if cc & np.uint32(0x80000000):
                    continue

                bb = _rol_u32(tt24 + _gg(cc, dd, aa), 20) + cc
                if (bb & np.uint32(0x80000000)) == np.uint32(0):
                    continue

                block[10] = m10
                Q[QOFF + 9] = q9
                Q[QOFF + 10] = q10
                block[13] = _reverse_step_first_round(Q, 13, 0xFD987193, 12)

                iv1 = IV.copy()
                iv2 = np.empty(4, dtype=np.uint32)
                iv2[0] = IV[0] + np.uint32(1 << 31)
                iv2[1] = IV[1] + np.uint32(1 << 31) - np.uint32(1 << 25)
                iv2[2] = IV[2] + np.uint32(1 << 31) - np.uint32(1 << 25)
                iv2[3] = IV[3] + np.uint32(1 << 31) - np.uint32(1 << 25)

                if _try_block1_q9_loop(
                    q9,
                    _S01_Q9MASK,
                    Q,
                    Q[QOFF + 5],
                    Q[QOFF + 6],
                    Q[QOFF + 7],
                    Q[QOFF + 8],
                    q10,
                    Q[QOFF + 11],
                    Q[QOFF + 12],
                    Q[QOFF + 13],
                    aa,
                    bb,
                    cc,
                    dd,
                    block,
                    0,
                    iv1,
                    iv2,
                    False,
                ):
                    return block.copy()

        # unreachable

    @njit(cache=True)
    def find_block1_stevens_10_numba(rng_state: np.ndarray, IV: np.ndarray) -> np.ndarray:
        Q = np.zeros(68, dtype=np.uint32)
        Q[0] = IV[0]
        Q[1] = IV[3]
        Q[2] = IV[2]
        Q[3] = IV[1]
        block = np.zeros(16, dtype=np.uint32)

        while True:
            aa_bit = Q[QOFF] & np.uint32(0x80000000)

            Q[QOFF + 2] = (_rng_next_u32(rng_state) & np.uint32(0x79B0C6BA)) | np.uint32(0x024C3841) | aa_bit
            Q[QOFF + 3] = (
                (_rng_next_u32(rng_state) & np.uint32(0x19300210))
                | np.uint32(0x2603096D)
                | (Q[QOFF + 2] & np.uint32(0x80000082))
            )
            Q[QOFF + 4] = (
                (_rng_next_u32(rng_state) & np.uint32(0x10300000))
                | np.uint32(0xE4CAE30C)
                | (Q[QOFF + 3] & np.uint32(0x01000030))
            )
            Q[QOFF + 5] = (
                (_rng_next_u32(rng_state) & np.uint32(0x10000000))
                | np.uint32(0x63494061)
                | (Q[QOFF + 4] & np.uint32(0x00300000))
            )
            Q[QOFF + 6] = np.uint32(0x7DEAFF68)
            Q[QOFF + 7] = (_rng_next_u32(rng_state) & np.uint32(0x20444000)) | np.uint32(0x09091EE0)
            Q[QOFF + 8] = (_rng_next_u32(rng_state) & np.uint32(0x09040000)) | np.uint32(0xB2529F6D)
            Q[QOFF + 9] = (_rng_next_u32(rng_state) & np.uint32(0x00040000)) | np.uint32(0x10885184)
            Q[QOFF + 10] = (
                (_rng_next_u32(rng_state) & np.uint32(0x00000080))
                | np.uint32(0x428AFB11)
                | (Q[QOFF + 9] & np.uint32(0x00040000))
            )
            Q[QOFF + 11] = (
                (_rng_next_u32(rng_state) & np.uint32(0x128A8110))
                | np.uint32(0x6571266B)
                | (Q[QOFF + 10] & np.uint32(0x00000080))
            )
            Q[QOFF + 12] = (
                (_rng_next_u32(rng_state) & np.uint32(0x3EF38D7F))
                | np.uint32(0x00003080)
                | (np.uint32(~Q[QOFF + 11]) & np.uint32(0x00080000))
            )
            Q[QOFF + 13] = (_rng_next_u32(rng_state) & np.uint32(0x3EFB1D77)) | np.uint32(0x0004C008)
            Q[QOFF + 14] = (_rng_next_u32(rng_state) & np.uint32(0x5FFF5D77)) | np.uint32(0x8000A288)
            Q[QOFF + 15] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1EFE7FF7))
                | np.uint32(0xE0008000)
                | (np.uint32(~Q[QOFF + 14]) & np.uint32(0x00010000))
            )
            Q[QOFF + 16] = (
                (_rng_next_u32(rng_state) & np.uint32(0x5FFDFFFF))
                | np.uint32(0x20000000)
                | (np.uint32(~Q[QOFF + 15]) & np.uint32(0x00020000))
            )

            block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
            block[6] = _reverse_step_first_round(Q, 6, 0xA8304613, 17)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)
            block[11] = _reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
            block[14] = _reverse_step_first_round(Q, 14, 0xA679438E, 17)
            block[15] = _reverse_step_first_round(Q, 15, 0x49B40821, 22)

            tt17 = _gg(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + np.uint32(0xF61E2562)
            tt18 = Q[QOFF + 14] + np.uint32(0xC040B340) + block[6]
            tt19 = Q[QOFF + 15] + np.uint32(0x265E5A51) + block[11]

            tt0 = _ff(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + np.uint32(0xD76AA478)
            tt1 = Q[QOFF - 2] + np.uint32(0xE8C7B756)
            q1a = np.uint32(0x02000941) ^ (Q[QOFF + 0] & np.uint32(0x80000000))

            counter = 0
            while counter < (1 << 12):
                counter += 1
                q1 = q1a | (_rng_next_u32(rng_state) & np.uint32(0x7DFDF6BE))
                m1 = _ror_u32(Q[QOFF + 2] - q1, 12) - _ff(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1

                q16 = Q[QOFF + 16]
                q17 = _rol_u32(tt17 + m1, 5) + q16
                if np.uint32(0x80000000) != ((q17 ^ q16) & np.uint32(0x80008008)):
                    continue
                if q17 & np.uint32(0x00020000):
                    continue

                q18 = _rol_u32(_gg(q17, q16, Q[QOFF + 15]) + tt18, 9) + q17
                if np.uint32(0x80020000) != ((q18 ^ q17) & np.uint32(0xA0020000)):
                    continue

                q19 = _rol_u32(_gg(q18, q17, q16) + tt19, 14) + q18
                if np.uint32(0x00000000) != (q19 & np.uint32(0x80020000)):
                    continue

                m0 = _ror_u32(q1 - Q[QOFF + 0], 7) - tt0
                q20 = _rol_u32(_gg(q19, q18, q17) + q16 + np.uint32(0xE9B6C7AA) + m0, 20) + q19
                if np.uint32(0x00040000) != ((q20 ^ q19) & np.uint32(0x80040000)):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1

                block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = _rol_u32(_gg(q20, q19, q18) + q17 + np.uint32(0xD62F105D) + block[5], 5) + q20
                if (q21 ^ q20) & np.uint32(0x80020000):
                    continue
                Q[QOFF + 21] = q21
                counter = 0
                break

            if counter != 0:
                continue

            q9b = Q[QOFF + 9]
            q10b = Q[QOFF + 10]

            block[2] = _reverse_step_first_round(Q, 2, 0x242070DB, 17)
            block[3] = _reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
            block[4] = _reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)

            tt10 = Q[QOFF + 7] + np.uint32(0xFFFF5BB1)
            tt22 = _gg(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + np.uint32(0x02441453)
            tt23 = Q[QOFF + 19] + np.uint32(0xD8A1E681) + block[15]
            tt24 = Q[QOFF + 20] + np.uint32(0xE7D3FBC8) + block[4]

            for k10 in range(_S10_Q9Q10MASK.shape[0]):
                mask = _S10_Q9Q10MASK[k10]
                q10 = q10b | (mask & np.uint32(0x08000004))
                q9 = q9b | (mask & np.uint32(0x00004200))

                m10 = _ror_u32(Q[QOFF + 11] - q10, 17)
                m10 = m10 - _ff(q10, q9, Q[QOFF + 8]) - tt10

                aa = Q[QOFF + 21]
                dd = _rol_u32(tt22 + m10, 9) + aa
                if dd & np.uint32(0x80000000):
                    continue

                bb = Q[QOFF + 20]
                cc = tt23 + _gg(dd, aa, bb)
                if cc & np.uint32(0x20000):
                    continue
                cc = _rol_u32(cc, 14) + dd
                if cc & np.uint32(0x80000000):
                    continue

                bb = _rol_u32(tt24 + _gg(cc, dd, aa), 20) + cc
                if (bb & np.uint32(0x80000000)) == np.uint32(0):
                    continue

                block[10] = m10
                Q[QOFF + 9] = q9
                Q[QOFF + 10] = q10
                block[13] = _reverse_step_first_round(Q, 13, 0xFD987193, 12)

                iv1 = IV.copy()
                iv2 = np.empty(4, dtype=np.uint32)
                iv2[0] = IV[0] + np.uint32(1 << 31)
                iv2[1] = IV[1] + np.uint32(1 << 31) - np.uint32(1 << 25)
                iv2[2] = IV[2] + np.uint32(1 << 31) - np.uint32(1 << 25)
                iv2[3] = IV[3] + np.uint32(1 << 31) - np.uint32(1 << 25)

                if _try_block1_q9_loop(
                    q9,
                    _S10_Q9MASK,
                    Q,
                    Q[QOFF + 5],
                    Q[QOFF + 6],
                    Q[QOFF + 7],
                    Q[QOFF + 8],
                    q10,
                    Q[QOFF + 11],
                    Q[QOFF + 12],
                    Q[QOFF + 13],
                    aa,
                    bb,
                    cc,
                    dd,
                    block,
                    0,
                    iv1,
                    iv2,
                    False,
                ):
                    return block.copy()

        # unreachable

    @njit(cache=True)
    def find_block1_stevens_11_numba(rng_state: np.ndarray, IV: np.ndarray) -> np.ndarray:
        Q = np.zeros(68, dtype=np.uint32)
        Q[0] = IV[0]
        Q[1] = IV[3]
        Q[2] = IV[2]
        Q[3] = IV[1]
        block = np.zeros(16, dtype=np.uint32)

        while True:
            aa = Q[QOFF] & np.uint32(0x80000000)

            Q[QOFF + 2] = (_rng_next_u32(rng_state) & np.uint32(0x75BEF63E)) | np.uint32(0x0A410041) | aa
            Q[QOFF + 3] = (_rng_next_u32(rng_state) & np.uint32(0x10345614)) | np.uint32(0x0202A9E1) | (Q[QOFF + 2] & np.uint32(0x84000002))
            Q[QOFF + 4] = (_rng_next_u32(rng_state) & np.uint32(0x00145400)) | np.uint32(0xE84BA909) | (Q[QOFF + 3] & np.uint32(0x00000014))
            Q[QOFF + 5] = (_rng_next_u32(rng_state) & np.uint32(0x80000000)) | np.uint32(0x75E90B1D) | (Q[QOFF + 4] & np.uint32(0x00145400))
            Q[QOFF + 6] = np.uint32(0x7C23FF5A) | (Q[QOFF + 5] & np.uint32(0x80000000))
            Q[QOFF + 7] = (_rng_next_u32(rng_state) & np.uint32(0x40000880)) | np.uint32(0x114BF41A)
            Q[QOFF + 8] = (_rng_next_u32(rng_state) & np.uint32(0x00002090)) | np.uint32(0xB352DD01)
            Q[QOFF + 9] = (_rng_next_u32(rng_state) & np.uint32(0x00044000)) | np.uint32(0x7A803124)
            Q[QOFF + 10] = (_rng_next_u32(rng_state) & np.uint32(0x00002000)) | np.uint32(0xF28A92C9) | (Q[QOFF + 9] & np.uint32(0x00044000))
            Q[QOFF + 11] = (_rng_next_u32(rng_state) & np.uint32(0x128A8108)) | np.uint32(0xC5710ED7) | (Q[QOFF + 10] & np.uint32(0x00002000))
            Q[QOFF + 12] = (
                (_rng_next_u32(rng_state) & np.uint32(0x9EDB8D7F))
                | np.uint32(0x20003080)
                | (np.uint32(~Q[QOFF + 11]) & np.uint32(0x00200000))
            )
            Q[QOFF + 13] = (_rng_next_u32(rng_state) & np.uint32(0x3EFB1D77)) | np.uint32(0x4004C008) | (Q[QOFF + 12] & np.uint32(0x80000000))
            Q[QOFF + 14] = (_rng_next_u32(rng_state) & np.uint32(0x1FFF5D77)) | np.uint32(0x0000A288)
            Q[QOFF + 15] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1EFE7FF7))
                | np.uint32(0x20008000)
                | (np.uint32(~Q[QOFF + 14]) & np.uint32(0x00010000))
            )
            Q[QOFF + 16] = (
                (_rng_next_u32(rng_state) & np.uint32(0x1FFDFFFF))
                | np.uint32(0x20000000)
                | (np.uint32(~Q[QOFF + 15]) & np.uint32(0x40020000))
            )

            block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
            block[6] = _reverse_step_first_round(Q, 6, 0xA8304613, 17)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)
            block[11] = _reverse_step_first_round(Q, 11, 0x895CD7BE, 22)
            block[14] = _reverse_step_first_round(Q, 14, 0xA679438E, 17)
            block[15] = _reverse_step_first_round(Q, 15, 0x49B40821, 22)

            tt17 = _gg(Q[QOFF + 16], Q[QOFF + 15], Q[QOFF + 14]) + Q[QOFF + 13] + np.uint32(0xF61E2562)
            tt18 = Q[QOFF + 14] + np.uint32(0xC040B340) + block[6]
            tt19 = Q[QOFF + 15] + np.uint32(0x265E5A51) + block[11]

            tt0 = _ff(Q[QOFF + 0], Q[QOFF - 1], Q[QOFF - 2]) + Q[QOFF - 3] + np.uint32(0xD76AA478)
            tt1 = Q[QOFF - 2] + np.uint32(0xE8C7B756)
            q1a = np.uint32(0x02000861) ^ (Q[QOFF + 0] & np.uint32(0x80000020))

            counter = 0
            while counter < (1 << 12):
                counter += 1
                q1 = q1a | (_rng_next_u32(rng_state) & np.uint32(0x7DFFF79E))
                m1 = _ror_u32(Q[QOFF + 2] - q1, 12) - _ff(q1, Q[QOFF + 0], Q[QOFF - 1]) - tt1

                q16 = Q[QOFF + 16]
                q17 = _rol_u32(tt17 + m1, 5) + q16
                if np.uint32(0x40000000) != ((q17 ^ q16) & np.uint32(0xC0008008)):
                    continue
                if q17 & np.uint32(0x00020000):
                    continue

                q18 = _rol_u32(_gg(q17, q16, Q[QOFF + 15]) + tt18, 9) + q17
                if np.uint32(0x80020000) != ((q18 ^ q17) & np.uint32(0xA0020000)):
                    continue

                q19 = _rol_u32(_gg(q18, q17, q16) + tt19, 14) + q18
                if np.uint32(0x80000000) != (q19 & np.uint32(0x80020000)):
                    continue

                m0 = _ror_u32(q1 - Q[QOFF + 0], 7) - tt0
                q20 = _rol_u32(_gg(q19, q18, q17) + q16 + np.uint32(0xE9B6C7AA) + m0, 20) + q19
                if np.uint32(0x00040000) != ((q20 ^ q19) & np.uint32(0x80040000)):
                    continue

                Q[QOFF + 1] = q1
                Q[QOFF + 17] = q17
                Q[QOFF + 18] = q18
                Q[QOFF + 19] = q19
                Q[QOFF + 20] = q20
                block[0] = m0
                block[1] = m1

                block[5] = _reverse_step_first_round(Q, 5, 0x4787C62A, 12)
                q21 = _rol_u32(_gg(q20, q19, q18) + q17 + np.uint32(0xD62F105D) + block[5], 5) + q20
                if (q21 ^ q20) & np.uint32(0x80020000):
                    continue
                Q[QOFF + 21] = q21
                counter = 0
                break

            if counter != 0:
                continue

            q9b = Q[QOFF + 9]
            q10b = Q[QOFF + 10]

            block[2] = _reverse_step_first_round(Q, 2, 0x242070DB, 17)
            block[3] = _reverse_step_first_round(Q, 3, 0xC1BDCEEE, 22)
            block[4] = _reverse_step_first_round(Q, 4, 0xF57C0FAF, 7)
            block[7] = _reverse_step_first_round(Q, 7, 0xFD469501, 22)

            tt10 = Q[QOFF + 7] + np.uint32(0xFFFF5BB1)
            tt22 = _gg(Q[QOFF + 21], Q[QOFF + 20], Q[QOFF + 19]) + Q[QOFF + 18] + np.uint32(0x02441453)
            tt23 = Q[QOFF + 19] + np.uint32(0xD8A1E681) + block[15]
            tt24 = Q[QOFF + 20] + np.uint32(0xE7D3FBC8) + block[4]

            for k10 in range(_S11_Q9Q10MASK.shape[0]):
                mask = _S11_Q9Q10MASK[k10]
                q10 = q10b | (mask & np.uint32(0x08000040))
                q9 = q9b | (mask & np.uint32(0x80000280))

                m10 = _ror_u32(Q[QOFF + 11] - q10, 17)
                m10 = m10 - _ff(q10, q9, Q[QOFF + 8]) - tt10

                aa = Q[QOFF + 21]
                dd = _rol_u32(tt22 + m10, 9) + aa
                if (dd & np.uint32(0x80000000)) == np.uint32(0):
                    continue

                bb = Q[QOFF + 20]
                cc = tt23 + _gg(dd, aa, bb)
                if cc & np.uint32(0x20000):
                    continue
                cc = _rol_u32(cc, 14) + dd
                if cc & np.uint32(0x80000000):
                    continue

                bb = _rol_u32(tt24 + _gg(cc, dd, aa), 20) + cc
                if (bb & np.uint32(0x80000000)) == np.uint32(0):
                    continue

                block[10] = m10
                Q[QOFF + 9] = q9
                Q[QOFF + 10] = q10
                block[13] = _reverse_step_first_round(Q, 13, 0xFD987193, 12)

                iv1 = IV.copy()
                iv2 = np.empty(4, dtype=np.uint32)
                iv2[0] = IV[0] + np.uint32(1 << 31)
                iv2[1] = IV[1] + np.uint32(1 << 31) - np.uint32(1 << 25)
                iv2[2] = IV[2] + np.uint32(1 << 31) - np.uint32(1 << 25)
                iv2[3] = IV[3] + np.uint32(1 << 31) - np.uint32(1 << 25)

                if _try_block1_q9_loop(
                    q9,
                    _S11_Q9MASK,
                    Q,
                    Q[QOFF + 5],
                    Q[QOFF + 6],
                    Q[QOFF + 7],
                    Q[QOFF + 8],
                    q10,
                    Q[QOFF + 11],
                    Q[QOFF + 12],
                    Q[QOFF + 13],
                    aa,
                    bb,
                    cc,
                    dd,
                    block,
                    0,
                    iv1,
                    iv2,
                    False,
                ):
                    return block.copy()

        # unreachable

    @njit(cache=True)
    def find_block1_numba(rng_state: np.ndarray, IV: np.ndarray) -> np.ndarray:
        if (
            ((IV[1] ^ IV[2]) & np.uint32(1 << 31)) == np.uint32(0)
            and ((IV[1] ^ IV[3]) & np.uint32(1 << 31)) == np.uint32(0)
            and (IV[3] & np.uint32(1 << 25)) == np.uint32(0)
            and (IV[2] & np.uint32(1 << 25)) == np.uint32(0)
            and (IV[1] & np.uint32(1 << 25)) == np.uint32(0)
            and ((IV[2] ^ IV[1]) & np.uint32(1)) == np.uint32(0)
        ):
            IV2 = np.empty(4, dtype=np.uint32)
            IV2[0] = IV[0] + np.uint32(1 << 31)
            IV2[1] = IV[1] + np.uint32((1 << 31) + (1 << 25))
            IV2[2] = IV[2] + np.uint32((1 << 31) + (1 << 25))
            IV2[3] = IV[3] + np.uint32((1 << 31) + (1 << 25))

            if (IV[1] & np.uint32(1 << 6)) != np.uint32(0) and (IV[1] & np.uint32(1)) != np.uint32(0):
                msg2_block = find_block1_stevens_11_numba(rng_state, IV2)
            elif (IV[1] & np.uint32(1 << 6)) != np.uint32(0) and (IV[1] & np.uint32(1)) == np.uint32(0):
                msg2_block = find_block1_stevens_10_numba(rng_state, IV2)
            elif (IV[1] & np.uint32(1 << 6)) == np.uint32(0) and (IV[1] & np.uint32(1)) != np.uint32(0):
                msg2_block = find_block1_stevens_01_numba(rng_state, IV2)
            else:
                msg2_block = find_block1_stevens_00_numba(rng_state, IV2)

            msg1_block = msg2_block.copy()
            _apply_delta_block0_inplace(msg1_block)
            return msg1_block

        return find_block1_wang_numba(rng_state, IV)


def find_block1_dispatch(rng_state: np.ndarray, IV: np.ndarray) -> np.ndarray:
    if njit is None:
        raise RuntimeError("numba is not available (pip install numba) or disabled via MD5FASTCOLL_NO_NUMBA=1")

    if (
        ((IV[1] ^ IV[2]) & np.uint32(1 << 31)) == np.uint32(0)
        and ((IV[1] ^ IV[3]) & np.uint32(1 << 31)) == np.uint32(0)
        and (IV[3] & np.uint32(1 << 25)) == np.uint32(0)
        and (IV[2] & np.uint32(1 << 25)) == np.uint32(0)
        and (IV[1] & np.uint32(1 << 25)) == np.uint32(0)
        and ((IV[2] ^ IV[1]) & np.uint32(1)) == np.uint32(0)
    ):
        IV2 = np.empty(4, dtype=np.uint32)
        IV2[0] = IV[0] + np.uint32(1 << 31)
        IV2[1] = IV[1] + np.uint32((1 << 31) + (1 << 25))
        IV2[2] = IV[2] + np.uint32((1 << 31) + (1 << 25))
        IV2[3] = IV[3] + np.uint32((1 << 31) + (1 << 25))

        if (IV[1] & np.uint32(1 << 6)) != np.uint32(0) and (IV[1] & np.uint32(1)) != np.uint32(0):
            msg2_block = find_block1_stevens_11_numba(rng_state, IV2)
        elif (IV[1] & np.uint32(1 << 6)) != np.uint32(0) and (IV[1] & np.uint32(1)) == np.uint32(0):
            msg2_block = find_block1_stevens_10_numba(rng_state, IV2)
        elif (IV[1] & np.uint32(1 << 6)) == np.uint32(0) and (IV[1] & np.uint32(1)) != np.uint32(0):
            msg2_block = find_block1_stevens_01_numba(rng_state, IV2)
        else:
            msg2_block = find_block1_stevens_00_numba(rng_state, IV2)

        msg1_block = msg2_block.copy()
        msg1_block[4] = msg1_block[4] + np.uint32(1 << 31)
        msg1_block[11] = msg1_block[11] + np.uint32(1 << 15)
        msg1_block[14] = msg1_block[14] + np.uint32(1 << 31)
        return msg1_block

    return find_block1_wang_numba(rng_state, IV)


def find_collision_blocks_numba(
    ihv: Tuple[int, int, int, int],
    *,
    seed: int | None = None,
) -> Tuple[List[int], List[int]]:
    if njit is None:
        raise RuntimeError("numba is not available (pip install numba) or disabled via MD5FASTCOLL_NO_NUMBA=1")
    # Hybrid strategy:
    # - Block0: use the optimized Python+NumPy implementation (fastest in practice).
    # - Block1: use Numba JIT (orders of magnitude faster than pure Python here).
    from .py_fastcoll import find_block0, rng_from_seed

    rng = rng_from_seed(seed)
    iv = np.array([ihv[0] & 0xFFFFFFFF, ihv[1] & 0xFFFFFFFF, ihv[2] & 0xFFFFFFFF, ihv[3] & 0xFFFFFFFF], dtype=np.uint32)

    b0_list = find_block0(rng, iv)
    b0 = np.array(b0_list, dtype=np.uint32)
    ihv_after = md5_compress_u32(iv, b0)
    iv_after = np.array([int(ihv_after[0]), int(ihv_after[1]), int(ihv_after[2]), int(ihv_after[3])], dtype=np.uint32)

    rng_state = np.array([rng.seed32_1 & 0xFFFFFFFF, rng.seed32_2 & 0xFFFFFFFF], dtype=np.uint32)
    b1 = find_block1_dispatch(rng_state, iv_after)

    return [int(x) for x in b0_list], [int(x) for x in b1.tolist()]
