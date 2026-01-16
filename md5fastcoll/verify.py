from __future__ import annotations

from typing import Dict, List, Tuple

from .core import u32


def bit(x: int, b: int) -> int:
    return (x >> b) & 1


# Full list of T restrictions from section 3 (3.1 .. 3.9) + (3.10, 3.11)
# We verify them directly on Tt bits as suggested in 3.10 and 3.11 notes.

def check_T_restrictions_full(trace: Dict[str, List[int]]) -> Tuple[bool, Dict[str, int]]:
    T = trace["T"]
    bad: Dict[str, int] = {}

    # 3.1: δT4 = -2^31 -> T4[31] must be 1
    if bit(T[4], 31) != 1:
        bad["T4[31]"] = bit(T[4], 31)

    # 3.2: -2^14 in δT6 must propagate to at least bit 15 -> T6[14] == 0
    if bit(T[6], 14) != 0:
        bad["T6[14]"] = bit(T[6], 14)

    # 3.3: +2^13 in δT10 must not propagate past bit 14 -> T10[13] == 0
    if bit(T[10], 13) != 0:
        bad["T10[13]"] = bit(T[10], 13)

    # 3.4: -2^8 in δT11 must not propagate past bit 9 -> enforce T11[8] == 1
    if bit(T[11], 8) != 1:
        bad["T11[8]"] = bit(T[11], 8)

    # 3.5: -2^30 in δT14 must not propagate past bit 31 -> T14[30] == 1 or T14[31] == 1
    if not (bit(T[14], 30) == 1 or bit(T[14], 31) == 1):
        bad["T14[30|31]"] = (bit(T[14], 31) << 1) | bit(T[14], 30)

    # 3.6: -2^7 in δT15 must not propagate past bit 9 -> any of T15[7],T15[8],T15[9] is 1
    if not (bit(T[15], 7) == 1 or bit(T[15], 8) == 1 or bit(T[15], 9) == 1):
        bad["T15[7|8|9]"] = (bit(T[15], 9) << 2) | (bit(T[15], 8) << 1) | bit(T[15], 7)

    # 3.7: +2^25 in δT15 must not propagate past bit 31 -> any of T15[25],T15[26],T15[27] is 0  
    if not (bit(T[15], 25) == 0 or bit(T[15], 26) == 0 or bit(T[15], 27) == 0):
        bad["T15[25|26|27]"] = (bit(T[15], 27) << 2) | (bit(T[15], 26) << 1) | bit(T[15], 25)

    # 3.8: +2^24 in δT16 must not propagate past bit 26 -> T16[24] == 0 or T16[25] == 0
    if not (bit(T[16], 24) == 0 or bit(T[16], 25) == 0):
        bad["T16[24|25]"] = (bit(T[16], 25) << 1) | bit(T[16], 24)

    # 3.9: -2^29 in δT19 must not propagate past bit 31 -> T19[29] == 1 or T19[30] == 1
    if not (bit(T[19], 29) == 1 or bit(T[19], 30) == 1):
        bad["T19[29|30]"] = (bit(T[19], 30) << 1) | bit(T[19], 29)

    # 3.10: +2^17 in δT22 must not propagate past bit 17 -> T22[17] == 0
    if bit(T[22], 17) != 0:
        bad["T22[17]"] = bit(T[22], 17)

    # 3.11: +2^15 in δT34 must not propagate past bit 15 -> T34[15] == 0
    if bit(T[34], 15) != 0:
        bad["T34[15]"] = bit(T[34], 15)

    return (len(bad) == 0), bad


def check_next_block_iv_conditions(ihv_after_block1: Tuple[int, int, int, int]) -> Tuple[bool, Dict[str, int]]:
    # From section 5: IHV2[25] == 1 and IHV3[25] == 0 are needed for block 2 best path
    _, _, IHV2, IHV3 = ihv_after_block1
    ok = True
    issues: Dict[str, int] = {}
    if ((IHV2 >> 25) & 1) != 1:
        ok = False
        issues["IHV2[25]"] = (IHV2 >> 25) & 1
    if ((IHV3 >> 25) & 1) != 0:
        ok = False
        issues["IHV3[25]"] = (IHV3 >> 25) & 1
    return ok, issues


def check_recommended_iv_conditions(iv: Tuple[int, int, int, int]) -> Tuple[bool, Dict[str, int]]:
    # From section 5: IV2[25] = IV2[24] and IV3[25] = IV3[24] to avoid worst cases
    _, _, IV2, IV3 = iv
    ok = True
    issues: Dict[str, int] = {}
    
    iv2_25 = (IV2 >> 25) & 1
    iv2_24 = (IV2 >> 24) & 1
    iv3_25 = (IV3 >> 25) & 1 
    iv3_24 = (IV3 >> 24) & 1
    
    if iv2_25 != iv2_24:
        ok = False
        issues["IV2[25]!=IV2[24]"] = (iv2_25 << 1) | iv2_24
    if iv3_25 != iv3_24:
        ok = False
        issues["IV3[25]!=IV3[24]"] = (iv3_25 << 1) | iv3_24
    return ok, issues
