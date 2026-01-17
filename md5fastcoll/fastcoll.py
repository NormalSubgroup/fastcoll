from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable, List, Tuple

from .core import MASK32, MD5_IV, bytes_to_words_le, compress_block, u32, words_to_bytes_le

FASTCOLL_DEFAULT_IHV_HEX = "0123456789abcdeffedcba9876543210"


def parse_ihv_hex(hexstr: str) -> Tuple[int, int, int, int]:
    s = hexstr.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) != 32 or any(ch not in "0123456789abcdef" for ch in s):
        raise ValueError("IHV must be 32 hex characters")
    words: List[int] = []
    for i in range(4):
        w = 0
        for b in range(4):
            byte = int(s[i * 8 + b * 2 : i * 8 + b * 2 + 2], 16)
            w |= byte << (8 * b)
        words.append(w & MASK32)
    return (words[0], words[1], words[2], words[3])


def format_ihv_hex(ihv: Tuple[int, int, int, int]) -> str:
    parts = []
    for w in ihv:
        for b in range(4):
            parts.append(f"{(w >> (8 * b)) & 0xFF:02x}")
    return "".join(parts)


def random_iv(rng: random.Random) -> Tuple[int, int, int, int]:
    return (
        u32(rng.getrandbits(32)),
        u32(rng.getrandbits(32)),
        u32(rng.getrandbits(32)),
        u32(rng.getrandbits(32)),
    )


def recommended_iv(rng: random.Random) -> Tuple[int, int, int, int]:
    iv0, iv1, iv2, iv3 = random_iv(rng)
    iv2 = (iv2 & ~(1 << 25)) | (((iv2 >> 24) & 1) << 25)
    iv3 = (iv3 & ~(1 << 25)) | (((iv3 >> 24) & 1) << 25)
    return (u32(iv0), u32(iv1), u32(iv2), u32(iv3))


def iter_prefix_blocks(prefix: bytes) -> Iterable[bytes]:
    for off in range(0, len(prefix), 64):
        block = prefix[off : off + 64]
        if len(block) < 64:
            block += b"\x00" * (64 - len(block))
        yield block


def ihv_after_prefix(prefix: bytes, iv: Tuple[int, int, int, int] = MD5_IV) -> Tuple[Tuple[int, int, int, int], bytes]:
    ihv = (u32(iv[0]), u32(iv[1]), u32(iv[2]), u32(iv[3]))
    padded = b""
    for block in iter_prefix_blocks(prefix):
        padded += block
        ihv, _ = compress_block(ihv, bytes_to_words_le(block))
    return ihv, padded


def apply_fastcoll_delta(words: List[int], second_block: bool) -> List[int]:
    delta11 = -(1 << 15) if second_block else (1 << 15)
    out = list(words)
    out[4] = u32(out[4] + (1 << 31))
    out[11] = u32(out[11] + delta11)
    out[14] = u32(out[14] + (1 << 31))
    return out


def build_collision_messages(
    prefix_padded: bytes,
    block1_words: List[int],
    block2_words: List[int],
) -> Tuple[bytes, bytes]:
    msg1 = prefix_padded + words_to_bytes_le(block1_words) + words_to_bytes_le(block2_words)
    block1_b = apply_fastcoll_delta(block1_words, second_block=False)
    block2_b = apply_fastcoll_delta(block2_words, second_block=True)
    msg2 = prefix_padded + words_to_bytes_le(block1_b) + words_to_bytes_le(block2_b)
    return msg1, msg2


def default_output_names(prefix_path: Path) -> Tuple[Path, Path]:
    name = prefix_path.name
    if len(name) >= 4 and name[-4] == "." and "." not in name[-3:]:
        base = name[:-4]
        ext = name[-4:]
        out1 = prefix_path.with_name(base + "_msg1" + ext)
        out2 = prefix_path.with_name(base + "_msg2" + ext)
    else:
        out1 = prefix_path.with_name(name + "_msg1")
        out2 = prefix_path.with_name(name + "_msg2")
    idx = 1
    while out1.exists() or out2.exists():
        if len(name) >= 4 and name[-4] == "." and "." not in name[-3:]:
            base = name[:-4]
            ext = name[-4:]
            out1 = prefix_path.with_name(base + f"_msg1_{idx}" + ext)
            out2 = prefix_path.with_name(base + f"_msg2_{idx}" + ext)
        else:
            out1 = prefix_path.with_name(name + f"_msg1_{idx}")
            out2 = prefix_path.with_name(name + f"_msg2_{idx}")
        idx += 1
    return out1, out2
