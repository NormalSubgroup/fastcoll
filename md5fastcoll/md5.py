from __future__ import annotations

from typing import Tuple
from .core import (
    MD5_IV,
    compress_block,
    md5_padding,
    bytes_to_words_le,
    u32,
)


def md5_bytes(data: bytes, iv: Tuple[int, int, int, int] = MD5_IV) -> bytes:
    ihv = (u32(iv[0]), u32(iv[1]), u32(iv[2]), u32(iv[3]))
    msg = data + md5_padding(len(data))
    for off in range(0, len(msg), 64):
        block = msg[off : off + 64]
        words = bytes_to_words_le(block)
        ihv, _ = compress_block(ihv, words)
    # digest is little-endian of ihv words in order (IV0, IV1, IV2, IV3)
    return (
        u32(ihv[0]).to_bytes(4, "little")
        + u32(ihv[1]).to_bytes(4, "little")
        + u32(ihv[2]).to_bytes(4, "little")
        + u32(ihv[3]).to_bytes(4, "little")
    )


def md5_hex(data: bytes, iv: Tuple[int, int, int, int] = MD5_IV) -> str:
    return md5_bytes(data, iv).hex()