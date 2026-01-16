import hashlib
from md5fastcoll.md5_core import md5_digest


def test_md5_matches_hashlib():
    vectors = [
        b"", b"a", b"abc", b"message digest", b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    ]
    for m in vectors:
        assert md5_digest(m) == hashlib.md5(m).digest()