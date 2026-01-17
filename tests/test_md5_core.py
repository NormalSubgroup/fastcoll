import hashlib
import unittest

from md5fastcoll.md5_core import md5_digest


class TestMD5Core(unittest.TestCase):
    def test_md5_matches_hashlib(self) -> None:
        vectors = [
            b"",
            b"a",
            b"abc",
            b"message digest",
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        ]
        for m in vectors:
            self.assertEqual(md5_digest(m), hashlib.md5(m).digest())


if __name__ == "__main__":
    unittest.main()
