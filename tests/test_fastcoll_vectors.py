import unittest

from md5fastcoll.core import bytes_to_words_le, u32
from md5fastcoll.fastcoll import apply_fastcoll_delta
from md5fastcoll.md5 import md5_hex


FASTCOLL1_HEX = (
    "2f3d2d3d2d3d2d3d2d3d2d3d2d3d2d5c7c2020204964656e746963616c0000"
    "7c7c20202020507265666978200000207c5c3d2d3d2d3d2d3d2d3d2d3d2d3d"
    "2d2f379ae6c3dc19edf5725bb4e473df31bcc6316c9edfaf6c7c51ce444ac6"
    "b3a7d46da2fbe6ea6e46a54b2a5a3c8a6b6cbe217f84d2ae750611dadc4c568"
    "7f378b664c4150ac4b2d1c2aac9573d6f357e48286e793b25c63e27c91a7639"
    "ec4602661e64a65704d0fa4e888344b7f1dcc2ece695a79e6d52bf6bba60990"
    "2a89ec39e"
)

FASTCOLL2_HEX = (
    "2f3d2d3d2d3d2d3d2d3d2d3d2d3d2d5c7c2020204964656e746963616c0000"
    "7c7c20202020507265666978200000207c5c3d2d3d2d3d2d3d2d3d2d3d2d3d"
    "2d2f379ae6c3dc19edf5725bb4e473df31bcc6316c1edfaf6c7c51ce444ac6"
    "b3a7d46da2fbe6ea6e46a54b2a5a3c8aeb6cbe217f84d2ae750611dadc4cd68"
    "7f378b664c4150ac4b2d1c2aac9573d6f357e48286e79bb25c63e27c91a7639"
    "ec4602661e64a65704d0fa4e888344b7f15cc2ece695a79e6d52bf6bba60998"
    "2a89ec39e"
)


class TestFastcollVectors(unittest.TestCase):
    def test_fastcoll_sample_collision(self) -> None:
        m1 = bytes.fromhex(FASTCOLL1_HEX)
        m2 = bytes.fromhex(FASTCOLL2_HEX)
        self.assertEqual(len(m1), 192)
        self.assertEqual(len(m2), 192)
        self.assertNotEqual(m1, m2)
        self.assertEqual(md5_hex(m1), md5_hex(m2))

    def test_fastcoll_delta_mask(self) -> None:
        m1 = bytes.fromhex(FASTCOLL1_HEX)
        m2 = bytes.fromhex(FASTCOLL2_HEX)
        b1_0 = m1[64:128]
        b1_1 = m1[128:192]
        b2_0 = m2[64:128]
        b2_1 = m2[128:192]
        w1_0 = bytes_to_words_le(b1_0)
        w1_1 = bytes_to_words_le(b1_1)
        w2_0 = bytes_to_words_le(b2_0)
        w2_1 = bytes_to_words_le(b2_1)

        w1_0b = apply_fastcoll_delta(w1_0, second_block=False)
        w1_1b = apply_fastcoll_delta(w1_1, second_block=True)
        self.assertEqual(w1_0b, w2_0)
        self.assertEqual(w1_1b, w2_1)

        for i in range(16):
            if i in (4, 11, 14):
                continue
            self.assertEqual(w1_0[i], w2_0[i])
            self.assertEqual(w1_1[i], w2_1[i])

        self.assertEqual(u32(w1_0[4] + (1 << 31)), w2_0[4])
        self.assertEqual(u32(w1_0[11] + (1 << 15)), w2_0[11])
        self.assertEqual(u32(w1_0[14] + (1 << 31)), w2_0[14])

        self.assertEqual(u32(w1_1[4] + (1 << 31)), w2_1[4])
        self.assertEqual(u32(w1_1[11] - (1 << 15)), w2_1[11])
        self.assertEqual(u32(w1_1[14] + (1 << 31)), w2_1[14])


if __name__ == "__main__":
    unittest.main()
