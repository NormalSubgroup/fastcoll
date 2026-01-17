#include <iostream>
#include <streambuf>
#include "main.hpp"

void find_block1_wang(uint32 block[], const uint32 IV[]);
void find_block1_stevens_11(uint32 block[], const uint32 IV[]);
void find_block1_stevens_10(uint32 block[], const uint32 IV[]);
void find_block1_stevens_01(uint32 block[], const uint32 IV[]);
void find_block1_stevens_00(uint32 block[], const uint32 IV[]);

struct _NullBuffer final : public std::streambuf {
    int overflow(int c) override { return traits_type::not_eof(c); }
};

extern "C" void md5fastcoll_find_blocks(
    uint32 seed1,
    uint32 seed2,
    const uint32 IV_in[4],
    uint32 out_block0[16],
    uint32 out_block1[16]
) {
    seed32_1 = seed1;
    seed32_2 = seed2;
    if (seed32_1 == 0 && seed32_2 == 0) {
        seed32_2 = 0x12345678;
    }

    _NullBuffer nullbuf;
    std::streambuf* oldbuf = std::cout.rdbuf(&nullbuf);

    find_block0(out_block0, IV_in);

    uint32 ihv[4] = {IV_in[0], IV_in[1], IV_in[2], IV_in[3]};
    md5_compress(ihv, out_block0);

    // `find_block1` selection logic (from HashClash block1.cpp), without printing.
    if (
        ((ihv[1] ^ ihv[2]) & (1U << 31)) == 0 &&
        ((ihv[1] ^ ihv[3]) & (1U << 31)) == 0 &&
        (ihv[3] & (1U << 25)) == 0 &&
        (ihv[2] & (1U << 25)) == 0 &&
        (ihv[1] & (1U << 25)) == 0 &&
        ((ihv[2] ^ ihv[1]) & 1U) == 0
    ) {
        uint32 IV2[4] = {
            ihv[0] + (1U << 31),
            ihv[1] + (1U << 31) + (1U << 25),
            ihv[2] + (1U << 31) + (1U << 25),
            ihv[3] + (1U << 31) + (1U << 25),
        };

        // Stevens paths return msg2block1; convert to msg1block1 by applying delta (as in block1.cpp).
        if ((ihv[1] & (1U << 6)) != 0 && (ihv[1] & 1U) != 0) {
            find_block1_stevens_11(out_block1, IV2);
        } else if ((ihv[1] & (1U << 6)) != 0 && (ihv[1] & 1U) == 0) {
            find_block1_stevens_10(out_block1, IV2);
        } else if ((ihv[1] & (1U << 6)) == 0 && (ihv[1] & 1U) != 0) {
            find_block1_stevens_01(out_block1, IV2);
        } else {
            find_block1_stevens_00(out_block1, IV2);
        }

        out_block1[4] += 1U << 31;
        out_block1[11] += 1U << 15;
        out_block1[14] += 1U << 31;
    } else {
        find_block1_wang(out_block1, ihv);
    }

    std::cout.rdbuf(oldbuf);
}

