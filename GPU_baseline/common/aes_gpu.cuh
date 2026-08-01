// Self-contained device AES-128 (forward direction only), bit-exact with
// emp-tool's AES (both are standard FIPS-197 AES-128, so for identical 16-byte
// key + 16-byte plaintext they yield identical 16-byte ciphertext).
//
// Byte-oriented reference implementation (SubBytes/ShiftRows/MixColumns via the
// S-box + xtime), matching the layout of the public-domain tiny-AES-c "Cipher":
// the 16-byte state s[] maps s[i*4 + j] = state column i, row j; input/output
// bytes are in memory order, and round-key bytes are XORed positionally. This
// is deliberately simple (correctness-first); it is the GPU PRG core shared by
// the DPF tree expansion (CCR hash H_k(x)=AES_k(x)^x) and the LPN kernel.
//
// Key schedules are expanded ONCE on the host (aes128_expand_key_host) and
// uploaded; the device only runs the 10-round encryption.

#ifndef COMMON_AES_GPU_CUH__
#define COMMON_AES_GPU_CUH__

#include <cstdint>
#include <cstring>

#include "common/aes_gpu.h"

namespace pcg_cuda {
namespace aesdev {

// Standard AES forward S-box.
__device__ static const uint8_t kSboxDev[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
};

__device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

// Encrypt a single 16-byte block in place. round_keys = 176 bytes. `sbox` is
// the 256-byte S-box table; kernels with hot AES loops should stage it (and the
// round keys) in __shared__ memory and pass those copies in — the global
// kSboxDev overload below is the convenience path.
__device__ __forceinline__ void encrypt_block(const uint8_t* round_keys, uint8_t s[16],
                                              const uint8_t* sbox) {
    // AddRoundKey round 0.
    #pragma unroll
    for (int k = 0; k < 16; ++k) s[k] ^= round_keys[k];

    for (int round = 1; round <= 10; ++round) {
        // SubBytes.
        #pragma unroll
        for (int k = 0; k < 16; ++k) s[k] = sbox[s[k]];

        // ShiftRows (s[i*4 + j] = state column i, row j).
        uint8_t tmp;
        // row 1: rotate left by 1
        tmp = s[1];  s[1] = s[5];  s[5] = s[9];  s[9] = s[13]; s[13] = tmp;
        // row 2: rotate left by 2
        tmp = s[2];  s[2] = s[10]; s[10] = tmp;
        tmp = s[6];  s[6] = s[14]; s[14] = tmp;
        // row 3: rotate left by 3 (== right by 1)
        tmp = s[3];  s[3] = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = tmp;

        if (round != 10) {
            // MixColumns.
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint8_t* c = &s[i * 4];
                uint8_t t  = c[0];
                uint8_t all = c[0] ^ c[1] ^ c[2] ^ c[3];
                c[0] ^= xtime(c[0] ^ c[1]) ^ all;
                c[1] ^= xtime(c[1] ^ c[2]) ^ all;
                c[2] ^= xtime(c[2] ^ c[3]) ^ all;
                c[3] ^= xtime(c[3] ^ t)    ^ all;
            }
            // AddRoundKey round.
            #pragma unroll
            for (int k = 0; k < 16; ++k) s[k] ^= round_keys[round * 16 + k];
        } else {
            // Final AddRoundKey (round 10).
            #pragma unroll
            for (int k = 0; k < 16; ++k) s[k] ^= round_keys[160 + k];
        }
    }
}

__device__ __forceinline__ void encrypt_block(const uint8_t* round_keys, uint8_t s[16]) {
    encrypt_block(round_keys, s, kSboxDev);
}

// Convenience: CCR hash H_k(x) = AES_k(x) ^ x on a 128-bit block held as two
// little-endian uint64 halves (lo = bytes 0..7, hi = bytes 8..15) — the in-memory
// layout of emp::block. Returns the hashed block in (out_lo, out_hi).
__device__ __forceinline__ void ccr_hash_u64(const uint8_t* round_keys,
                                             const uint8_t* sbox,
                                             uint64_t lo, uint64_t hi,
                                             uint64_t& out_lo, uint64_t& out_hi) {
    uint8_t s[16];
    std::memcpy(&s[0], &lo, 8);
    std::memcpy(&s[8], &hi, 8);
    encrypt_block(round_keys, s, sbox);
    uint64_t clo, chi;
    std::memcpy(&clo, &s[0], 8);
    std::memcpy(&chi, &s[8], 8);
    out_lo = clo ^ lo;
    out_hi = chi ^ hi;
}

__device__ __forceinline__ void ccr_hash_u64(const uint8_t* round_keys,
                                             uint64_t lo, uint64_t hi,
                                             uint64_t& out_lo, uint64_t& out_hi) {
    ccr_hash_u64(round_keys, kSboxDev, lo, hi, out_lo, out_hi);
}

// Plain AES (no CCR) on a 128-bit block held as two uint64 halves.
__device__ __forceinline__ void encrypt_u64(const uint8_t* round_keys,
                                            uint64_t lo, uint64_t hi,
                                            uint64_t& out_lo, uint64_t& out_hi) {
    uint8_t s[16];
    std::memcpy(&s[0], &lo, 8);
    std::memcpy(&s[8], &hi, 8);
    encrypt_block(round_keys, s);
    std::memcpy(&out_lo, &s[0], 8);
    std::memcpy(&out_hi, &s[8], 8);
}

}  // namespace aesdev
}  // namespace pcg_cuda

#endif  // COMMON_AES_GPU_CUH__
