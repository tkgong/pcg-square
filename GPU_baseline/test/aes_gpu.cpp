// Phase 0 gate test: the device AES-128 (common/aes_gpu.cuh) must be bit-exact
// with emp-tool's AES (the PRG used by the DPF tree expansion). If this fails,
// nothing downstream (GPU DPF, GPU LPN) can be trusted, so we treat it as the
// foundation check.
//
// Checks:
//   1) FIPS-197 known-answer vector (pins our AES to the standard, independent
//      of emp).
//   2) Random (key, plaintext) pairs: GPU AES == emp::AES_ecb_encrypt_blks.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include <emp-tool/utils/aes.h>
#include <emp-tool/utils/block.h>

#include "common/aes_gpu.h"
#include "common/poly_mul_cuda.h"

namespace {

std::string hex(const uint8_t* p, int n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 0xf]; }
    return s;
}

// emp AES-128 ECB of a single 16-byte block (reference).
void emp_aes(const uint8_t key[16], const uint8_t in[16], uint8_t out[16]) {
    emp::block kb, cb;
    std::memcpy(&kb, key, 16);
    std::memcpy(&cb, in, 16);
    emp::AES_KEY k;
    emp::AES_set_encrypt_key(kb, &k);
    emp::AES_ecb_encrypt_blks(&cb, 1, &k);
    std::memcpy(out, &cb, 16);
}

}  // namespace

int main() {
    if (!pcg_cuda::is_cuda_available()) {
        std::cout << "[aes_gpu] no CUDA device — skipping\n";
        return 77;
    }

    int failures = 0;

    // 1) FIPS-197 Appendix B/C known-answer vector.
    {
        uint8_t key[16], in[16], out[16];
        for (int i = 0; i < 16; ++i) key[i] = static_cast<uint8_t>(i);
        const uint8_t pt[16] = {0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
                                0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff};
        const uint8_t ct[16] = {0x69,0xc4,0xe0,0xd8,0x6a,0x7b,0x04,0x30,
                                0xd8,0xcd,0xb7,0x80,0x70,0xb4,0xc5,0x5a};
        std::memcpy(in, pt, 16);
        pcg_cuda::aes128_ecb_encrypt_gpu(key, in, out, 1);
        if (std::memcmp(out, ct, 16) != 0) {
            std::cout << "[aes_gpu] FIPS-197 KAT FAILED: got " << hex(out, 16)
                      << " want " << hex(ct, 16) << "\n";
            ++failures;
        } else {
            std::cout << "[aes_gpu] FIPS-197 KAT passed\n";
        }
    }

    // 2) Random vectors vs emp.
    {
        const int count = 8192;
        std::mt19937_64 rng(0xC0FFEE);
        std::vector<uint8_t> keys(count * 16), in(count * 16), gpu(count * 16);
        for (auto& b : keys) b = static_cast<uint8_t>(rng());
        for (auto& b : in)   b = static_cast<uint8_t>(rng());

        pcg_cuda::aes128_ecb_encrypt_gpu(keys.data(), in.data(), gpu.data(), count);

        int mism = 0;
        for (int i = 0; i < count; ++i) {
            uint8_t ref[16];
            emp_aes(&keys[i * 16], &in[i * 16], ref);
            if (std::memcmp(ref, &gpu[i * 16], 16) != 0) {
                if (mism < 3) {
                    std::cout << "[aes_gpu] mismatch @" << i
                              << " key=" << hex(&keys[i * 16], 16)
                              << " pt="  << hex(&in[i * 16], 16)
                              << " gpu=" << hex(&gpu[i * 16], 16)
                              << " emp=" << hex(ref, 16) << "\n";
                }
                ++mism;
            }
        }
        if (mism) {
            std::cout << "[aes_gpu] random KAT FAILED: " << mism << "/" << count
                      << " mismatched\n";
            ++failures;
        } else {
            std::cout << "[aes_gpu] random KAT passed: " << count
                      << "/" << count << " match emp\n";
        }
    }

    if (failures) {
        std::cout << "[aes_gpu] FAIL\n";
        return 1;
    }
    std::cout << "[aes_gpu] PASS — device AES is bit-exact with emp\n";
    return 0;
}
