// Host-facing entry points for the device AES-128 core (common/aes_gpu.cuh).
// Plain C++ header (no CUDA) so host-only translation units (e.g. tests) can
// include it without the nvcc device code.

#ifndef COMMON_AES_GPU_H__
#define COMMON_AES_GPU_H__

#include <cstdint>

namespace pcg_cuda {

// Standard AES-128 key schedule: 16-byte key -> 176 round-key bytes.
void aes128_expand_key_host(const uint8_t key[16], uint8_t round_keys[176]);

// Bring-up / correctness entry: AES-128 ECB encrypt `count` independent blocks
// on the GPU. keys, in, out are each count*16 bytes (one 16-byte key per block).
// Used by test/aes_gpu.cpp to assert bit-exactness vs emp-tool's AES.
void aes128_ecb_encrypt_gpu(const uint8_t* keys, const uint8_t* in,
                            uint8_t* out, int count);

}  // namespace pcg_cuda

#endif  // COMMON_AES_GPU_H__
