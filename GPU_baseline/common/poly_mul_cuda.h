#ifndef COMMON_POLY_MUL_CUDA_H__
#define COMMON_POLY_MUL_CUDA_H__

#include <cstdint>
#include <string>

namespace pcg_cuda {

struct PolyMulStats {
    double device_ms = 0.0;
};

bool is_cuda_available();
bool poly_mul_supported(uint64_t prime, int N, std::string* reason = nullptr);
std::string device_name();
std::string runtime_version();

// v1: hand-rolled radix-2 negacyclic NTT in Montgomery form (N <= 32768).
void poly_mul_u64(uint64_t* out, const uint64_t* a, const uint64_t* b,
                  int batch, int N, uint64_t prime,
                  PolyMulStats* stats = nullptr);

// GPU-NTT library backend, BALANCED ("square") four-step: n1 = 2^ceil(logN/2),
// n2 = 2^floor(logN/2), so both sub-transforms are single fused merge-NTT passes
// instead of one warp-local pass plus two partial passes. Every reorganisation is
// an explicit standalone GPU_Transpose so the movement share is measurable.
// Cyclic-only, wrapped in the psi-twist sandwich; logN in [8, 24].
bool poly_mul_gpuntt_square_supported(uint64_t prime, int N, std::string* reason = nullptr);
void poly_mul_u64_gpuntt_square(uint64_t* out, const uint64_t* a, const uint64_t* b,
                                int batch, int N, uint64_t prime,
                                PolyMulStats* stats = nullptr);

}  // namespace pcg_cuda

#endif  // COMMON_POLY_MUL_CUDA_H__
