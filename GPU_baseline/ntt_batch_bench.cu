// Batched NTT poly-mul bench for the GPU-NTT backends (baseline_v2 PHASE 2).
//
// Extends endtoend/ntt_trace_bench.cu (single poly-mul, trace capture) with a
// --batch flag and the 4-step backend, for the L40S batch sweep + 4-step stage
// split. Backends:
//   --backend v1     : hand-rolled radix-2 Montgomery NTT (N <= 32768).
//   --backend square : BALANCED four-step (n1 = 2^ceil(logN/2)), every
//                      reorganisation an explicit standalone pass.
//   (the skewed GPU-NTT 4-step backend was removed from this branch:
//                      logN in [8,24]; set PCG_SQUARE_STAGE_MS=1 for the
//                      per-stage device-time split on stderr.
//
// Each call runs `batch` independent poly-muls (fwd NTT x2 + pointwise + inv)
// in one launch set via the batch parameter of pcg_cuda::poly_mul_u64_gpuntt /
// _gpuntt_square. --iters repeats the whole call (first extra call is warmup
// when iters > 1); nsys runs should use --iters 1 (single shot, kernel-only
// sums come from the profile).
//
// Build (L40S):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 -I. \
//       -I$HOME/GPU-NTT/install/include/GPUNTT-1.0 -DHAVE_GPUNTT \
//       endtoend/ntt_batch_bench.cu common/poly_mul_cuda.cu \
//       common/poly_mul_gpuntt_square.cu \
//       -L$HOME/GPU-NTT/install/lib -lntt-1.0 -o endtoend/bin/ntt_batch_bench
#include "common/poly_mul_cuda.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static const uint64_t P = 4611686018326724609ULL;

int main(int argc, char** argv) {
    int N = 4096, batch = 1, iters = 1;
    const char* backend = "square";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
    }
    size_t total = (size_t)batch * N;
    std::vector<uint64_t> a(total), b(total), out(total);
    for (size_t j = 0; j < total; ++j) {
        a[j] = (0x9e3779b97f4a7c15ULL * (j + 1)) % P;
        b[j] = (0xc2b2ae3d27d4eb4fULL * (j + 7)) % P;
    }
    std::string why;
    pcg_cuda::PolyMulStats st;
    double dev_ms = 0.0;
    int warm = (iters > 1) ? 1 : 0;
    for (int it = 0; it < warm + iters; ++it) {
        if (!strcmp(backend, "v1")) {
            pcg_cuda::poly_mul_u64(out.data(), a.data(), b.data(), batch, N, P, &st);
        } else if (!strcmp(backend, "square")) {
            if (!pcg_cuda::poly_mul_gpuntt_square_supported(P, N, &why)) {
                fprintf(stderr, "square unsupported: %s\n", why.c_str()); return 1;
            }
            pcg_cuda::poly_mul_u64_gpuntt_square(out.data(), a.data(), b.data(), batch, N, P, &st);
        } else {
            fprintf(stderr, "unknown backend %s\n", backend); return 1;
        }
        if (it >= warm) dev_ms += st.device_ms;
    }
    dev_ms /= iters;
    uint64_t cks = 0;
    for (size_t j = 0; j < total; ++j) cks ^= out[j];
    printf("backend=%s N=%d batch=%d iters=%d device_ms=%.4f ms_per_mul=%.4f checksum=%016llx tier=phase-wall\n",
           backend, N, batch, iters, dev_ms, dev_ms / batch, (unsigned long long)cks);
    return 0;
}
