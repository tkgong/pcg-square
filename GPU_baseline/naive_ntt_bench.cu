// naive_ntt_bench.cu — TEXTBOOK per-stage radix-2 NTT poly-mul baseline.
//
// The "naive" tier of the naive -> 4-step ladder: one kernel launch
// per butterfly stage, the whole array crosses DRAM once per stage (logN
// round trips per transform). Negacyclic mul via the standard psi-twist
// sandwich; DIF forward (natural -> bit-reversed) + mirrored DIT inverse
// (bit-reversed -> natural), pointwise in the bit-reversed domain.
//
// Fairness: the butterfly modmul is gpuntt's own patched Barrett
// (OPERATOR_GPU<Data64>::mult) -- identical arithmetic quality to the square backend, so the measured difference is purely structural
// (DRAM pass count), not modmul quality.
//
// Correctness gate: output compared ELEMENTWISE against the 4-step backend
// (poly_mul_u64_gpuntt_square) on the same inputs -> BITEXACT or the run is
// invalid. Two structurally independent implementations must agree.
//
// Build: /usr/local/cuda-12.6/bin/nvcc -O3 -std=c++17 -arch=sm_89 -I. \
//   -I$HOME/.local/include/GPUNTT-1.0 endtoend/naive_ntt_bench.cu \
//   common/poly_mul_gpuntt_square.cu common/poly_mul_cuda.cu \
//   -L$HOME/.local/lib -lntt-1.0 -o naive_ntt_bench
// Run:   naive_ntt_bench --logN 20 --batch 16 --iters 5

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "common/poly_mul_cuda.h"
#include "common/ntt_roots.h"
#include "gpuntt/common/modular_arith.cuh"

using pcg_cuda::mod_mul_u64;
using pcg_cuda::mod_inv_u64;
using pcg_cuda::negacyclic_psi;

static constexpr uint64_t kPrime = 4611686018326724609ULL;   // 2^62-3*2^25+1

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// ---- kernels ---------------------------------------------------------------

__global__ void twist_kernel(Data64* a, const Data64* tw, size_t N,
                             size_t total, Modulus<Data64> mod) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total) a[i] = OPERATOR_GPU<Data64>::mult(a[i], tw[i % N], mod);
}

// One DIF stage, half-size m: pairs (i, i+m); X[i]=u+v, X[i+m]=(u-v)*w^off*s.
__global__ void dif_stage(Data64* a, const Data64* w, size_t N, size_t m,
                          size_t total_pairs, Modulus<Data64> mod) {
    size_t p = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (p >= total_pairs) return;
    const size_t half = N >> 1;
    const size_t b = p / half, q = p % half;
    const size_t blk = q / m, off = q % m;
    const size_t i = b * N + blk * 2 * m + off;
    const Data64 u = a[i], v = a[i + m];
    a[i] = OPERATOR_GPU<Data64>::add(u, v, mod);
    Data64 d = OPERATOR_GPU<Data64>::sub(u, v, mod);
    a[i + m] = OPERATOR_GPU<Data64>::mult(d, w[off * (N / (2 * m))], mod);
}

// Mirrored DIT stage (exact inverse of dif_stage with inverse twiddles):
// t = a[i+m]*winv^off*s; a[i]=u+t; a[i+m]=u-t.
__global__ void dit_stage(Data64* a, const Data64* winv, size_t N, size_t m,
                          size_t total_pairs, Modulus<Data64> mod) {
    size_t p = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (p >= total_pairs) return;
    const size_t half = N >> 1;
    const size_t b = p / half, q = p % half;
    const size_t blk = q / m, off = q % m;
    const size_t i = b * N + blk * 2 * m + off;
    const Data64 u = a[i];
    const Data64 t = OPERATOR_GPU<Data64>::mult(a[i + m],
                                                winv[off * (N / (2 * m))], mod);
    a[i] = OPERATOR_GPU<Data64>::add(u, t, mod);
    a[i + m] = OPERATOR_GPU<Data64>::sub(u, t, mod);
}

__global__ void pointwise_kernel(Data64* a, const Data64* b, size_t total,
                                 Modulus<Data64> mod) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total) a[i] = OPERATOR_GPU<Data64>::mult(a[i], b[i], mod);
}

// ---- host ------------------------------------------------------------------

static uint64_t fnv(const std::vector<uint64_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t x : v) { h ^= x; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    int logN = 20, batch = 16, iters = 5;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--logN")) logN = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch")) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    }
    const size_t N = 1ull << logN, total = N * batch;
    const size_t bytes = total * sizeof(Data64);

    // roots: omega = psi^2 (N-th root); tables built iteratively on host
    const uint64_t psi = negacyclic_psi(kPrime, N);
    const uint64_t psi_inv = mod_inv_u64(psi, kPrime);
    const uint64_t omega = mod_mul_u64(psi, psi, kPrime);
    const uint64_t omega_inv = mod_inv_u64(omega, kPrime);
    const uint64_t n_inv = mod_inv_u64(N % kPrime, kPrime);

    std::vector<uint64_t> w(N / 2), winv(N / 2), twf(N), twi(N);
    w[0] = winv[0] = 1;
    for (size_t k = 1; k < N / 2; ++k) {
        w[k] = mod_mul_u64(w[k - 1], omega, kPrime);
        winv[k] = mod_mul_u64(winv[k - 1], omega_inv, kPrime);
    }
    twf[0] = 1; twi[0] = n_inv;
    for (size_t k = 1; k < N; ++k) {
        twf[k] = mod_mul_u64(twf[k - 1], psi, kPrime);
        twi[k] = mod_mul_u64(twi[k - 1], psi_inv, kPrime);   // psi^-k * n_inv
    }

    std::mt19937_64 rng(11);
    std::vector<uint64_t> ha(total), hb(total);
    for (auto& x : ha) x = rng() % kPrime;
    for (auto& x : hb) x = rng() % kPrime;

    Data64 *d_a, *d_b, *d_w, *d_winv, *d_twf, *d_twi;
    CK(cudaMalloc(&d_a, bytes)); CK(cudaMalloc(&d_b, bytes));
    CK(cudaMalloc(&d_w, N / 2 * 8)); CK(cudaMalloc(&d_winv, N / 2 * 8));
    CK(cudaMalloc(&d_twf, N * 8)); CK(cudaMalloc(&d_twi, N * 8));
    CK(cudaMemcpy(d_w, w.data(), N / 2 * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_winv, winv.data(), N / 2 * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_twf, twf.data(), N * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_twi, twi.data(), N * 8, cudaMemcpyHostToDevice));

    const Modulus<Data64> mod(kPrime);
    const unsigned TPB = 256;
    const size_t pairs = total / 2;
    const unsigned gE = (unsigned)((total + TPB - 1) / TPB);
    const unsigned gP = (unsigned)((pairs + TPB - 1) / TPB);

    auto run_naive = [&]() {
        // fwd(a), fwd(b): DIF m = N/2 .. 1
        twist_kernel<<<gE, TPB>>>(d_a, d_twf, N, total, mod);
        twist_kernel<<<gE, TPB>>>(d_b, d_twf, N, total, mod);
        for (size_t m = N / 2; m >= 1; m >>= 1) {
            dif_stage<<<gP, TPB>>>(d_a, d_w, N, m, pairs, mod);
            dif_stage<<<gP, TPB>>>(d_b, d_w, N, m, pairs, mod);
        }
        pointwise_kernel<<<gE, TPB>>>(d_a, d_b, total, mod);
        for (size_t m = 1; m <= N / 2; m <<= 1)
            dit_stage<<<gP, TPB>>>(d_a, d_winv, N, m, pairs, mod);
        twist_kernel<<<gE, TPB>>>(d_a, d_twi, N, total, mod);
    };

    // correctness gate vs the 4-step backend, then timing
    std::vector<uint64_t> ref(total), got(total);
    pcg_cuda::poly_mul_u64_gpuntt_square(ref.data(), ha.data(), hb.data(),
                                  batch, (int)N, kPrime, nullptr);

    CK(cudaMemcpy(d_a, ha.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, hb.data(), bytes, cudaMemcpyHostToDevice));
    run_naive(); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    const bool ok = (ref == got);

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float ms = 0, acc = 0;
    for (int it = 0; it < iters; ++it) {
        CK(cudaMemcpy(d_a, ha.data(), bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_b, hb.data(), bytes, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(e0));
        run_naive();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1)); acc += ms;
    }
    acc /= iters;

    printf("backend=naive N=%zu batch=%d iters=%d device_ms=%.4f "
           "ms_per_mul=%.4f checksum=%016lx vs_square=%s tier=phase-wall\n",
           N, batch, iters, acc, acc / batch, (unsigned long)fnv(got),
           ok ? "BITEXACT" : "MISMATCH");
    return ok ? 0 : 1;
}
