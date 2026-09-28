// Fused four-step negacyclic polynomial multiplication in R_p = Z_p[X]/(X^N+1),
// split into an SM lane (butterfly kernels) and a DRU lane (layout transposes).
//
// WHY THIS BACKEND EXISTS
// -----------------------
// The square backend (common/poly_mul_gpuntt_square.cu) keeps every
// reorganisation as its own global-memory pass: 2 psi-twists, 3 twiddles,
// 1 scale and 6 bit-reversals on top of the 6 transposes, and at logN >= 23
// each length-2^12 sub-transform costs two library kernels. Offloading only the
// transposes to the DRU therefore leaves the SMs with MORE passes than the
// GPU-NTT merge backend (10 kernels per poly-mul, no transposes at all).
//
// Here the SM lane is only what arithmetic requires:
//   * every sub-transform is ONE shared-memory kernel (n1, n2 <= 2^12);
//   * the psi-twist is absorbed into the column transform, which is itself a
//     negacyclic NTT (root psi1 = psi^n2);
//   * the four-step twiddle, the cyclic<->negacyclic correction of the row
//     transform, and 1/N are one per-element factor fused into the row kernel's
//     load (forward) or store (inverse);
//   * bit-reversal is never materialised: the twiddle tables are indexed in the
//     order the kernels produce, and the pointwise product is order-agnostic.
// So one poly-mul is 7 SM kernels (fwd a: 2, fwd b: 2, pointwise, inv: 2) and
// 6 transposes. The transposes are pure data movement with no arithmetic, i.e.
// exactly what the channel-level DRU executes; they are issued as separate
// kernels so the two lanes can be timed separately.
//
// DERIVATION (index conventions are auditable)
// --------------------------------------------
// psi: primitive 2N-th root.  n = i*n2 + j  (i<n1, j<n2),  k = k1 + n1*k2.
//   A[k] = sum_n a[n] psi^{n(2k+1)}
//        = sum_j omega2^{j k2} * psi^{j(2k1+1)} * sum_i a[i n2 + j] psi1^{i(2k1+1)}
//   with psi1 = psi^{n2} (2n1-th root), omega2 = psi^{2 n1} (n2-th root).
// The inner sum is a negacyclic length-n1 NTT of column j. The outer sum is a
// cyclic length-n2 NTT; writing it as a negacyclic NTT with psi2 = psi^{n1}
// (psi2^2 = omega2) needs the input pre-multiplied by psi2^{-j}, so the row
// kernel's fused factor is  TW[k1][j] = psi^{j(2k1+1)} * psi2^{-j}
//                                      = psi^{j(2k1+1-n1)}.
// Inverse: c[n] = N^{-1} sum_k C[k] psi^{-n(2k+1)}; the mirrored factor
//   TWI[k1][j] = N^{-1} psi^{-j(2k1+1-n1)} is fused into the inverse row
// kernel's store.
//
// Pipeline for one operand X (n1 x n2 row-major, row i = coefficients i*n2..):
//   T1  [DRU] X (n1 x n2) -> Y (n2 x n1)          columns become rows
//   K1  [SM ] Y rows: negacyclic NTT_n1 (psi1)     out: bit-reversed k1
//   T2  [DRU] Y (n2 x n1) -> X (n1 x n2)           row p holds k1 = brev(p)
//   K2  [SM ] X rows: x TW[brev(p)][j], NTT_n2 (psi2)  out: bit-reversed k2
// then PW [SM] C = A o B, and for C:
//   K3  [SM ] rows: INTT_n2 (psi2), x TWI[brev(p)][j]  in: bit-reversed k2
//   T4  [DRU] (n1 x n2) -> (n2 x n1)               row j over p = brev(k1)
//   K4  [SM ] rows: INTT_n1 (psi1)                 in: bit-reversed k1
//   T5  [DRU] (n2 x n1) -> (n1 x n2)               natural coefficient order
//
// Arithmetic: GPU-NTT's CooleyTukeyUnit / GentlemanSandeUnit / OPERATOR_GPU
// Barrett multiply (the same device code the merge backend runs), with the
// 62-bit-prime shift fix already applied to the installed GPU-NTT headers.

#ifndef GPU_BASELINE_FUSED4_NTT_CUH__
#define GPU_BASELINE_FUSED4_NTT_CUH__

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gpuntt/ntt_merge/ntt.cuh"
#include "common/ntt_roots.h"

namespace fused4 {

using gpuntt::CooleyTukeyUnit;
using gpuntt::GentlemanSandeUnit;

inline void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::ostringstream os;
    os << "fused4: " << what << ": " << cudaGetErrorString(err);
    throw std::runtime_error(os.str());
}

inline int bitrev(int x, int bits) {
    int r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// ---------------------------------------------------------------------------
// SM lane
// ---------------------------------------------------------------------------

// Forward negacyclic NTT of every length-2^logn row (one block per row).
// Natural order in, bit-reversed order out, psi merged into the butterflies
// (Longa-Naehrig Alg. 1 with psi_rev[k] = psi^{bitrev(k)}). With PRE, element
// idx of row r is first multiplied by pre[(r % rows_per_poly) * n + idx].
template <bool PRE>
__global__ void __launch_bounds__(1024)
fwd_rows(Data64* __restrict__ data, const Data64* __restrict__ psi_rev,
         const Data64* __restrict__ pre, Modulus<Data64> mod, int logn,
         int rows_per_poly) {
    extern __shared__ Data64 s[];
    const int n = 1 << logn;
    const size_t row = blockIdx.x;
    Data64* g = data + row * n;
    const Data64* p = PRE ? pre + static_cast<size_t>(row % rows_per_poly) * n : nullptr;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        Data64 v = g[i];
        if (PRE) v = OPERATOR_GPU<Data64>::mult(v, p[i], mod);
        s[i] = v;
    }
    __syncthreads();
    const int half = n >> 1;
    int logt = logn - 1;
    for (int m = 1; m < n; m <<= 1, --logt) {
        const int tmask = (1 << logt) - 1;
        for (int b = threadIdx.x; b < half; b += blockDim.x) {
            const int i = b >> logt;
            const int j = (i << (logt + 1)) + (b & tmask);
            CooleyTukeyUnit(s[j], s[j + (1 << logt)], psi_rev[m + i], mod);
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) g[i] = s[i];
}

// Inverse negacyclic NTT of every row (Longa-Naehrig Alg. 2 without the final
// n^{-1}): bit-reversed order in, natural order out, psiinv_rev[k] =
// psi^{-bitrev(k)}. With POST, element idx of row r is multiplied by
// post[(r % rows_per_poly) * n + idx] on the way out.
template <bool POST>
__global__ void __launch_bounds__(1024)
inv_rows(Data64* __restrict__ data, const Data64* __restrict__ psiinv_rev,
         const Data64* __restrict__ post, Modulus<Data64> mod, int logn,
         int rows_per_poly) {
    extern __shared__ Data64 s[];
    const int n = 1 << logn;
    const size_t row = blockIdx.x;
    Data64* g = data + row * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = g[i];
    __syncthreads();
    const int half = n >> 1;
    int logt = 0;
    for (int h = half; h >= 1; h >>= 1, ++logt) {
        const int tmask = (1 << logt) - 1;
        for (int b = threadIdx.x; b < half; b += blockDim.x) {
            const int i = b >> logt;
            const int j = (i << (logt + 1)) + (b & tmask);
            GentlemanSandeUnit(s[j], s[j + (1 << logt)], psiinv_rev[h + i], mod);
        }
        __syncthreads();
    }
    const Data64* p = POST ? post + static_cast<size_t>(row % rows_per_poly) * n : nullptr;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        Data64 v = s[i];
        if (POST) v = OPERATOR_GPU<Data64>::mult(v, p[i], mod);
        g[i] = v;
    }
}

__global__ void pointwise_kernel(Data64* __restrict__ a, const Data64* __restrict__ b,
                                 Modulus<Data64> mod, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) a[idx] = OPERATOR_GPU<Data64>::mult(a[idx], b[idx], mod);
}

// ---------------------------------------------------------------------------
// DRU lane: batched out-of-place transpose of R x C matrices (no arithmetic).
// On the GPU it runs as a 32x33 shared-memory tile transpose; in PCG^2 it is
// the channel-level DRU's work and leaves the SM timeline.
// ---------------------------------------------------------------------------
__global__ void transpose_kernel(const Data64* __restrict__ src, Data64* __restrict__ dst,
                                 int R, int C) {
    __shared__ Data64 tile[32][33];
    const size_t base = static_cast<size_t>(blockIdx.z) * R * C;
    const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int t = threadIdx.y; t < 32; t += blockDim.y) {
        const int r = r0 + t, c = c0 + threadIdx.x;
        if (r < R && c < C) tile[t][threadIdx.x] = src[base + static_cast<size_t>(r) * C + c];
    }
    __syncthreads();
    for (int t = threadIdx.y; t < 32; t += blockDim.y) {
        const int c = c0 + t, r = r0 + threadIdx.x;
        if (c < C && r < R) dst[base + static_cast<size_t>(c) * R + r] = tile[threadIdx.x][t];
    }
}

// ---------------------------------------------------------------------------
// Plan: tables + launch helpers for one ring degree N = 2^logN.
// ---------------------------------------------------------------------------
struct Plan {
    int logN, N, log1, log2, n1, n2;
    uint64_t p;
    Modulus<Data64> mod;
    Data64 *psi1_rev = nullptr, *psi1inv_rev = nullptr;
    Data64 *psi2_rev = nullptr, *psi2inv_rev = nullptr;
    Data64 *tw = nullptr, *twi = nullptr;

    // row_log > 0 fixes the row-transform length n2 = 2^row_log (n1 = N / n2);
    // default is the balanced split n1 = 2^ceil(logN/2), n2 = 2^floor(logN/2).
    Plan(int logN_, uint64_t prime, int row_log = 0) : logN(logN_), N(1 << logN_), p(prime), mod(prime) {
        if (logN < 8 || logN > 24) throw std::runtime_error("fused4: logN must be in [8, 24]");
        log2 = row_log > 0 ? row_log : logN / 2;
        log1 = logN - log2;
        n1 = 1 << log1;
        n2 = 1 << log2;
        using pcg_cuda::mod_mul_u64;
        using pcg_cuda::mod_pow_u64;
        using pcg_cuda::mod_inv_u64;
        const uint64_t psi = pcg_cuda::negacyclic_psi(p, N);
        const uint64_t psi_inv = mod_inv_u64(psi, p);
        const uint64_t psi1 = mod_pow_u64(psi, n2, p), psi1i = mod_inv_u64(psi1, p);
        const uint64_t psi2 = mod_pow_u64(psi, n1, p), psi2i = mod_inv_u64(psi2, p);
        const uint64_t n_inv = mod_inv_u64(static_cast<uint64_t>(N), p);

        auto rev_table = [&](uint64_t root, int lg) {
            const int n = 1 << lg;
            std::vector<uint64_t> pw(n), t(n);
            uint64_t x = 1;
            for (int k = 0; k < n; ++k) { pw[k] = x; x = mod_mul_u64(x, root, p); }
            for (int k = 0; k < n; ++k) t[k] = pw[bitrev(k, lg)];
            return t;
        };
        upload(&psi1_rev, rev_table(psi1, log1));
        upload(&psi1inv_rev, rev_table(psi1i, log1));
        upload(&psi2_rev, rev_table(psi2, log2));
        upload(&psi2inv_rev, rev_table(psi2i, log2));

        // TW[p][j]  = psi^{ j(2k1+1-n1)},  TWI[p][j] = N^{-1} psi^{-j(2k1+1-n1)},
        // k1 = bitrev(p): row p of the (n1 x n2) layout after T2 holds k1 = brev(p).
        std::vector<uint64_t> h_tw(static_cast<size_t>(N)), h_twi(static_cast<size_t>(N));
        const uint64_t twoN = 2ull * N;
        for (int prow = 0; prow < n1; ++prow) {
            const int64_t k1 = bitrev(prow, log1);
            int64_t e = (2 * k1 + 1 - n1) % static_cast<int64_t>(twoN);
            if (e < 0) e += twoN;
            const uint64_t r = mod_pow_u64(psi, static_cast<uint64_t>(e), p);
            const uint64_t ri = mod_pow_u64(psi_inv, static_cast<uint64_t>(e), p);
            uint64_t x = 1, xi = n_inv;
            for (int j = 0; j < n2; ++j) {
                h_tw[static_cast<size_t>(prow) * n2 + j] = x;
                h_twi[static_cast<size_t>(prow) * n2 + j] = xi;
                x = mod_mul_u64(x, r, p);
                xi = mod_mul_u64(xi, ri, p);
            }
        }
        upload(&tw, h_tw);
        upload(&twi, h_twi);
    }

    ~Plan() {
        cudaFree(psi1_rev); cudaFree(psi1inv_rev);
        cudaFree(psi2_rev); cudaFree(psi2inv_rev);
        cudaFree(tw); cudaFree(twi);
    }
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    static void upload(Data64** dst, const std::vector<uint64_t>& v) {
        check(cudaMalloc(reinterpret_cast<void**>(dst), v.size() * sizeof(Data64)), "cudaMalloc table");
        check(cudaMemcpy(*dst, v.data(), v.size() * sizeof(Data64), cudaMemcpyHostToDevice),
              "cudaMemcpy table");
    }

    // ---- lane primitives (all on the default stream) ----
    static unsigned threads_for(int lg) { return static_cast<unsigned>(((1 << lg) / 2) < 1024 ? (1 << lg) / 2 : 1024); }

    void transpose(const Data64* src, Data64* dst, int R, int C, int batch) const {
        dim3 blk(32, 8), grd((C + 31) / 32, (R + 31) / 32, batch);
        transpose_kernel<<<grd, blk>>>(src, dst, R, C);
        check(cudaGetLastError(), "transpose_kernel");
    }
    // column transform: batch*n2 rows of length n1 (negacyclic, psi1)
    void k1_fwd(Data64* y, int batch) const {
        fwd_rows<false><<<static_cast<unsigned>(batch) * n2, threads_for(log1), n1 * sizeof(Data64)>>>(
            y, psi1_rev, nullptr, mod, log1, n2);
        check(cudaGetLastError(), "fwd_rows K1");
    }
    // row transform: batch*n1 rows of length n2 with the fused twiddle
    void k2_fwd(Data64* x, int batch) const {
        fwd_rows<true><<<static_cast<unsigned>(batch) * n1, threads_for(log2), n2 * sizeof(Data64)>>>(
            x, psi2_rev, tw, mod, log2, n1);
        check(cudaGetLastError(), "fwd_rows K2");
    }
    void k3_inv(Data64* x, int batch) const {
        inv_rows<true><<<static_cast<unsigned>(batch) * n1, threads_for(log2), n2 * sizeof(Data64)>>>(
            x, psi2inv_rev, twi, mod, log2, n1);
        check(cudaGetLastError(), "inv_rows K3");
    }
    void k4_inv(Data64* y, int batch) const {
        inv_rows<false><<<static_cast<unsigned>(batch) * n2, threads_for(log1), n1 * sizeof(Data64)>>>(
            y, psi1inv_rev, nullptr, mod, log1, n2);
        check(cudaGetLastError(), "inv_rows K4");
    }
    void pointwise(Data64* a, const Data64* b, int batch) const {
        const size_t total = static_cast<size_t>(batch) * N;
        pointwise_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(a, b, mod, total);
        check(cudaGetLastError(), "pointwise_kernel");
    }

    // ---- whole poly-mul: out (in a) = a * b mod (X^N + 1), batch polys ----
    // lanes: 1 = SM kernels, 2 = DRU transposes, 3 = both (functional).
    // Running one lane alone is for timing only (its output is not a product).
    void multiply(Data64* a, Data64* b, Data64* ta, Data64* tb, int batch, int lanes = 3) const {
        const bool sm = lanes & 1, dru = lanes & 2;
        if (dru) transpose(a, ta, n1, n2, batch);   // T1(a)
        if (sm)  k1_fwd(ta, batch);                  // K1(a)
        if (dru) transpose(ta, a, n2, n1, batch);   // T2(a)
        if (sm)  k2_fwd(a, batch);                   // K2(a)
        if (dru) transpose(b, tb, n1, n2, batch);   // T1(b)
        if (sm)  k1_fwd(tb, batch);                  // K1(b)
        if (dru) transpose(tb, b, n2, n1, batch);   // T2(b)
        if (sm)  k2_fwd(b, batch);                   // K2(b)
        if (sm)  pointwise(a, b, batch);             // PW
        if (sm)  k3_inv(a, batch);                   // K3
        if (dru) transpose(a, ta, n1, n2, batch);   // T4
        if (sm)  k4_inv(ta, batch);                  // K4
        if (dru) transpose(ta, a, n2, n1, batch);   // T5 -> natural order in a
    }
};

}  // namespace fused4

#endif  // GPU_BASELINE_FUSED4_NTT_CUH__
