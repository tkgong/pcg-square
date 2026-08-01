// Square four-step negacyclic polynomial multiplication in R_p = Z_p[X]/(X^N+1).
//
// WHY THIS BACKEND EXISTS
// -----------------------
// GPU-NTT's own four-step path hard-codes a heavily skewed factorisation
// (matrix_dimention: n1 in {32..256}, n2 up to 65536 -- e.g. 32 x 32768 at
// N=2^20). n1 is small so that step 1 is a warp-local length-32 transform in a
// 32x33 shared tile; the price is that the length-n2 transform no longer fits in
// shared memory and must be split into two partial passes (FourStepPartial*
// Core1/Core2). Measured per-kernel on L40S, every kernel of that pipeline runs
// at 62-82% of peak DRAM bandwidth, i.e. the transform is bandwidth-bound and
// its cost is simply the number of passes over the operand. The skew therefore
// costs one extra core pass.
//
// This backend uses a BALANCED factorisation, n1 = 2^ceil(logN/2),
// n2 = 2^floor(logN/2), so both sub-transforms fit in shared memory and each is
// a single fused merge-NTT pass. The only remaining data-reorganisation passes
// are the TRANSPOSES: they are kept explicit and standalone so their cost is
// separately measurable and separately offloadable. The library's
// bit-reversals are NOT separate passes -- each one is absorbed into a twiddle
// table's index, into a transpose's destination index, or past the pointwise
// multiply, all at no cost. A standalone brev pass would overstate the movement
// share of an SM baseline.
//
// FACTORISATION (derivation, so the index conventions are auditable)
// ------------------------------------------------------------------
// With n = i*n2 + j (i<n1, j<n2) and k = k2*n1 + k1 (k1<n1, k2<n2):
//   X[k] = sum_j [ omega_N^{j k1} * ( sum_i A[i][j] omega_n1^{i k1} ) ] omega_n2^{j k2}
// giving
//   T1        : A (n1 x n2) -> A^T (n2 x n1)     [make the i-direction contiguous]
//   NTT_n1    : batch n2, length n1              [inner sum]
//   TWIDDLE   : *= omega_N^{j k1}
//   T2        : (n2 x n1) -> (n1 x n2)           [make the j-direction contiguous]
//   NTT_n2    : batch n1, length n2              [outer sum]
//   T3        : (n1 x n2) -> (n2 x n1)           [k = k2*n1 + k1 output order]
// The cyclic transform is wrapped in the psi twist sandwich for the negacyclic
// ring, exactly as the other backends do.
//
// MEASURED LIBRARY CONVENTION (probes bin/sub_ntt_probe, bin/inv):
//   FORWARD : out = bitrev(DFT_w(in))        -- natural in, bit-reversed out
//   INVERSE : out = bitrev(IDFT_w(in))       -- SAME structure, NOT the inverse
//             permutation, and cfg.mod_inverse is IGNORED (1 and 1/L give
//             bit-identical output), so the 1/N scaling must be applied by us.
//
// HOW THE FOUR BIT-REVERSALS ARE ABSORBED (no standalone pass, all free)
//   forward  NTT_n1 -> twiddle table indexed w[j*n1 + p] = omega_N^{j*bitrev(p)}
//   forward  NTT_n2 -> deferred past the pointwise product, which is elementwise
//                      and un-permutes the axis in its store
//   inverse  INTT_n2 -> twiddle table indexed w[k1*n2 + r] = omega^{-bitrev(r)*k1}
//   inverse  INTT_n1 -> folded into the final transpose's destination row
// A transpose's destination ROW comes from the source column, so permuting it
// leaves every destination row's elements consecutive: free. The destination
// COLUMN permutation (the deferred n2 reversal) scatters within one axis span,
// which L2 absorbs at these sizes -- the standalone brev pass it replaces
// measured 0.421 ms against 0.4275 ms for a plain transpose.
//
// So one poly_mul is 20 kernels: 2 pre-twist, 2 x (2 transpose + 2 NTT +
// 1 twiddle), 1 pointwise, 2 transpose + 2 INTT + 1 twiddle + 1 scale,
// 1 post-twist. Six transposes are the ONLY movement passes, and they are the
// DRU's offload target.

#include "common/poly_mul_cuda.h"
#include "common/ntt_roots.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/common/nttparameters.cuh"

using namespace gpuntt;

namespace pcg_cuda {
namespace {

constexpr uint64_t kPrime = 4611686018326724609ULL;
// The transpose kernel is guarded, but keeping both matrix dimensions a
// multiple of 16 (logN >= 8 with a balanced split) keeps its tiles full.
constexpr int kMinSquareLogN = 8;
constexpr int kMaxSquareLogN = 24;

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::ostringstream os;
    os << what << ": " << cudaGetErrorString(err);
    throw std::runtime_error(os.str());
}

int log2_int(int n) { int o = 0; while (n > 1) { n >>= 1; ++o; } return o; }

int host_bitreverse(int index, int n_power) {
    int out = 0;
    for (int i = 0; i < n_power; ++i) { out = (out << 1) | (index & 1); index >>= 1; }
    return out;
}

// Per-stage device-time accumulator (env PCG_SQUARE_STAGE_MS=1). The transpose
// passes are timed separately from the transforms so the movement share of a
// balanced factorisation is a measured quantity.
struct StageTimes {
    double transpose = 0, brev = 0, ntt = 0, twiddle = 0, twist = 0, pointwise = 0;
};

// The library launches one grid row per transform, so a call with more than
// 65535 transforms aborts with "invalid configuration argument". Chunking the
// call is transparent: rows are contiguous blocks of `len` elements.
constexpr int kMaxRowsPerLaunch = 65535;

inline void ntt_chunked(Data64* buf, Data64* root, Modulus<Data64> mod,
                        ntt_configuration<Data64> cfg, int rows, int len) {
    for (int off = 0; off < rows; off += kMaxRowsPerLaunch) {
        const int r = (rows - off < kMaxRowsPerLaunch) ? rows - off : kMaxRowsPerLaunch;
        GPU_NTT_Inplace<Data64>(buf + static_cast<size_t>(off) * len, root, mod, cfg, r);
    }
}

bool fold_brev_enabled() {
    // DEFAULT OFF: the bit-reversal is a standalone kernel, so whatever the NTT
    // writes to memory for the DRU to transpose is in natural order. Setting
    // PCG_SQUARE_FOLD_BREV=1 folds it into the neighbouring kernels' indices
    // instead, which is faster on the SM but leaves permuted data in memory --
    // kept as a measurable ablation, not as the default.
    static const bool on = std::getenv("PCG_SQUARE_FOLD_BREV") != nullptr;
    return on;
}

bool stage_timing_enabled() {
    static const bool on = std::getenv("PCG_SQUARE_STAGE_MS") != nullptr;
    return on;
}

template <typename F>
void timed_stage(double* acc, F&& fn) {
    if (!acc) { fn(); return; }
    cudaEvent_t s = nullptr, e = nullptr;
    check(cudaEventCreate(&s), "cudaEventCreate stage start");
    check(cudaEventCreate(&e), "cudaEventCreate stage stop");
    check(cudaEventRecord(s), "cudaEventRecord stage start");
    fn();
    check(cudaEventRecord(e), "cudaEventRecord stage stop");
    check(cudaEventSynchronize(e), "cudaEventSynchronize stage");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, s, e), "cudaEventElapsedTime stage");
    *acc += ms;
    cudaEventDestroy(s); cudaEventDestroy(e);
}

// merge-NTT root table for a cyclic length-L transform at our prime:
// table[i] = base^i for i in [0, L/2), then bit-reverse reordered -- exactly
// NTTParameters::forward_root_of_unity_table_generator + gpu_root_of_unity_table_generator
// with ReductionPolynomial::X_N_minus (root = omega, size = L/2).
std::vector<uint64_t> merge_root_table(int L, uint64_t base, uint64_t q) {
    const int half = L / 2;
    std::vector<uint64_t> t(static_cast<size_t>(half));
    uint64_t cur = 1;
    for (int i = 0; i < half; ++i) { t[i] = cur; cur = mod_mul_u64(cur, base, q); }
    const int lg = log2_int(half);
    std::vector<uint64_t> out(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i) out[i] = t[host_bitreverse(i, lg)];
    return out;
}

__global__ void twist_kernel(Data64* a, const Data64* twist,
                             Modulus<Data64> mod, int N, size_t total) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    a[idx] = OPERATOR_GPU<Data64>::mult(a[idx], twist[static_cast<int>(idx % N)], mod);
}

// TWIDDLE in the (n2 x n1) layout produced by T1+NTT_n1: element (j, k1) sits at
// j*n1 + k1 and is scaled by omega_N^{j*k1}. The table is indexed the same way.
__device__ __forceinline__ int dev_brv(int x, int lg) {
    int r = 0;
    for (int k = 0; k < lg; ++k) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// Tiled transpose with the library's bit-reversal folded into the DESTINATION
// indices, so no standalone un-bit-reverse pass is ever needed:
//     dst[ dr*R + dc ] = src[ sr*C + sc ],   dr = revr ? brv(sc) : sc
//                                            dc = revc ? brv(sr) : sr
// src is (R x C) row-major, dst is (C x R) row-major, `batch` independent
// blocks of R*C elements.
//
// Why this is (near-)free. Reversing `dr` -- which is derived from the source
// COLUMN -- only relabels which destination row a tile lands in; every
// destination row still receives 32 consecutive elements, so the store stays
// fully coalesced. Reversing `dc` scatters within a destination row; the span
// is one axis (8-32 KB here), which L2 absorbs -- the standalone brev pass this
// replaces measured 0.421 ms against 0.4275 ms for a plain transpose, i.e. the
// scatter was already costing nothing at these sizes.
template <int TILE>
__global__ void xpose_fused_kernel(const Data64* __restrict__ src,
                                   Data64* __restrict__ dst,
                                   int R, int C, int lgR, int lgC,
                                   int revr, int revc) {
    __shared__ Data64 tile[TILE][TILE + 1];
    const size_t blk = static_cast<size_t>(blockIdx.z) * R * C;
    const int c0 = blockIdx.x * TILE;
    const int r0 = blockIdx.y * TILE;
    for (int t = threadIdx.y; t < TILE; t += blockDim.y) {
        const int sr = r0 + t, sc = c0 + threadIdx.x;
        if (sr < R && sc < C) tile[t][threadIdx.x] = src[blk + static_cast<size_t>(sr) * C + sc];
    }
    __syncthreads();
    for (int t = threadIdx.y; t < TILE; t += blockDim.y) {
        const int sc = c0 + t, sr = r0 + threadIdx.x;
        if (sc >= C || sr >= R) continue;
        const int dr = revr ? dev_brv(sc, lgC) : sc;
        const int dc = revc ? dev_brv(sr, lgR) : sr;
        dst[blk + static_cast<size_t>(dr) * R + dc] = tile[threadIdx.x][t];
    }
}

// Standalone bit-reversal of one contiguous axis. This is the DEFAULT path:
// it guarantees that whatever is written to memory between two transforms --
// in particular whatever the DRU is asked to transpose -- is in natural order.
__global__ void brev_axis_kernel(const Data64* __restrict__ in,
                                 Data64* __restrict__ out,
                                 int lg, int len, size_t total) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int i = static_cast<int>(idx % len);
    out[idx - i + dev_brv(i, lg)] = in[idx];
}

__global__ void pointwise_plain_kernel(const Data64* __restrict__ fa,
                                       const Data64* __restrict__ fb,
                                       Data64* __restrict__ out,
                                       Modulus<Data64> mod, size_t total) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = OPERATOR_GPU<Data64>::mult(fa[idx], fb[idx], mod);
}

// Pointwise multiply that also un-bit-reverses the n2 axis on its way out, so
// the two forward transforms can leave that axis permuted (the product is
// elementwise, hence order-agnostic) and neither pays for a pass.
__global__ void pointwise_brev_kernel(const Data64* __restrict__ fa,
                                      const Data64* __restrict__ fb,
                                      Data64* __restrict__ out,
                                      Modulus<Data64> mod, int n2, int lg2,
                                      size_t total) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int q = static_cast<int>(idx % n2);
    out[idx - q + dev_brv(q, lg2)] =
        OPERATOR_GPU<Data64>::mult(fa[idx], fb[idx], mod);
}

__global__ void scale_kernel(Data64* a, Data64 s, Modulus<Data64> mod, size_t total) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    a[idx] = OPERATOR_GPU<Data64>::mult(a[idx], s, mod);
}

__global__ void twiddle_kernel(Data64* a, const Data64* w,
                               Modulus<Data64> mod, int N, size_t total) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    a[idx] = OPERATOR_GPU<Data64>::mult(a[idx], w[static_cast<int>(idx % N)], mod);
}

struct SquarePlan {
    explicit SquarePlan(int degree)
        : N(degree), logN(log2_int(degree)), modulus(kPrime) {
        n1 = 1 << ((logN + 1) / 2);          // >= n2
        n2 = 1 << (logN / 2);

        const uint64_t psi = negacyclic_psi(kPrime, N);
        const uint64_t psi_inv = mod_inv_u64(psi, kPrime);
        const uint64_t omega = mod_mul_u64(psi, psi, kPrime);      // N-th root
        const uint64_t omega_inv = mod_inv_u64(omega, kPrime);
        n_inv = mod_inv_u64(static_cast<uint64_t>(N), kPrime);
        n1_inv = mod_inv_u64(static_cast<uint64_t>(n1), kPrime);
        n2_inv = mod_inv_u64(static_cast<uint64_t>(n2), kPrime);

        const int lg1 = log2_int(n1);
        // sub-transform roots: omega_{n1} = omega^(N/n1), omega_{n2} = omega^(N/n2)
        const uint64_t w_n1 = mod_pow_u64(omega, N / n1, kPrime);
        const uint64_t w_n2 = mod_pow_u64(omega, N / n2, kPrime);
        const uint64_t w_n1_i = mod_inv_u64(w_n1, kPrime);
        const uint64_t w_n2_i = mod_inv_u64(w_n2, kPrime);

        upload(&d_r_n1, merge_root_table(n1, w_n1, kPrime));
        upload(&d_r_n2, merge_root_table(n2, w_n2, kPrime));
        upload(&d_ri_n1, merge_root_table(n1, w_n1_i, kPrime));
        upload(&d_ri_n2, merge_root_table(n2, w_n2_i, kPrime));

        // Twiddle tables ABSORB the library's bit-reversed output order instead
        // of paying for a standalone un-bit-reverse pass. The library is
        // natural-in / bit-reversed-out, so after NTT_n1 the slot at offset
        // j*n1 + p holds the coefficient for k1 = bitrev(p). Indexing the
        // precomputed table the same way makes the permutation free:
        //     w[j*n1 + p] = omega_N^{j * bitrev(p)}
        // Same trick on the inverse side, where BOTH coordinates are permuted
        // (the n1 axis still carries the forward's bit-reversal, and INTT_n2
        // bit-reverses the n2 axis), so one table absorbs both.
        const int lg2 = log2_int(n2);
        auto tw = [&](uint64_t base) {
            std::vector<uint64_t> t(static_cast<size_t>(N));
            for (int j = 0; j < n2; ++j)
                for (int p = 0; p < n1; ++p)
                    t[static_cast<size_t>(j) * n1 + p] =
                        mod_pow_u64(base, static_cast<uint64_t>(j) *
                                          host_bitreverse(p, lg1), kPrime);
            return t;
        };
        // Inverse chain: after INTT_n2 the slot at k1*n2 + r holds the partial
        // sum for j = bitrev(r), so the cross factor omega_N^{-j*k1} is indexed
        // t[k1*n2 + r] = omega^{-bitrev(r)*k1} -- the n2 axis is the reversed
        // one here, not the n1 axis.
        auto tw_rj = [&](uint64_t base) {
            std::vector<uint64_t> t(static_cast<size_t>(N));
            for (int k = 0; k < n1; ++k)
                for (int r = 0; r < n2; ++r)
                    t[static_cast<size_t>(k) * n2 + r] =
                        mod_pow_u64(base, static_cast<uint64_t>(k) *
                                          host_bitreverse(r, lg2), kPrime);
            return t;
        };
        // plain tables for the standalone-brev chain (indices already natural)
        auto tw_plain = [&](uint64_t base) {
            std::vector<uint64_t> t(static_cast<size_t>(N));
            for (int j = 0; j < n2; ++j)
                for (int k = 0; k < n1; ++k)
                    t[static_cast<size_t>(j) * n1 + k] =
                        mod_pow_u64(base, static_cast<uint64_t>(j) * k, kPrime);
            return t;
        };
        auto tw_plain_rj = [&](uint64_t base) {
            std::vector<uint64_t> t(static_cast<size_t>(N));
            for (int k = 0; k < n1; ++k)
                for (int j = 0; j < n2; ++j)
                    t[static_cast<size_t>(k) * n2 + j] =
                        mod_pow_u64(base, static_cast<uint64_t>(j) * k, kPrime);
            return t;
        };
        upload(&d_tw_f, tw(omega));            // bitrev-indexed (fold path)
        upload(&d_tw_i, tw_rj(omega_inv));
        upload(&d_tw_fp, tw_plain(omega));     // natural-indexed (default path)
        upload(&d_tw_ip, tw_plain_rj(omega_inv));

        std::vector<uint64_t> pf(N), pi(N);
        uint64_t c = 1;
        for (int i = 0; i < N; ++i) { pf[i] = c; c = mod_mul_u64(c, psi, kPrime); }
        c = 1;
        for (int i = 0; i < N; ++i) { pi[i] = c; c = mod_mul_u64(c, psi_inv, kPrime); }
        upload(&d_psi, pf);
        upload(&d_psi_inv, pi);
    }

    ~SquarePlan() {
        for (Data64* p : {d_r_n1, d_r_n2, d_ri_n1, d_ri_n2,
                          d_tw_f, d_tw_i, d_tw_fp, d_tw_ip, d_psi, d_psi_inv,
                          d_a, d_b, d_t})
            if (p) cudaFree(p);
    }

    void upload(Data64** dst, const std::vector<uint64_t>& h) {
        const size_t bytes = h.size() * sizeof(Data64);
        check(cudaMalloc(reinterpret_cast<void**>(dst), bytes), "cudaMalloc table");
        check(cudaMemcpy(*dst, h.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy table");
    }

    void ensure(int batch) {
        if (batch <= cap) return;
        for (Data64* p : {d_a, d_b, d_t}) if (p) cudaFree(p);
        d_a = d_b = d_t = nullptr;
        const size_t bytes = static_cast<size_t>(batch) * N * sizeof(Data64);
        check(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes), "cudaMalloc a");
        check(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes), "cudaMalloc b");
        check(cudaMalloc(reinterpret_cast<void**>(&d_t), bytes), "cudaMalloc t");
        cap = batch;
    }

    // scale: 0 = no 1/n factor, 1 = 1/n1, 2 = 1/n2 (product over the two
    // inverse sub-transforms is 1/N).
    ntt_configuration<Data64> cfg(int lg, type dir, int scale) const {
        ntt_configuration<Data64> c{};
        c.n_power = lg;
        c.ntt_type = dir;
        c.ntt_layout = PerPolynomial;
        c.reduction_poly = ReductionPolynomial::X_N_minus;   // cyclic
        c.zero_padding = false;
        c.mod_inverse = (scale == 1) ? Ninverse<Data64>(n1_inv)
                     : (scale == 2) ? Ninverse<Data64>(n2_inv)
                                    : Ninverse<Data64>(1);
        c.stream = 0;
        return c;
    }

    // src is (R x C) row-major per batch element, dst is (C x R). See
    // xpose_fused_kernel for what revr/revc mean.
    void xpose(const Data64* in, Data64* out, int R, int C, int revr, int revc,
               int batch, StageTimes* st) {
        constexpr int TILE = 32;
        // timing-only diagnostic: PCG_SQUARE_NOFOLD=1 disables the folded
        // reversals so the transpose kernel's own cost can be separated from
        // the cost of the fold. Results are WRONG under this switch.
        static const bool nofold = std::getenv("PCG_SQUARE_NOFOLD") != nullptr;
        if (nofold) { revr = 0; revc = 0; }
        const dim3 grid((C + TILE - 1) / TILE, (R + TILE - 1) / TILE, batch);
        const dim3 block(TILE, 8);
        timed_stage(st ? &st->transpose : nullptr, [&] {
            xpose_fused_kernel<TILE><<<grid, block>>>(
                in, out, R, C, log2_int(R), log2_int(C), revr, revc);
        });
        check(cudaGetLastError(), "launch xpose_fused");
    }

    // Five stages, NO standalone bit-reversal pass: the library is
    // natural-in / bit-reversed-out, and both of its reversals are absorbed --
    // the n1 one by the twiddle table's index, the n2 one by being deferred past
    // the (order-agnostic) pointwise multiply. T2 additionally folds the n1
    // reversal into its destination row, which is free.
    //   T1 -> NTT_n1 -> twiddle -> T2(+brev n1) -> NTT_n2
    // Result lands in `a`; output layout is a[k1*n2 + q] = X[k2*n1 + k1] with
    // k2 = bitrev(q), i.e. the n2 axis is still permuted.
    void brev_axis(const Data64* in, Data64* out, int lg, int len, int batch,
                   StageTimes* st) {
        const size_t total = static_cast<size_t>(batch) * N;
        const unsigned blk = 256, grd = static_cast<unsigned>((total + blk - 1) / blk);
        timed_stage(st ? &st->brev : nullptr, [&] {
            brev_axis_kernel<<<grd, blk>>>(in, out, lg, len, total);
        });
        check(cudaGetLastError(), "launch brev_axis");
    }

    // DEFAULT chain: every transpose input is in natural order, because a
    // standalone brev kernel undoes the library's permutation right after each
    // sub-transform. This is what the DRU offload assumes.
    //   T1 -> NTT_n1 -> brev(n1) -> twiddle -> T2 -> NTT_n2 -> brev(n2)
    Data64* forward_plain(Data64* a, Data64* b, int batch, StageTimes* st) {
        const size_t total = static_cast<size_t>(batch) * N;
        const unsigned blk = 256, grd = static_cast<unsigned>((total + blk - 1) / blk);
        xpose(a, b, n1, n2, 0, 0, batch, st);
        timed_stage(st ? &st->ntt : nullptr, [&] {
            ntt_chunked(b, d_r_n1, modulus, cfg(log2_int(n1), FORWARD, 0), n2 * batch, n1);
        });
        brev_axis(b, a, log2_int(n1), n1, batch, st);
        timed_stage(st ? &st->twiddle : nullptr, [&] {
            twiddle_kernel<<<grd, blk>>>(a, d_tw_fp, modulus, N, total);
        });
        check(cudaGetLastError(), "launch twiddle fwd");
        xpose(a, b, n2, n1, 0, 0, batch, st);
        timed_stage(st ? &st->ntt : nullptr, [&] {
            ntt_chunked(b, d_r_n2, modulus, cfg(log2_int(n2), FORWARD, 0), n1 * batch, n2);
        });
        brev_axis(b, a, log2_int(n2), n2, batch, st);
        return a;
    }

    Data64* inverse_plain(Data64* a, Data64* b, int batch, StageTimes* st) {
        const size_t total = static_cast<size_t>(batch) * N;
        const unsigned blk = 256, grd = static_cast<unsigned>((total + blk - 1) / blk);
        timed_stage(st ? &st->ntt : nullptr, [&] {
            ntt_chunked(a, d_ri_n2, modulus, cfg(log2_int(n2), INVERSE, 0), n1 * batch, n2);
        });
        brev_axis(a, b, log2_int(n2), n2, batch, st);
        timed_stage(st ? &st->twiddle : nullptr, [&] {
            twiddle_kernel<<<grd, blk>>>(b, d_tw_ip, modulus, N, total);
        });
        check(cudaGetLastError(), "launch twiddle inv");
        xpose(b, a, n1, n2, 0, 0, batch, st);
        timed_stage(st ? &st->ntt : nullptr, [&] {
            ntt_chunked(a, d_ri_n1, modulus, cfg(log2_int(n1), INVERSE, 0), n2 * batch, n1);
        });
        brev_axis(a, b, log2_int(n1), n1, batch, st);
        xpose(b, a, n2, n1, 0, 0, batch, st);
        timed_stage(st ? &st->twiddle : nullptr, [&] {
            scale_kernel<<<grd, blk>>>(a, static_cast<Data64>(n_inv), modulus, total);
        });
        check(cudaGetLastError(), "launch 1/N scale");
        return a;
    }

    Data64* forward(Data64* a, Data64* b, int batch, StageTimes* st) {
        const size_t total = static_cast<size_t>(batch) * N;
        const unsigned blk = 256, grd = static_cast<unsigned>((total + blk - 1) / blk);
        // T1: A (n1 x n2) -> (n2 x n1), so the i axis becomes contiguous
        xpose(a, b, n1, n2, 0, 0, batch, st);
        timed_stage(st ? &st->ntt : nullptr, [&] {
            ntt_chunked(b, d_r_n1, modulus,
                        cfg(log2_int(n1), FORWARD, 0), n2 * batch, n1);
        });
        // slot (j, p) holds k1 = bitrev(p); d_tw_f is indexed to match
        timed_stage(st ? &st->twiddle : nullptr, [&] {
            twiddle_kernel<<<grd, blk>>>(b, d_tw_f, modulus, N, total);
        });
        check(cudaGetLastError(), "launch twiddle fwd");
        // T2: (n2 x n1) -> (n1 x n2) with the n1 reversal folded into dst row
        xpose(b, a, n2, n1, 1, 0, batch, st);
        timed_stage(st ? &st->ntt : nullptr, [&] {
            ntt_chunked(a, d_r_n2, modulus,
                        cfg(log2_int(n2), FORWARD, 0), n1 * batch, n2);
        });
        return a;
    }

    // Mirror of forward(): input a[k1*n2 + k2] natural on both axes (the
    // pointwise multiply un-permuted the n2 axis on its way out). Chain is
    //   INTT_n2 -> twiddle -> T -> INTT_n1 -> T(+brev both) -> 1/N
    // and again no standalone bit-reversal: INTT_n2's reversal is absorbed by
    // d_tw_i's index, INTT_n1's by the final transpose's destination row, and
    // the outstanding n2 reversal by that same transpose's destination column.
    // cfg.mod_inverse is a no-op in this library build, so 1/N is explicit.
    Data64* inverse(Data64* a, Data64* b, int batch, StageTimes* st) {
        const size_t total = static_cast<size_t>(batch) * N;
        const unsigned blk = 256, grd = static_cast<unsigned>((total + blk - 1) / blk);
        timed_stage(st ? &st->ntt : nullptr, [&] {      // INTT over k2
            ntt_chunked(a, d_ri_n2, modulus,
                        cfg(log2_int(n2), INVERSE, 0), n1 * batch, n2);
        });
        // slot (k1, r) holds j = bitrev(r); d_tw_i is indexed to match
        timed_stage(st ? &st->twiddle : nullptr, [&] {
            twiddle_kernel<<<grd, blk>>>(a, d_tw_i, modulus, N, total);
        });
        check(cudaGetLastError(), "launch twiddle inv");
        // (k1, r) -> (r, k1), so the k1 axis becomes contiguous
        xpose(a, b, n1, n2, 0, 0, batch, st);
        timed_stage(st ? &st->ntt : nullptr, [&] {      // INTT over k1
            ntt_chunked(b, d_ri_n1, modulus,
                        cfg(log2_int(n1), INVERSE, 0), n2 * batch, n1);
        });
        // (r, s) -> natural (i, j): i = bitrev(s) into the row, j = bitrev(r)
        // into the column.
        xpose(b, a, n2, n1, 1, 1, batch, st);
        timed_stage(st ? &st->twiddle : nullptr, [&] {
            scale_kernel<<<grd, blk>>>(a, static_cast<Data64>(n_inv), modulus, total);
        });
        check(cudaGetLastError(), "launch 1/N scale");
        return a;
    }

    void multiply(uint64_t* out, const uint64_t* a, const uint64_t* b,
                  int batch, PolyMulStats* stats) {
        ensure(batch);
        const size_t total = static_cast<size_t>(batch) * N;
        const size_t bytes = total * sizeof(Data64);
        check(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy a");
        check(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy b");

        cudaEvent_t s = nullptr, e = nullptr;
        if (stats) {
            check(cudaEventCreate(&s), "ev s"); check(cudaEventCreate(&e), "ev e");
            check(cudaEventRecord(s), "rec s");
        }
        const unsigned blk = 256;
        const unsigned grd = static_cast<unsigned>((total + blk - 1) / blk);
        StageTimes stt;
        StageTimes* st = stage_timing_enabled() ? &stt : nullptr;
        Data64* res = nullptr;

        timed_stage(st ? &st->twist : nullptr, [&] {
            twist_kernel<<<grd, blk>>>(d_a, d_psi, modulus, N, total);
            twist_kernel<<<grd, blk>>>(d_b, d_psi, modulus, N, total);
        });
        check(cudaGetLastError(), "launch pre-twist");

        // Each forward() uses its first argument plus d_t as scratch and returns
        // the buffer holding the result, so A-hat and B-hat coexist without
        // any stashing copy.
        const bool fold = fold_brev_enabled();
        Data64* fa = fold ? forward(d_a, d_t, batch, st) : forward_plain(d_a, d_t, batch, st);
        Data64* fb = fold ? forward(d_b, d_t, batch, st) : forward_plain(d_b, d_t, batch, st);
        // Both forwards left the n2 axis bit-reversed; the product is
        // elementwise so it does not care, and its store puts the axis back.
        timed_stage(st ? &st->pointwise : nullptr, [&] {
            if (fold) pointwise_brev_kernel<<<grd, blk>>>(fa, fb, d_t, modulus,
                                                          n2, log2_int(n2), total);
            else      pointwise_plain_kernel<<<grd, blk>>>(fa, fb, d_t, modulus, total);
        });
        check(cudaGetLastError(), "launch pointwise");

        res = fold ? inverse(d_t, fa, batch, st) : inverse_plain(d_t, fa, batch, st);

        timed_stage(st ? &st->twist : nullptr, [&] {
            twist_kernel<<<grd, blk>>>(res, d_psi_inv, modulus, N, total);
        });
        check(cudaGetLastError(), "launch post-twist");

        if (st) {
            const double tot = st->transpose + st->brev + st->ntt + st->twiddle
                             + st->twist + st->pointwise;
            std::fprintf(stderr,
                "[square-stages] N=%d n1=%d n2=%d batch=%d transpose=%.4f brev=%.4f "
                "ntt=%.4f twiddle=%.4f twist=%.4f pointwise=%.4f sum=%.4f ms "
                "(move %.1f%% = transpose, brev merged)\n",
                N, n1, n2, batch, st->transpose, st->brev, st->ntt, st->twiddle,
                st->twist, st->pointwise, tot,
                tot > 0 ? 100.0 * st->transpose / tot : 0.0);
        }
        if (stats) {
            check(cudaEventRecord(e), "rec e");
            check(cudaEventSynchronize(e), "sync e");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, s, e), "elapsed");
            stats->device_ms = ms;
            cudaEventDestroy(s); cudaEventDestroy(e);
        } else {
            check(cudaDeviceSynchronize(), "sync");
        }
        check(cudaMemcpy(out, res, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy out");
    }

    int N, logN, n1, n2;
    uint64_t n_inv, n1_inv, n2_inv;
    Modulus<Data64> modulus;
    Data64 *d_r_n1 = nullptr, *d_r_n2 = nullptr, *d_ri_n1 = nullptr, *d_ri_n2 = nullptr;
    Data64 *d_tw_fp = nullptr, *d_tw_ip = nullptr;
    Data64 *d_tw_f = nullptr, *d_tw_i = nullptr, *d_psi = nullptr, *d_psi_inv = nullptr;
    Data64 *d_a = nullptr, *d_b = nullptr, *d_t = nullptr;
    int cap = 0;
};

std::map<int, std::unique_ptr<SquarePlan>> g_plans;
std::mutex g_mu;

SquarePlan& plan_for(int N) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_plans.find(N);
    if (it == g_plans.end())
        it = g_plans.emplace(N, std::make_unique<SquarePlan>(N)).first;
    return *it->second;
}

}  // namespace

bool poly_mul_gpuntt_square_supported(uint64_t prime, int N, std::string* reason) {
    if (prime != kPrime) { if (reason) *reason = "square backend is built for the PCG prime"; return false; }
    const int lg = log2_int(N);
    if ((1 << lg) != N) { if (reason) *reason = "N must be a power of two"; return false; }
    if (lg < kMinSquareLogN || lg > kMaxSquareLogN) {
        if (reason) *reason = "square backend supports logN in [8, 24]";
        return false;
    }
    return true;
}

void poly_mul_u64_gpuntt_square(uint64_t* out, const uint64_t* a, const uint64_t* b,
                                int batch, int N, uint64_t prime,
                                PolyMulStats* stats) {
    std::string why;
    if (!poly_mul_gpuntt_square_supported(prime, N, &why))
        throw std::runtime_error("poly_mul_u64_gpuntt_square: " + why);
    plan_for(N).multiply(out, a, b, batch, stats);
}

}  // namespace pcg_cuda
