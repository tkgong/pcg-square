// Device DPF output-layer kernels (see leaf_convert_cuda.h for the contract).
// Beaver-corrected scheme: per leaf one ChaCha8 out-hash + one mod-P reduce +
// one conditional add — no modular multiply on the device (the CW = A*B
// Beaver multiply happens once per instance on the host).
//
// Bit-exactness vs the CPU path (half_tree_dpf::ChaCha8OutHash + the host
// loops in pcg_ole_impl.h): identical keystream (same key/nonce/counter word
// placement and little-endian word->byte assembly as chacha8.h), and all
// cross-thread accumulation is mod-P addition (associative + commutative),
// so reduction order cannot change the result.

#include "common/leaf_convert_cuda.h"

#include <cuda_runtime.h>

#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace pcg_cuda {
namespace {

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::ostringstream os;
    os << what << ": " << cudaGetErrorString(err);
    throw std::runtime_error(os.str());
}

// Grow-only buffer pool (same pattern as dpf_gpu.cu / the poly-mul caches):
// called c*c times per gen_and_expand with identical shapes.
struct LeafGpuBuffers {
    uint64_t* partials = nullptr;  // 2 * B * blocks_per_inst (C-lane, T-lane)
    uint64_t* sums = nullptr;      // 2 * B (sumC, sumT)
    uint64_t* cws = nullptr;       // B
    uint64_t* g = nullptr;         // N residues
    size_t cap_partials = 0, cap_b = 0, cap_g = 0;

    static void ensure(uint64_t*& ptr, size_t& cap, size_t need,
                       const char* what) {
        if (need <= cap) return;
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        cap = 0;
        check(cudaMalloc(&ptr, need * sizeof(uint64_t)), what);
        cap = need;
    }
};

std::mutex& leaf_gpu_mutex() {
    static std::mutex m;
    return m;
}

LeafGpuBuffers& leaf_gpu_buffers() {
    static LeafGpuBuffers bufs;
    return bufs;
}

__device__ __forceinline__ uint64_t modadd(uint64_t a, uint64_t b,
                                           uint64_t P) {
    // a, b < P < 2^62 — no uint64 overflow.
    uint64_t s = a + b;
    return s >= P ? s - P : s;
}

__device__ __forceinline__ void atomic_modadd(uint64_t* addr, uint64_t val,
                                              uint64_t P) {
    auto* a = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        modadd(static_cast<uint64_t>(assumed), val, P));
    } while (old != assumed);
}

// ---- ChaCha8 output-layer hash (device mirror of ChaCha8OutHash) -----------
// One keystream block, key = leaf ‖ 0^128, nonce = kOutHashDomain, ctr = 0.
// Identical to dpf_gpu.cu's chacha8_block_dev except the nonce words x14/x15
// carry the domain constant (the tree PRG runs with nonce = 0). Only the low
// 64 bits (words o0,o1 assembled little-endian) are consumed.

#define CHACHA8_QR_LC(a, b, c, d)                         \
    a += b; d ^= a; d = (d << 16) | (d >> 16);            \
    c += d; b ^= c; b = (b << 12) | (b >> 20);            \
    a += b; d ^= a; d = (d << 8)  | (d >> 24);            \
    c += d; b ^= c; b = (b << 7)  | (b >> 25);

__device__ __forceinline__ uint64_t out_hash_lo64(const DpfBlk& leaf) {
    const uint32_t k0 = static_cast<uint32_t>(leaf.lo);
    const uint32_t k1 = static_cast<uint32_t>(leaf.lo >> 32);
    const uint32_t k2 = static_cast<uint32_t>(leaf.hi);
    const uint32_t k3 = static_cast<uint32_t>(leaf.hi >> 32);
    const uint32_t n0 = static_cast<uint32_t>(kOutHashDomain);
    const uint32_t n1 = static_cast<uint32_t>(kOutHashDomain >> 32);

    const uint32_t s0 = 0x61707865u, s1 = 0x3320646eu;
    const uint32_t s2 = 0x79622d32u, s3 = 0x6b206574u;
    uint32_t x0 = s0, x1 = s1, x2 = s2, x3 = s3;
    uint32_t x4 = k0, x5 = k1, x6 = k2, x7 = k3;
    uint32_t x8 = 0, x9 = 0, x10 = 0, x11 = 0;
    uint32_t x12 = 0, x13 = 0, x14 = n0, x15 = n1;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        CHACHA8_QR_LC(x0, x4, x8,  x12)
        CHACHA8_QR_LC(x1, x5, x9,  x13)
        CHACHA8_QR_LC(x2, x6, x10, x14)
        CHACHA8_QR_LC(x3, x7, x11, x15)
        CHACHA8_QR_LC(x0, x5, x10, x15)
        CHACHA8_QR_LC(x1, x6, x11, x12)
        CHACHA8_QR_LC(x2, x7, x8,  x13)
        CHACHA8_QR_LC(x3, x4, x9,  x14)
    }

    const uint32_t o0 = x0 + s0;
    const uint32_t o1 = x1 + s1;
    return static_cast<uint64_t>(o0) | (static_cast<uint64_t>(o1) << 32);
}

#undef CHACHA8_QR_LC

// C = low64(H'(leaf)) mod P (sign-free; the host applies party signs).
__device__ __forceinline__ uint64_t out_C(const DpfBlk& leaf, uint64_t P) {
    return out_hash_lo64(leaf) % P;
}

// (1) Per-instance raw sums: grid = (blocks_per_inst, B). Each block reduces
// a strided slice of instance b's D leaves in shared memory and writes one
// partial per lane: lane 0 = Sigma C (mod P), lane 1 = Sigma tau (plain).
__global__ void out_sum_kernel(const DpfBlk* __restrict__ leaves,
                               uint64_t* __restrict__ partials,
                               size_t D, uint64_t P, int nparts) {
    extern __shared__ uint64_t s_red[];   // [2 * blockDim.x]
    uint64_t* s_c = s_red;
    uint64_t* s_t = s_red + blockDim.x;
    const int b = blockIdx.y;
    const size_t base = static_cast<size_t>(b) * D;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    uint64_t accC = 0, accT = 0;
    for (size_t d = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         d < D; d += stride) {
        const DpfBlk leaf = leaves[base + d];
        accC = modadd(accC, out_C(leaf, P), P);
        accT += leaf.lo & 1;
    }

    s_c[threadIdx.x] = accC;
    s_t[threadIdx.x] = accT;
    __syncthreads();
    for (unsigned off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            s_c[threadIdx.x] = modadd(s_c[threadIdx.x],
                                      s_c[threadIdx.x + off], P);
            s_t[threadIdx.x] += s_t[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const size_t slot = static_cast<size_t>(b) * gridDim.x + blockIdx.x;
        partials[slot] = s_c[0];
        partials[static_cast<size_t>(nparts) * gridDim.y + slot] = s_t[0];
    }
}

// (2) Fold each instance's partials: sums[b] = sumC, sums[B + b] = sumT.
__global__ void out_fold_kernel(const uint64_t* __restrict__ partials,
                                uint64_t* __restrict__ sums,
                                int nparts, int B, uint64_t P) {
    extern __shared__ uint64_t s_red[];
    uint64_t* s_c = s_red;
    uint64_t* s_t = s_red + blockDim.x;
    const int b = blockIdx.x;
    const size_t t_off = static_cast<size_t>(nparts) * B;
    uint64_t accC = 0, accT = 0;
    for (int i = threadIdx.x; i < nparts; i += blockDim.x) {
        const size_t slot = static_cast<size_t>(b) * nparts + i;
        accC = modadd(accC, partials[slot], P);
        accT += partials[t_off + slot];
    }
    s_c[threadIdx.x] = accC;
    s_t[threadIdx.x] = accT;
    __syncthreads();
    for (unsigned off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            s_c[threadIdx.x] = modadd(s_c[threadIdx.x],
                                      s_c[threadIdx.x + off], P);
            s_t[threadIdx.x] += s_t[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sums[b] = s_c[0];
        sums[B + b] = s_t[0];
    }
}

// (3) y = C + (tau ? CW : 0); party 1 negates; negacyclic fold into g
// (pos >= N lands negated at pos - N). Overlap across instances (same kk+ll
// bins straddle) is resolved with a CAS-loop modular atomic — modadd is
// associative + commutative mod P, so the result is bit-exact regardless of
// scheduling. H'(leaf) is recomputed here rather than persisted (B*D*8 bytes
// of C would dwarf the recompute cost).
__global__ void out_scatter_kernel(const DpfBlk* __restrict__ leaves,
                                   const uint64_t* __restrict__ cws,
                                   uint64_t* __restrict__ g,
                                   size_t D, int t, int party,
                                   uint64_t P, int N) {
    const int b = blockIdx.y;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const int kk = b / t, ll = b % t;
    const size_t bin_sz = D >> 1;
    const size_t pos_base = static_cast<size_t>(kk + ll) * bin_sz;
    const uint64_t cw = cws[b];
    const size_t base = static_cast<size_t>(b) * D;

    for (size_t d = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         d < D; d += stride) {
        const DpfBlk leaf = leaves[base + d];
        uint64_t y = out_C(leaf, P);
        if (leaf.lo & 1) y = modadd(y, cw, P);
        if (party != 0 && y != 0) y = P - y;
        if (y == 0) continue;
        size_t pos = pos_base + d;                 // < 2N by construction
        if (pos >= static_cast<size_t>(N)) {       // X^N = -1 fold
            pos -= N;
            y = P - y;
        }
        atomic_modadd(&g[pos], y, P);
    }
}


// (1') Single-hash conversion: one pass computes H'(leaf) ONCE, accumulates the
// per-instance sums AND scatters +-C into g (negacyclic fold) while storing tau
// (1 byte/leaf). After the Beaver opens, (3') adds +-tau*CW[b] from the tau bytes.
// g = sum +-(C + tau*CW) = sum +-C + sum +-tau*CW (mod P): bit-exact with the two-pass version.
__global__ void out_sum_scatter_kernel(const DpfBlk* __restrict__ leaves,
                                       uint64_t* __restrict__ partials,
                                       uint8_t* __restrict__ tau,
                                       uint64_t* __restrict__ g,
                                       size_t D, int t, int party, uint64_t P, int N, int nparts) {
    extern __shared__ uint64_t s_red[];
    uint64_t* s_c = s_red; uint64_t* s_t = s_red + blockDim.x;
    const int b = blockIdx.y; const size_t base = static_cast<size_t>(b) * D;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const int kk = b / t, ll = b % t; const size_t pos_base = static_cast<size_t>(kk + ll) * (D >> 1);
    uint64_t accC = 0, accT = 0;
    for (size_t d = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; d < D; d += stride) {
        const DpfBlk leaf = leaves[base + d];
        const uint64_t C = out_C(leaf, P); const uint8_t tb = leaf.lo & 1;
        accC = modadd(accC, C, P); accT += tb; tau[base + d] = tb;
        uint64_t y = C;
        if (party != 0 && y != 0) y = P - y;
        size_t pos = pos_base + d;
        if (pos >= static_cast<size_t>(N)) { pos -= N; if (y != 0) y = P - y; }
        if (y != 0) atomic_modadd(&g[pos], y, P);
    }
    s_c[threadIdx.x] = accC; s_t[threadIdx.x] = accT; __syncthreads();
    for (unsigned off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) { s_c[threadIdx.x] = modadd(s_c[threadIdx.x], s_c[threadIdx.x + off], P); s_t[threadIdx.x] += s_t[threadIdx.x + off]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const size_t slot = static_cast<size_t>(b) * gridDim.x + blockIdx.x;
        partials[slot] = s_c[0]; partials[static_cast<size_t>(nparts) * gridDim.y + slot] = s_t[0];
    }
}
__global__ void out_tau_scatter_kernel(const uint8_t* __restrict__ tau, const uint64_t* __restrict__ cws,
                                       uint64_t* __restrict__ g, size_t D, int t, int party, uint64_t P, int N) {
    const int b = blockIdx.y; const size_t base = static_cast<size_t>(b) * D;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const int kk = b / t, ll = b % t; const size_t pos_base = static_cast<size_t>(kk + ll) * (D >> 1);
    uint64_t cw = cws[b]; if (party != 0 && cw != 0) cw = P - cw;
    if (cw == 0) return;
    for (size_t d = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; d < D; d += stride) {
        if (!tau[base + d]) continue;
        uint64_t y = cw; size_t pos = pos_base + d;
        if (pos >= static_cast<size_t>(N)) { pos -= N; y = P - y; }
        atomic_modadd(&g[pos], y, P);
    }
}

int blocks_per_instance(size_t D, int tpb) {
    // Cap the per-instance grid so partials stay tiny; each thread strides.
    size_t blocks = (D + tpb - 1) / tpb;
    if (blocks > 1024) blocks = 1024;
    return static_cast<int>(blocks);
}

}  // namespace

void dpf_out_sums(const DpfBlk* d_leaves, int B, size_t D, uint64_t prime,
                  uint64_t* sumC_host, uint64_t* sumT_host) {
    if (B <= 0) return;
    std::lock_guard<std::mutex> lock(leaf_gpu_mutex());
    LeafGpuBuffers& bufs = leaf_gpu_buffers();

    const int TPB = 256;
    const int nparts = blocks_per_instance(D, TPB);
    LeafGpuBuffers::ensure(bufs.partials, bufs.cap_partials,
                           2 * static_cast<size_t>(B) * nparts,
                           "cudaMalloc out partials");
    if (static_cast<size_t>(B) > bufs.cap_b) {
        if (bufs.sums) cudaFree(bufs.sums);
        if (bufs.cws) cudaFree(bufs.cws);
        bufs.sums = bufs.cws = nullptr;
        bufs.cap_b = 0;
        check(cudaMalloc(&bufs.sums, 2 * B * sizeof(uint64_t)),
              "cudaMalloc out sums");
        check(cudaMalloc(&bufs.cws, B * sizeof(uint64_t)),
              "cudaMalloc out cws");
        bufs.cap_b = B;
    }

    dim3 grid(nparts, B);
    out_sum_kernel<<<grid, TPB, 2 * TPB * sizeof(uint64_t)>>>(
        d_leaves, bufs.partials, D, prime, nparts);
    check(cudaGetLastError(), "launch out_sum_kernel");
    out_fold_kernel<<<B, TPB, 2 * TPB * sizeof(uint64_t)>>>(
        bufs.partials, bufs.sums, nparts, B, prime);
    check(cudaGetLastError(), "launch out_fold_kernel");

    std::vector<uint64_t> both(2 * static_cast<size_t>(B));
    check(cudaMemcpy(both.data(), bufs.sums, 2 * B * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost), "D2H out sums");
    for (int b = 0; b < B; ++b) {
        sumC_host[b] = both[b];
        sumT_host[b] = both[B + b];
    }
}

void dpf_out_scatter_g(const DpfBlk* d_leaves, int B, size_t D, int t,
                       int party, uint64_t prime, const uint64_t* CW_host,
                       uint64_t* g_out_host, int N) {
    if (B <= 0) return;
    std::lock_guard<std::mutex> lock(leaf_gpu_mutex());
    LeafGpuBuffers& bufs = leaf_gpu_buffers();

    LeafGpuBuffers::ensure(bufs.g, bufs.cap_g, static_cast<size_t>(N),
                           "cudaMalloc out g");
    // cws capacity is guaranteed by the preceding dpf_out_sums call.
    check(cudaMemcpy(bufs.cws, CW_host, B * sizeof(uint64_t),
                     cudaMemcpyHostToDevice), "H2D out cws");
    check(cudaMemset(bufs.g, 0, static_cast<size_t>(N) * sizeof(uint64_t)),
          "memset out g");

    const int TPB = 256;
    dim3 grid(blocks_per_instance(D, TPB), B);
    out_scatter_kernel<<<grid, TPB>>>(d_leaves, bufs.cws, bufs.g, D, t,
                                      party, prime, N);
    check(cudaGetLastError(), "launch out_scatter_kernel");
    check(cudaMemcpy(g_out_host, bufs.g,
                     static_cast<size_t>(N) * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost), "D2H out g");
}


static uint8_t* g_tau = nullptr; static size_t g_tau_cap = 0;
void dpf_out_sums_v2(const DpfBlk* d_leaves, int B, size_t D, int t, int party, uint64_t prime, int N,
                     uint64_t* sumC_host, uint64_t* sumT_host) {
    if (B <= 0) return;
    std::lock_guard<std::mutex> lock(leaf_gpu_mutex());
    LeafGpuBuffers& bufs = leaf_gpu_buffers();
    const int TPB = 256; const int nparts = blocks_per_instance(D, TPB);
    LeafGpuBuffers::ensure(bufs.partials, bufs.cap_partials, 2 * static_cast<size_t>(B) * nparts, "cudaMalloc out partials");
    if (static_cast<size_t>(B) > bufs.cap_b) {
        if (bufs.sums) cudaFree(bufs.sums); if (bufs.cws) cudaFree(bufs.cws);
        bufs.sums = bufs.cws = nullptr; bufs.cap_b = 0;
        check(cudaMalloc(&bufs.sums, 2 * B * sizeof(uint64_t)), "cudaMalloc out sums");
        check(cudaMalloc(&bufs.cws, B * sizeof(uint64_t)), "cudaMalloc out cws"); bufs.cap_b = B;
    }
    LeafGpuBuffers::ensure(bufs.g, bufs.cap_g, static_cast<size_t>(N), "cudaMalloc out g");
    const size_t need = static_cast<size_t>(B) * D;
    if (need > g_tau_cap) { if (g_tau) cudaFree(g_tau); check(cudaMalloc(&g_tau, need), "cudaMalloc tau"); g_tau_cap = need; }
    check(cudaMemset(bufs.g, 0, static_cast<size_t>(N) * sizeof(uint64_t)), "memset out g");
    dim3 grid(nparts, B);
    out_sum_scatter_kernel<<<grid, TPB, 2 * TPB * sizeof(uint64_t)>>>(d_leaves, bufs.partials, g_tau, bufs.g, D, t, party, prime, N, nparts);
    check(cudaGetLastError(), "launch out_sum_scatter_kernel");
    out_fold_kernel<<<B, TPB, 2 * TPB * sizeof(uint64_t)>>>(bufs.partials, bufs.sums, nparts, B, prime);
    check(cudaGetLastError(), "launch out_fold_kernel");
    std::vector<uint64_t> both(2 * static_cast<size_t>(B));
    check(cudaMemcpy(both.data(), bufs.sums, 2 * B * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H out sums");
    for (int b = 0; b < B; ++b) { sumC_host[b] = both[b]; sumT_host[b] = both[B + b]; }
}
void dpf_out_scatter_g_v2(int B, size_t D, int t, int party, uint64_t prime, const uint64_t* CW_host,
                          uint64_t* g_out_host, int N) {
    if (B <= 0) return;
    std::lock_guard<std::mutex> lock(leaf_gpu_mutex());
    LeafGpuBuffers& bufs = leaf_gpu_buffers();
    check(cudaMemcpy(bufs.cws, CW_host, B * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D out cws");
    const int TPB = 256; dim3 grid(blocks_per_instance(D, TPB), B);
    out_tau_scatter_kernel<<<grid, TPB>>>(g_tau, bufs.cws, bufs.g, D, t, party, prime, N);
    check(cudaGetLastError(), "launch out_tau_scatter_kernel");
    check(cudaMemcpy(g_out_host, bufs.g, static_cast<size_t>(N) * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H out g");
}

}  // namespace pcg_cuda
