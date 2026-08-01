// GPU full-domain half-tree DPF evaluation (drop-in for
// HalfTreeDPF::batch_gen_full_eval). Mirrors the CPU per-level loop exactly:
//
//   for each level i (frontier size sz = m^i):
//     (0) seed my_cws[b][k] = DltSft_share[b][i][k]      [device memcpy2D]
//     (1) hash_kernel    : H_k(parent)=AES_k(parent)^parent for all parents,
//                          fused XOR-reduction of each H_k column into
//                          my_cws[b][k] (shared-memory tree per block +
//                          one atomicXor per block).                       [GPU]
//     (2) host exchange   : swap B*(m-1) CW shares with the peer.          [host/net]
//     (3) expand_kernel   : reconstruct CW, emit m children per parent
//                          (recomputes the per-parent fold from `hashed`). [GPU]
//
// All blocks stay resident in device memory across levels (ping-pong frontier
// buffers); only the tiny B*(m-1)-block CW shares cross the host/network.
// The S-box and round keys are staged in __shared__ memory inside hash_kernel
// (the AES inner loop is the hot path).

#include "common/dpf_gpu.h"
#include "common/aes_gpu.cuh"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pcg_cuda {
namespace {

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::ostringstream os;
    os << what << ": " << cudaGetErrorString(err);
    throw std::runtime_error(os.str());
}

// Grow-only device buffer cache (same idea as the poly-mul plan caches): the
// PCG calls this c*c times per gen_and_expand with identical shapes, and
// per-call cudaMalloc/cudaFree otherwise dominates at PCG-sized domains.
struct DpfGpuBuffers {
    DpfBlk* cur = nullptr;
    DpfBlk* nxt = nullptr;
    DpfBlk* hashed = nullptr;
    DpfBlk* dltsft = nullptr;
    DpfBlk* mycws = nullptr;
    DpfBlk* peercws = nullptr;
    uint8_t* rks = nullptr;
    size_t cap_frontier = 0, cap_hashed = 0;
    size_t cap_dltsft = 0, cap_cws = 0, cap_rks = 0;

    template <typename T>
    static void ensure(T*& ptr, size_t& cap, size_t need, const char* what) {
        if (need <= cap) return;
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        cap = 0;
        check(cudaMalloc(&ptr, need * sizeof(T)), what);
        cap = need;
    }
};

std::mutex& dpf_gpu_mutex() {
    static std::mutex m;
    return m;
}

DpfGpuBuffers& dpf_gpu_buffers() {
    static DpfGpuBuffers bufs;
    return bufs;
}

__device__ __forceinline__ DpfBlk bxor(const DpfBlk& a, const DpfBlk& b) {
    DpfBlk r;
    r.lo = a.lo ^ b.lo;
    r.hi = a.hi ^ b.hi;
    return r;
}

// (1) Hash every parent in the frontier and fold each H_k column into
// my_cws[b][k] (which must be pre-seeded with the DltSft share for this
// level). When sz >= blockDim.x every block lies entirely inside one
// instance (sz and blockDim.x are both powers of two), so we tree-reduce in
// shared memory and issue ONE atomicXor per (block, k). For the few early
// levels with sz < blockDim.x the per-thread atomic path is cheap.
__global__ void hash_kernel(const DpfBlk* __restrict__ cur,
                            DpfBlk* __restrict__ hashed,
                            DpfBlk* __restrict__ my_cws,
                            const uint8_t* __restrict__ rks,
                            int B, int sz, int nk,
                            size_t row_cur, size_t row_hashed) {
    extern __shared__ uint8_t smem[];
    uint8_t* s_sbox = smem;                         // 256 B
    uint8_t* s_rks  = smem + 256;                   // nk*176 B
    DpfBlk*  s_red  = reinterpret_cast<DpfBlk*>(smem + 256 + nk * 176);  // TPB blocks

    for (int t = threadIdx.x; t < 256; t += blockDim.x)
        s_sbox[t] = aesdev::kSboxDev[t];
    for (int t = threadIdx.x; t < nk * 176; t += blockDim.x)
        s_rks[t] = rks[t];
    __syncthreads();

    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(B) * sz;
    const bool active = idx < total;
    int b = 0, j = 0;
    DpfBlk parent;
    parent.lo = 0;
    parent.hi = 0;
    if (active) {
        b = static_cast<int>(idx / sz);
        j = static_cast<int>(idx % sz);
        parent = cur[b * row_cur + j];
    }

    const bool segmented = sz >= static_cast<int>(blockDim.x);
    for (int k = 0; k < nk; ++k) {
        DpfBlk h;
        h.lo = 0;
        h.hi = 0;
        if (active) {
            aesdev::ccr_hash_u64(&s_rks[k * 176], s_sbox, parent.lo, parent.hi,
                                 h.lo, h.hi);
            hashed[b * row_hashed + static_cast<size_t>(j) * nk + k] = h;
        }
        if (segmented) {
            // All threads in this block share b (blockDim.x | sz), and total is
            // a multiple of blockDim.x, so every thread is active here.
            s_red[threadIdx.x] = h;
            __syncthreads();
            for (unsigned off = blockDim.x >> 1; off > 0; off >>= 1) {
                if (threadIdx.x < off)
                    s_red[threadIdx.x] = bxor(s_red[threadIdx.x],
                                              s_red[threadIdx.x + off]);
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                DpfBlk* dst = &my_cws[static_cast<size_t>(b) * nk + k];
                atomicXor(reinterpret_cast<unsigned long long*>(&dst->lo),
                          static_cast<unsigned long long>(s_red[0].lo));
                atomicXor(reinterpret_cast<unsigned long long*>(&dst->hi),
                          static_cast<unsigned long long>(s_red[0].hi));
            }
            __syncthreads();
        } else if (active) {
            DpfBlk* dst = &my_cws[static_cast<size_t>(b) * nk + k];
            atomicXor(reinterpret_cast<unsigned long long*>(&dst->lo),
                      static_cast<unsigned long long>(h.lo));
            atomicXor(reinterpret_cast<unsigned long long*>(&dst->hi),
                      static_cast<unsigned long long>(h.hi));
        }
    }
}

// ---- ChaCha8 device tree-PRG -----------------------------------------------
// Bit-exact mirror of ChaCha8TreePRG (half_tree_dpf/tree_prg.h): the parent
// block is the 128-bit key (zero-padded to 256 bits), nonce = 0, and child k
// is the k-th 16-byte chunk of the keystream (counter = k/4). All-register:
// no S-box table, no round keys, no shared-memory staging.

#define CHACHA8_QR_DEV(a, b, c, d)                        \
    a += b; d ^= a; d = (d << 16) | (d >> 16);            \
    c += d; b ^= c; b = (b << 12) | (b >> 20);            \
    a += b; d ^= a; d = (d << 8)  | (d >> 24);            \
    c += d; b ^= c; b = (b << 7)  | (b >> 25);

// One 64-byte keystream block for key = (k0,k1,k2,k3) ‖ 0^128, counter = ctr,
// nonce = 0. out[i] = final little-endian word i (matches store32_le on the
// CPU: assembling uint64s from word pairs reproduces the byte stream).
__device__ __forceinline__ void chacha8_block_dev(uint32_t k0, uint32_t k1,
                                                  uint32_t k2, uint32_t k3,
                                                  uint32_t ctr,
                                                  uint32_t out[16]) {
    const uint32_t s0 = 0x61707865u, s1 = 0x3320646eu;
    const uint32_t s2 = 0x79622d32u, s3 = 0x6b206574u;
    uint32_t x0 = s0, x1 = s1, x2 = s2, x3 = s3;
    uint32_t x4 = k0, x5 = k1, x6 = k2, x7 = k3;
    uint32_t x8 = 0, x9 = 0, x10 = 0, x11 = 0;
    uint32_t x12 = ctr, x13 = 0, x14 = 0, x15 = 0;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        CHACHA8_QR_DEV(x0, x4, x8,  x12)
        CHACHA8_QR_DEV(x1, x5, x9,  x13)
        CHACHA8_QR_DEV(x2, x6, x10, x14)
        CHACHA8_QR_DEV(x3, x7, x11, x15)
        CHACHA8_QR_DEV(x0, x5, x10, x15)
        CHACHA8_QR_DEV(x1, x6, x11, x12)
        CHACHA8_QR_DEV(x2, x7, x8,  x13)
        CHACHA8_QR_DEV(x3, x4, x9,  x14)
    }

    out[0]  = x0  + s0;  out[1]  = x1  + s1;
    out[2]  = x2  + s2;  out[3]  = x3  + s3;
    out[4]  = x4  + k0;  out[5]  = x5  + k1;
    out[6]  = x6  + k2;  out[7]  = x7  + k3;
    out[8]  = x8;        out[9]  = x9;
    out[10] = x10;       out[11] = x11;
    out[12] = x12 + ctr; out[13] = x13;
    out[14] = x14;       out[15] = x15;
}

#undef CHACHA8_QR_DEV

// (1-chacha8) Same contract and reduction skeleton as hash_kernel, but
// H_k(parent) comes from the ChaCha8 keystream instead of the AES CCR hash.
// Shared memory holds only the reduction buffer (TPB blocks).
__global__ void hash_kernel_chacha8(const DpfBlk* __restrict__ cur,
                                    DpfBlk* __restrict__ hashed,
                                    DpfBlk* __restrict__ my_cws,
                                    int B, int sz, int nk,
                                    size_t row_cur, size_t row_hashed) {
    extern __shared__ uint8_t smem[];
    DpfBlk* s_red = reinterpret_cast<DpfBlk*>(smem);

    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(B) * sz;
    const bool active = idx < total;
    int b = 0, j = 0;
    DpfBlk h_all[15];  // nk <= 15 (w <= 4)
    if (active) {
        b = static_cast<int>(idx / sz);
        j = static_cast<int>(idx % sz);
        const DpfBlk parent = cur[b * row_cur + j];
        const uint32_t k0 = static_cast<uint32_t>(parent.lo);
        const uint32_t k1 = static_cast<uint32_t>(parent.lo >> 32);
        const uint32_t k2 = static_cast<uint32_t>(parent.hi);
        const uint32_t k3 = static_cast<uint32_t>(parent.hi >> 32);
        const int nblocks = (nk + 3) / 4;
        for (int blk = 0; blk < nblocks; ++blk) {
            uint32_t o[16];
            chacha8_block_dev(k0, k1, k2, k3, static_cast<uint32_t>(blk), o);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int k = blk * 4 + c;
                if (k < nk) {
                    h_all[k].lo = static_cast<uint64_t>(o[4 * c])
                                | (static_cast<uint64_t>(o[4 * c + 1]) << 32);
                    h_all[k].hi = static_cast<uint64_t>(o[4 * c + 2])
                                | (static_cast<uint64_t>(o[4 * c + 3]) << 32);
                    hashed[b * row_hashed + static_cast<size_t>(j) * nk + k] =
                        h_all[k];
                }
            }
        }
    }

    const bool segmented = sz >= static_cast<int>(blockDim.x);
    for (int k = 0; k < nk; ++k) {
        DpfBlk h;
        h.lo = 0;
        h.hi = 0;
        if (active) h = h_all[k];
        if (segmented) {
            s_red[threadIdx.x] = h;
            __syncthreads();
            for (unsigned off = blockDim.x >> 1; off > 0; off >>= 1) {
                if (threadIdx.x < off)
                    s_red[threadIdx.x] = bxor(s_red[threadIdx.x],
                                              s_red[threadIdx.x + off]);
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                DpfBlk* dst = &my_cws[static_cast<size_t>(b) * nk + k];
                atomicXor(reinterpret_cast<unsigned long long*>(&dst->lo),
                          static_cast<unsigned long long>(s_red[0].lo));
                atomicXor(reinterpret_cast<unsigned long long*>(&dst->hi),
                          static_cast<unsigned long long>(s_red[0].hi));
            }
            __syncthreads();
        } else if (active) {
            DpfBlk* dst = &my_cws[static_cast<size_t>(b) * nk + k];
            atomicXor(reinterpret_cast<unsigned long long*>(&dst->lo),
                      static_cast<unsigned long long>(h.lo));
            atomicXor(reinterpret_cast<unsigned long long*>(&dst->hi),
                      static_cast<unsigned long long>(h.hi));
        }
    }
}

// (3) Reconstruct CW = my ^ peer (plus the XOR-fold child), then write m
// children per parent into the next frontier buffer. The per-parent fold
// (the half-tree m-th child) is recomputed from the H_k values this thread
// loads anyway — no separate fold buffer.
__global__ void expand_kernel(const DpfBlk* __restrict__ cur,
                              const DpfBlk* __restrict__ hashed,
                              const DpfBlk* __restrict__ my_cws,
                              const DpfBlk* __restrict__ peer_cws,
                              DpfBlk* __restrict__ nxt,
                              int B, int sz, int m, int nk,
                              size_t row_cur, size_t row_hashed) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(B) * sz;
    if (idx >= total) return;
    int b = static_cast<int>(idx / sz);
    int j = static_cast<int>(idx % sz);

    // Reconstruct the m correction words for this instance.
    DpfBlk CW[16];
    DpfBlk cwfold;
    cwfold.lo = 0;
    cwfold.hi = 0;
    for (int k = 0; k < nk; ++k) {
        DpfBlk c = bxor(my_cws[static_cast<size_t>(b) * nk + k],
                        peer_cws[static_cast<size_t>(b) * nk + k]);
        CW[k] = c;
        cwfold = bxor(cwfold, c);
    }
    CW[nk] = cwfold;  // nk == m-1

    DpfBlk parent = cur[b * row_cur + j];
    bool t = (parent.lo & 1ULL) != 0;

    DpfBlk* dst = &nxt[b * row_cur + static_cast<size_t>(j) * m];
    DpfBlk hfold;
    hfold.lo = 0;
    hfold.hi = 0;
    for (int k = 0; k < nk; ++k) {
        DpfBlk h = hashed[b * row_hashed + static_cast<size_t>(j) * nk + k];
        hfold = bxor(hfold, h);
        dst[k] = t ? bxor(h, CW[k]) : h;
    }
    DpfBlk last = bxor(hfold, parent);
    if (t) last = bxor(last, CW[nk]);
    dst[m - 1] = last;
}

}  // namespace

namespace {

// Shared core: runs the level loop and returns the DEVICE pointer holding the
// final frontier (B rows of stride D = leaves). The pointer aliases one of the
// pooled ping-pong buffers and stays valid until the next dpf_gpu_* call.
DpfBlk* batch_full_eval_core(int B, int n, int w,
                             const DpfBlk* roots,
                             const DpfBlk* dltsft,
                             const uint8_t* round_keys,
                             const DpfExchangeFn& exchange,
                             DpfGpuPrg prg) {
    if (w < 1 || w > 4)
        throw std::runtime_error(
            "dpf_gpu_batch_full_eval: w must be in [1,4] (expand_kernel CW[16])");
    if (n % w != 0)
        throw std::runtime_error("dpf_gpu_batch_full_eval: w must divide n");
    if (prg == DpfGpuPrg::AES && round_keys == nullptr)
        throw std::runtime_error(
            "dpf_gpu_batch_full_eval: AES PRG requires round_keys");

    const int m       = 1 << w;
    const int nk      = m - 1;
    const int levels  = n / w;
    const size_t D    = static_cast<size_t>(1) << n;          // leaves / instance
    const size_t maxsz = D / static_cast<size_t>(m);          // frontier at last level
    const size_t row_hashed = maxsz * static_cast<size_t>(nk);
    const size_t row_dltsft = static_cast<size_t>(levels) * nk;

    // ── Device allocations (cached across calls, grow-only) ─────────────
    std::lock_guard<std::mutex> lock(dpf_gpu_mutex());
    DpfGpuBuffers& bufs = dpf_gpu_buffers();

    // cur/nxt ping-pong and mycws/peercws are paired: grow each pair together
    // under one shared capacity.
    const size_t frontier_blocks = static_cast<size_t>(B) * D;
    if (frontier_blocks > bufs.cap_frontier) {
        if (bufs.cur) cudaFree(bufs.cur);
        if (bufs.nxt) cudaFree(bufs.nxt);
        bufs.cur = bufs.nxt = nullptr;
        bufs.cap_frontier = 0;
        check(cudaMalloc(&bufs.cur, frontier_blocks * sizeof(DpfBlk)), "cudaMalloc d_cur");
        check(cudaMalloc(&bufs.nxt, frontier_blocks * sizeof(DpfBlk)), "cudaMalloc d_nxt");
        bufs.cap_frontier = frontier_blocks;
    }
    const size_t cws_blocks = static_cast<size_t>(B) * nk;
    if (cws_blocks > bufs.cap_cws) {
        if (bufs.mycws) cudaFree(bufs.mycws);
        if (bufs.peercws) cudaFree(bufs.peercws);
        bufs.mycws = bufs.peercws = nullptr;
        bufs.cap_cws = 0;
        check(cudaMalloc(&bufs.mycws, cws_blocks * sizeof(DpfBlk)), "cudaMalloc d_mycws");
        check(cudaMalloc(&bufs.peercws, cws_blocks * sizeof(DpfBlk)), "cudaMalloc d_peercws");
        bufs.cap_cws = cws_blocks;
    }
    DpfGpuBuffers::ensure(bufs.hashed, bufs.cap_hashed,
                          static_cast<size_t>(B) * row_hashed, "cudaMalloc d_hashed");
    DpfGpuBuffers::ensure(bufs.dltsft, bufs.cap_dltsft,
                          static_cast<size_t>(B) * row_dltsft, "cudaMalloc d_dltsft");
    if (prg == DpfGpuPrg::AES)
        DpfGpuBuffers::ensure(bufs.rks, bufs.cap_rks,
                              static_cast<size_t>(nk) * 176, "cudaMalloc d_rks");

    DpfBlk* d_cur = bufs.cur;
    DpfBlk* d_nxt = bufs.nxt;
    DpfBlk* d_hashed = bufs.hashed;
    DpfBlk* d_dltsft = bufs.dltsft;
    DpfBlk* d_mycws = bufs.mycws;
    DpfBlk* d_peercws = bufs.peercws;
    uint8_t* d_rks = bufs.rks;

    // ── Uploads ──────────────────────────────────────────────────────────
    // roots[b] -> d_cur[b*D] (scatter into the wide frontier rows).
    check(cudaMemcpy2D(d_cur, D * sizeof(DpfBlk), roots, sizeof(DpfBlk),
                       sizeof(DpfBlk), static_cast<size_t>(B), cudaMemcpyHostToDevice),
          "H2D roots");
    check(cudaMemcpy(d_dltsft, dltsft,
                     static_cast<size_t>(B) * row_dltsft * sizeof(DpfBlk),
                     cudaMemcpyHostToDevice), "H2D dltsft");
    if (prg == DpfGpuPrg::AES)
        check(cudaMemcpy(d_rks, round_keys, static_cast<size_t>(nk) * 176,
                         cudaMemcpyHostToDevice), "H2D round_keys");

    std::vector<DpfBlk> h_mycws(static_cast<size_t>(B) * nk);
    std::vector<DpfBlk> h_peercws(static_cast<size_t>(B) * nk);
    const int TPB = 128;
    // AES stages the S-box + round keys in shared memory; ChaCha8 is
    // all-register and only needs the reduction buffer.
    const size_t smem_bytes =
        (prg == DpfGpuPrg::AES)
            ? 256 + static_cast<size_t>(nk) * 176
                  + static_cast<size_t>(TPB) * sizeof(DpfBlk)
            : static_cast<size_t>(TPB) * sizeof(DpfBlk);

    // Per-level profile (env PCG_DPF_LEVEL_MS=1): the GGM frontier grows as m^i
    // while every level pays ONE correction-word round trip of constant size, so
    // the compute/communication balance is wildly non-uniform across levels. The
    // schedule depends on that shape, so it is measured per level, not summed.
    const bool lvl_prof = (std::getenv("PCG_DPF_LEVEL_MS") != nullptr);
    cudaEvent_t ev[6];
    if (lvl_prof) for (auto& e : ev) check(cudaEventCreate(&e), "level ev");
    auto tick = [&](int k) { if (lvl_prof) check(cudaEventRecord(ev[k]), "level rec"); };
    auto span = [&](int a, int b) {
        float ms = 0.0f;
        if (lvl_prof) { check(cudaEventSynchronize(ev[b]), "level sync");
                        check(cudaEventElapsedTime(&ms, ev[a], ev[b]), "level el"); }
        return static_cast<double>(ms);
    };

    for (int i = 0; i < levels; ++i) {
        const int sz = 1 << (w * i);                 // m^i
        const size_t threads_p = static_cast<size_t>(B) * sz;
        const size_t cws_count = static_cast<size_t>(B) * nk;
        tick(0);

        // Seed my_cws with this level's DltSft shares; hash_kernel XORs the
        // per-column hash reduction on top.
        check(cudaMemcpy2D(d_mycws, static_cast<size_t>(nk) * sizeof(DpfBlk),
                           d_dltsft + static_cast<size_t>(i) * nk,
                           row_dltsft * sizeof(DpfBlk),
                           static_cast<size_t>(nk) * sizeof(DpfBlk),
                           static_cast<size_t>(B), cudaMemcpyDeviceToDevice),
              "D2D seed my_cws");

        if (prg == DpfGpuPrg::AES) {
            hash_kernel<<<static_cast<unsigned>((threads_p + TPB - 1) / TPB), TPB,
                          static_cast<unsigned>(smem_bytes)>>>(
                d_cur, d_hashed, d_mycws, d_rks, B, sz, nk, D, row_hashed);
        } else {
            hash_kernel_chacha8<<<static_cast<unsigned>((threads_p + TPB - 1) / TPB),
                                  TPB, static_cast<unsigned>(smem_bytes)>>>(
                d_cur, d_hashed, d_mycws, B, sz, nk, D, row_hashed);
        }
        check(cudaGetLastError(), "launch hash_kernel");
        tick(1);

        check(cudaMemcpy(h_mycws.data(), d_mycws,
                         cws_count * sizeof(DpfBlk), cudaMemcpyDeviceToHost),
              "D2H my_cws");
        tick(2);

        exchange(h_mycws.data(), h_peercws.data(), cws_count);
        tick(3);

        check(cudaMemcpy(d_peercws, h_peercws.data(),
                         cws_count * sizeof(DpfBlk), cudaMemcpyHostToDevice),
              "H2D peer_cws");
        tick(4);

        expand_kernel<<<static_cast<unsigned>((threads_p + TPB - 1) / TPB), TPB>>>(
            d_cur, d_hashed, d_mycws, d_peercws, d_nxt,
            B, sz, m, nk, D, row_hashed);
        check(cudaGetLastError(), "launch expand_kernel");
        tick(5);
        if (lvl_prof) {
            std::fprintf(stderr,
                "[dpf-level] i=%2d frontier=%d nodes=%zu hash=%.4f d2h=%.4f "
                "exch=%.4f h2d=%.4f expand=%.4f ms  (msg=%zu B, 1 RTT)\n",
                i, sz, threads_p, span(0, 1), span(1, 2), span(2, 3),
                span(3, 4), span(4, 5), cws_count * sizeof(DpfBlk));
        }

        std::swap(d_cur, d_nxt);
    }
    if (lvl_prof) for (auto& e : ev) cudaEventDestroy(e);

    // d_cur now holds the leaves: B rows of stride D, all D valid.
    return d_cur;
}

}  // namespace

void dpf_gpu_batch_full_eval(int B, int n, int w,
                             const DpfBlk* roots,
                             const DpfBlk* dltsft,
                             const uint8_t* round_keys,
                             int party,
                             const DpfExchangeFn& exchange,
                             DpfBlk* out_leaves,
                             DpfGpuPrg prg) {
    (void)party;  // ordering handled inside `exchange`
    if (B <= 0) return;
    DpfBlk* d_leaves =
        batch_full_eval_core(B, n, w, roots, dltsft, round_keys, exchange, prg);
    const size_t frontier_blocks =
        static_cast<size_t>(B) << n;  // B * D contiguous, same row stride
    check(cudaMemcpy(out_leaves, d_leaves, frontier_blocks * sizeof(DpfBlk),
                     cudaMemcpyDeviceToHost), "D2H leaves");
    // Device buffers stay cached in dpf_gpu_buffers() for the next call.
}

const DpfBlk* dpf_gpu_batch_full_eval_device(int B, int n, int w,
                                             const DpfBlk* roots,
                                             const DpfBlk* dltsft,
                                             const uint8_t* round_keys,
                                             int party,
                                             const DpfExchangeFn& exchange,
                                             DpfGpuPrg prg) {
    (void)party;  // ordering handled inside `exchange`
    if (B <= 0) return nullptr;
    return batch_full_eval_core(B, n, w, roots, dltsft, round_keys, exchange,
                                prg);
}

}  // namespace pcg_cuda
