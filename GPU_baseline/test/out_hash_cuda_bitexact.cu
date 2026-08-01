// Bit-exactness gate for the device output layer vs the host construction.
// Standalone (NO emp-tool dependency): the host reference reimplements
// ChaCha8OutHash directly on common/chacha8.h, and the device side is pulled
// in by #including leaf_convert_cuda.cu (same TU -> its anonymous-namespace
// kernels are reachable via the public entry points).
//
// Checks, on random leaves:
//   1. dpf_out_sums    == host sumC (mod-P sequential) and sumT, per instance
//   2. dpf_out_scatter_g == host y = ±(C + tau?CW) negacyclic-fold reference
// Any mismatch exits nonzero. Exit 77 if no CUDA device (skip convention).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_86 -I. test/out_hash_cuda_bitexact.cu \
//            -o out_hash_bitexact
// (leaf_convert_cuda.cu is #included; do not compile it separately here.)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "common/chacha8.h"
#include "common/leaf_convert_cuda.cu"   // device side under test (same TU)

using pcg_cuda::DpfBlk;

static constexpr uint64_t P = 4611686018326724609ULL;   // 2^62 - 3*2^25 + 1

// Host reference of ChaCha8OutHash (mirrors half_tree_dpf/tree_prg.h).
static uint64_t host_out_hash_lo64(const DpfBlk& leaf) {
    uint8_t key[32];
    std::memset(key, 0, 32);
    std::memcpy(key, &leaf, 16);
    ChaCha8PRG prg(key);
    prg.set_nonce(pcg_cuda::kOutHashDomain);
    uint8_t buf[64];
    prg.fill_blocks(buf, 1);
    uint64_t lo;
    std::memcpy(&lo, buf, 8);
    return lo;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::fprintf(stderr, "no CUDA device — skipping\n");
        return 77;
    }

    const int t = 3, B = t * t;          // 9 instances
    const size_t D = 1 << 10;            // 1024 leaves each
    const int bin = static_cast<int>(D / 2);
    const int N = t * bin;

    std::mt19937_64 rng(12345);
    std::vector<DpfBlk> leaves(B * D);
    for (auto& l : leaves) { l.lo = rng(); l.hi = rng(); }
    std::vector<uint64_t> CW(B);
    for (auto& c : CW) c = rng() % P;

    DpfBlk* d_leaves = nullptr;
    if (cudaMalloc(&d_leaves, leaves.size() * sizeof(DpfBlk)) != cudaSuccess)
        { std::fprintf(stderr, "cudaMalloc failed\n"); return 1; }
    cudaMemcpy(d_leaves, leaves.data(), leaves.size() * sizeof(DpfBlk),
               cudaMemcpyHostToDevice);

    int fails = 0;
    for (int party = 0; party <= 1; ++party) {
        // ---- device ----
        std::vector<uint64_t> sumC(B), sumT(B), g(N);
        pcg_cuda::dpf_out_sums(d_leaves, B, D, P, sumC.data(), sumT.data());
        pcg_cuda::dpf_out_scatter_g(d_leaves, B, D, t, party, P, CW.data(),
                                    g.data(), N);

        // ---- host reference ----
        std::vector<uint64_t> hC(B, 0), hT(B, 0), hg(N, 0);
        for (int b = 0; b < B; ++b) {
            const int kk = b / t, ll = b % t;
            for (size_t d = 0; d < D; ++d) {
                const DpfBlk& leaf = leaves[b * D + d];
                uint64_t C = host_out_hash_lo64(leaf) % P;
                uint64_t tau = leaf.lo & 1;
                hC[b] = (hC[b] + C) % P;
                hT[b] += tau;
                uint64_t y = C;
                if (tau) { y += CW[b]; if (y >= P) y -= P; }
                if (party == 1 && y != 0) y = P - y;
                if (y == 0) continue;
                size_t pos = static_cast<size_t>(kk + ll) * bin + d;
                if (pos >= static_cast<size_t>(N)) { pos -= N; y = P - y; }
                hg[pos] = (hg[pos] + y) % P;
            }
        }

        for (int b = 0; b < B; ++b) {
            if (sumC[b] != hC[b]) { ++fails;
                std::fprintf(stderr, "party%d sumC[%d] dev=%llu host=%llu\n",
                    party, b, (unsigned long long)sumC[b],
                    (unsigned long long)hC[b]); }
            if (sumT[b] != hT[b]) { ++fails;
                std::fprintf(stderr, "party%d sumT[%d] mismatch\n", party, b); }
        }
        for (int d = 0; d < N; ++d)
            if (g[d] != hg[d]) { ++fails;
                if (fails < 8) std::fprintf(stderr,
                    "party%d g[%d] dev=%llu host=%llu\n", party, d,
                    (unsigned long long)g[d], (unsigned long long)hg[d]); }
        std::printf("party %d: sums %s, scatter %s\n", party,
                    fails ? "FAIL" : "OK", fails ? "FAIL" : "OK");
    }

    cudaFree(d_leaves);
    if (fails) { std::printf("BITEXACT: FAIL (%d)\n", fails); return 1; }
    std::printf("BITEXACT: PASS (host ChaCha8OutHash == device, sums+scatter, "
                "both parties)\n");
    return 0;
}
