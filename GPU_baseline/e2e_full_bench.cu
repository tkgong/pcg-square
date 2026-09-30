// TRUE end-to-end PCG expansion on one GPU, measured in a single process.
//
// This replaces the stitched e2e (c^2*dpf_blk + 2c^2*ntt_per_mul) that earlier
// campaigns assembled arithmetically.  Here the whole expansion actually runs:
//
//   c^2 blocks, each t^2 DPF instances of depth n = log2(2N/t)  <-- the REAL
//   shape. The output layer is per block (pos = (b/t + b%t)*bin), so the
//   blocks must be run one at a time; earlier campaigns launched --B c^2,
//   i.e. 1/t^2 of the workload, which is why their ns/leaf was launch-bound
//   nonsense. Running block-by-block also caps residency at one block, so the
//   leaf tensor is c^2 times smaller than the whole expansion.
//   -> per-level correction-word exchange (latency injected, PCG_RTT_US)
//   -> out_sums  -> 2 Beaver opens -> out_scatter into the folded g
//   -> 2*c^2 negacyclic poly_muls of degree N, batch c^2 (square four-step)
//
// Any size whose leaf tensor does not fit is reported OOM and skipped; nothing
// is extrapolated into it.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/aes_gpu.h"
#include "common/dpf_gpu.h"
#include "common/leaf_convert_cuda.h"
#include "common/poly_mul_cuda.h"

using pcg_cuda::DpfBlk;
static const uint64_t P = 4611686018326724609ULL;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

int main(int argc, char** argv) {
    int c = 4, t = 16, logN = 20, iters = 1, batch_blocks = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--c") && i+1 < argc) c = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--t") && i+1 < argc) t = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logN") && i+1 < argc) logN = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-blocks")) batch_blocks = 1;
    }
    const int N = 1 << logN;
    int tlog = 0; while ((1 << tlog) < t) ++tlog;
    const int n = logN + 1 - tlog;              // depth: D = 2N/t
    const long D = 1L << n;
    const int B = t * t;                        // instances in ONE block
    const int nblk = c * c;                     // blocks per expansion
    const int batch = c * c;
    const long long leaves_blk = (long long)B * D;
    const long long leaves = leaves_blk * nblk;
    const int gN = (int)(t * (D / 2));

    // Residency: the reference implementation keeps ONE block live, so the leaf
    // tensor is c^2 times smaller than the whole expansion. --batch-blocks
    // expands all c^2*t^2 instances in a single call instead (maximum
    // parallelism, c^2 times the memory) and scatters per block via a pointer
    // offset. That is a legal reordering -- the instances are independent and
    // only the scatter is per-block -- and it bounds the GPU baseline from the
    // fast side, so both arms are measured rather than one being assumed.
    const double leaf_gb = (double)(batch_blocks ? leaves : leaves_blk) * 16.0 / (1ull<<30);
    const double ntt_gb  = (double)batch * N * 8.0 * 3 / (1ull<<30);
    cudaFree(0);   // force context creation so the query is meaningful
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    std::printf("e2e c=%d t=%d logN=%d %s | blocks=%d B/blk=%d n=%d D=%ld "
                "leaves=%lld | leaf=%.2f GB ntt=%.2f GB free=%.2f GB\n",
                c, t, logN, batch_blocks ? "[batched]" : "[serial]",
                nblk, B, n, D, leaves, leaf_gb, ntt_gb,
                (double)freeb/(1ull<<30));
    // 1.6x: the expansion holds cur+nxt frontier buffers (~1.5x the leaf
    // tensor at the last level), not just the leaves.
    if (leaf_gb * 1.60 + ntt_gb > (double)freeb/(1ull<<30)) {
        std::printf("RESULT c=%d t=%d logN=%d mode=%s STATUS=OOM_SKIP\n",
                    c, t, logN, batch_blocks ? "batched" : "serial");
        return 0;
    }

    const int w = 1, levels = n / w, nk = (1 << w) - 1;
    std::vector<DpfBlk> roots(B), dltsft((size_t)B * levels * nk);
    for (int b = 0; b < B; ++b) { roots[b].lo = 0x9e3779b97f4a7c15ULL*(b+1); roots[b].hi = b+1; }
    for (size_t i = 0; i < dltsft.size(); ++i) { dltsft[i].lo = 0x1234567ULL*(i+1); dltsft[i].hi = i; }
    std::vector<uint8_t> master(16); for (int i=0;i<16;i++) master[i]=i*7+1;
    std::vector<uint8_t> round_keys((size_t)nk * 176);
    for (int k = 0; k < nk; ++k) {
        std::vector<uint8_t> sub(16); for (int i=0;i<16;i++) sub[i]=master[i]^(uint8_t)(k*31+i);
        pcg_cuda::aes128_expand_key_host(sub.data(), &round_keys[(size_t)k*176]);
    }

    const double rtt_us = std::getenv("PCG_RTT_US") ? atof(std::getenv("PCG_RTT_US")) : 0.0;
    long long exch_n = 0; double exch_us = 0.0;
    auto spin = [&](double us){
        if (us <= 0.0) return;
        const auto t0 = std::chrono::steady_clock::now();
        const auto dl = t0 + std::chrono::nanoseconds((long long)(us*1000.0));
        while (std::chrono::steady_clock::now() < dl) {}
        exch_us += std::chrono::duration<double,std::micro>(
                       std::chrono::steady_clock::now() - t0).count();
    };
    auto exchange = [&](const DpfBlk*, DpfBlk* peer, size_t cnt){
        for (size_t i=0;i<cnt;i++){ peer[i].lo=0; peer[i].hi=0; }
        ++exch_n; spin(rtt_us);
    };

    std::vector<uint64_t> sumC(B), sumT(B), CW(B), g((size_t)gN);
    for (int b = 0; b < B; ++b) CW[b] = (0xABCDEFULL*(b+1)) % P;
    std::vector<uint64_t> pa((size_t)batch*N), pb((size_t)batch*N), pc((size_t)batch*N);
    for (size_t i = 0; i < pa.size(); ++i) { pa[i] = (0x9E37ULL*i+1)%P; pb[i] = (0xBEEFULL*i+7)%P; }

    // Warm-up poly_mul BEFORE the wall clock: SquarePlan construction (host
    // modpow twiddle tables, seconds at large N) is per-N one-time state that a
    // deployment reuses across expansions -- without this it dominated the
    // measured "ntt" phase 100:1.
    pcg_cuda::poly_mul_u64_gpuntt_square(pc.data(), pa.data(), pb.data(),
                                         batch, N, P, nullptr);
    double dpf_ms=0, sums_ms=0, sca_ms=0, ntt_ms=0, net_ms=0, wall_ms=0;
    uint64_t cks = 0;
    for (int it = 0; it < iters; ++it) {
        exch_n = 0; exch_us = 0.0;
        const auto W0 = std::chrono::steady_clock::now();

        double e_ms=0, s_ms=0, n_ms=0, c_ms=0;
        auto a0 = std::chrono::steady_clock::now();
        if (batch_blocks) {
            // one expansion over all c^2*t^2 instances
            auto b0 = std::chrono::steady_clock::now();
            const DpfBlk* all = pcg_cuda::dpf_gpu_batch_full_eval_device(
                B * nblk, n, w, roots.data(), dltsft.data(), round_keys.data(), 0,
                exchange, pcg_cuda::DpfGpuPrg::CHACHA8);
            CK(cudaDeviceSynchronize());
            auto b1 = std::chrono::steady_clock::now();
            auto d=[&](auto x, auto y){ return std::chrono::duration<double,std::milli>(y-x).count(); };
            e_ms += d(b0,b1);
            for (int blk = 0; blk < nblk; ++blk) {
                const DpfBlk* dl = all + (size_t)blk * B * D;   // this block's slice
                auto q0 = std::chrono::steady_clock::now();
                if (getenv("CONV_V2")) pcg_cuda::dpf_out_sums_v2(dl, B, D, t, 0, P, gN, sumC.data(), sumT.data()); else pcg_cuda::dpf_out_sums(dl, B, D, P, sumC.data(), sumT.data());
                CK(cudaDeviceSynchronize());
                auto q1 = std::chrono::steady_clock::now();
                spin(rtt_us); spin(rtt_us);
                auto q2 = std::chrono::steady_clock::now();
                if (getenv("CONV_V2")) pcg_cuda::dpf_out_scatter_g_v2(B, D, t, 0, P, CW.data(), g.data(), gN); else pcg_cuda::dpf_out_scatter_g(dl, B, D, t, 0, P, CW.data(), g.data(), gN);
                CK(cudaDeviceSynchronize());
                auto q3 = std::chrono::steady_clock::now();
                s_ms += d(q0,q1); n_ms += d(q1,q2); c_ms += d(q2,q3);
                for (int b = 0; b < B; b += 97) cks ^= sumC[b]^sumT[b];
                for (int i = 0; i < gN; i += 9973) cks ^= g[i];
            }
        } else
        for (int blk = 0; blk < nblk; ++blk) {
            auto b0 = std::chrono::steady_clock::now();
            const DpfBlk* d_leaves = pcg_cuda::dpf_gpu_batch_full_eval_device(
                B, n, w, roots.data(), dltsft.data(), round_keys.data(), 0,
                exchange, pcg_cuda::DpfGpuPrg::CHACHA8);
            CK(cudaDeviceSynchronize());
            auto b1 = std::chrono::steady_clock::now();
            if (getenv("CONV_V2")) pcg_cuda::dpf_out_sums_v2(d_leaves, B, D, t, 0, P, gN, sumC.data(), sumT.data()); else pcg_cuda::dpf_out_sums(d_leaves, B, D, P, sumC.data(), sumT.data());
            CK(cudaDeviceSynchronize());
            auto b2 = std::chrono::steady_clock::now();
            spin(rtt_us); spin(rtt_us);          // the block's two Beaver opens
            auto b3 = std::chrono::steady_clock::now();
            if (getenv("CONV_V2")) pcg_cuda::dpf_out_scatter_g_v2(B, D, t, 0, P, CW.data(), g.data(), gN); else pcg_cuda::dpf_out_scatter_g(d_leaves, B, D, t, 0, P, CW.data(), g.data(), gN);
            CK(cudaDeviceSynchronize());
            auto b4 = std::chrono::steady_clock::now();
            auto d=[&](auto x, auto y){ return std::chrono::duration<double,std::milli>(y-x).count(); };
            e_ms+=d(b0,b1); s_ms+=d(b1,b2); n_ms+=d(b2,b3); c_ms+=d(b3,b4);
            for (int b = 0; b < B; b += 97) cks ^= sumC[b]^sumT[b];
            for (int i = 0; i < gN; i += 9973) cks ^= g[i];
        }
        auto a4 = std::chrono::steady_clock::now();

        // step 4: 2*c^2 poly_muls of degree N, batch c^2 -> 2 batched calls
        for (int r = 0; r < 2; ++r)
            pcg_cuda::poly_mul_u64_gpuntt_square(pc.data(), pa.data(), pb.data(),
                                                 batch, N, P, nullptr);
        CK(cudaDeviceSynchronize());
        auto a5 = std::chrono::steady_clock::now();

        auto ms=[&](auto x, auto y){ return std::chrono::duration<double,std::milli>(y-x).count(); };
        dpf_ms += e_ms; sums_ms += s_ms; net_ms += n_ms; sca_ms += c_ms;
        ntt_ms += ms(a4,a5); wall_ms += ms(W0,a5);
        cks ^= pc[0] ^ pc[N/2];
    }
    { uint64_t fp = 0; for (int i = 0; i < gN; ++i) fp = fp * 0x9E3779B97F4A7C15ULL + g[i]; std::printf("G_FINGERPRINT=%016llx\n", (unsigned long long)fp); }
    dpf_ms/=iters; sums_ms/=iters; sca_ms/=iters; ntt_ms/=iters;
    net_ms/=iters; wall_ms/=iters;
    const double conv = sums_ms + sca_ms;
    std::printf("  expand=%.3f  out_sums=%.3f  out_scatter=%.3f  beaver_net=%.3f  ntt=%.3f  WALL=%.3f ms\n",
                dpf_ms, sums_ms, sca_ms, net_ms, ntt_ms, wall_ms);
    std::printf("RESULT c=%d t=%d logN=%d mode=%s rtt=%.1f B=%d n=%d leaves=%lld "
                "expand=%.4f convert=%.4f beaver=%.4f ntt=%.4f wall=%.4f "
                "ns_per_leaf=%.5f exch=%lld checksum=%016llx STATUS=OK\n",
                c, t, logN, batch_blocks ? "batched" : "serial", rtt_us, B, n, leaves,
                dpf_ms, conv, net_ms, ntt_ms, wall_ms,
                (dpf_ms+conv)*1e6/(double)leaves,
                exch_n, (unsigned long long)cks);
    return 0;
}
