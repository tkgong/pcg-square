// Kernel-only microbench over the PRODUCTION kernels (caliber fix D1):
// hash_kernel_chacha8 + expand_kernel via dpf_gpu_batch_full_eval_device, then
// the NEW Beaver-corrected output layer (dpf_out_sums + dpf_out_scatter_g:
// per-leaf ChaCha8 out-hash H' + sparse mod-P convert + control-bit; sign-free
// per-instance sums, then y = C + tau?CW + negacyclic 2N->N fold scatter). NO
// modular multiply on the device. Replaces the retired single-bin-atomicAdd
// stand-in AND the old CW=beta/v leaf_convert for all baseline measurements.
//
// Single process, stubbed exchange (timing only), nsys-friendly (--iters 1).
// Prints a leaf checksum so traced/untraced runs can be compared, and
// per-phase cudaEvent wall (tier: phase-wall; use nsys for kernel-only).
//
// Build (L40S): /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 -I. \
//     -I$HOME/GPU-NTT/install/include/GPUNTT-1.0 \
//     endtoend/dpf_real_bench.cu common/dpf_gpu.cu common/aes_gpu.cu \
//     common/leaf_convert_cuda.cu -L$HOME/GPU-NTT/install/lib -lntt-1.0 \
//     -o endtoend/bin/dpf_real_bench
// Build (B200): same with -arch=sm_100 and that host's GPU-NTT install.
#include "common/dpf_gpu.h"
#include "common/aes_gpu.h"
#include "common/leaf_convert_cuda.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using pcg_cuda::DpfBlk;

static const uint64_t P = 4611686018326724609ULL;

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

int main(int argc, char** argv) {
    int n = 14, B = 16, w = 1, iters = 5, t_param = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--B") && i+1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--w") && i+1 < argc) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--t") && i+1 < argc) t_param = atoi(argv[++i]);
    }
    // protocol shape: B = t^2 instances, D = 2^n leaves each, bin = D/2,
    // g length N = t*bin. Default t = floor(sqrt(B)) (exact for B = t^2).
    int t = t_param;
    if (t == 0) { t = 1; while ((t+1)*(t+1) <= B) ++t; }
    long D = 1L << n;
    long total = (long)B * D;
    int gN = (int)(t * (D / 2));
    int levels = n / w, nk = (1 << w) - 1;

    std::vector<DpfBlk> roots(B);
    std::vector<DpfBlk> dltsft((size_t)B * levels * nk);
    for (int b = 0; b < B; ++b) { roots[b].lo = 0x9e3779b97f4a7c15ULL*(b+1); roots[b].hi = b+1; }
    for (size_t i = 0; i < dltsft.size(); ++i) { dltsft[i].lo = 0x1234567ULL*(i+1); dltsft[i].hi = i; }
    std::vector<uint8_t> master(16); for (int i=0;i<16;i++) master[i]=i*7+1;
    std::vector<uint8_t> round_keys((size_t)nk * 176);
    for (int k = 0; k < nk; ++k) {
        std::vector<uint8_t> sub(16); for (int i=0;i<16;i++) sub[i]=master[i]^(uint8_t)(k*31+i);
        pcg_cuda::aes128_expand_key_host(sub.data(), &round_keys[(size_t)k*176]);
    }
    // Per-level correction-word exchange. The peer's shares are stubbed (this is a
    // single-process timing harness), but the LATENCY of the round trip is
    // injected so the same binary measures the baseline under every interconnect
    // tier: PCG_RTT_US selects the tier (NVLink ~5, PCIe ~16, TCP loopback ~25.2,
    // datacenter 50, WAN 2000). Busy-wait rather than sleep: at single-digit
    // microseconds a scheduler round trip would dominate the quantity measured.
    const double rtt_us = std::getenv("PCG_RTT_US") ? atof(std::getenv("PCG_RTT_US")) : 0.0;
    long long exch_n = 0; double exch_us = 0.0;
    auto exchange = [&](const DpfBlk*, DpfBlk* peer, size_t count){
        for (size_t i=0;i<count;i++){ peer[i].lo=0; peer[i].hi=0; }
        ++exch_n;
        if (rtt_us <= 0.0) return;
        const auto t0 = std::chrono::steady_clock::now();
        const auto deadline = t0 + std::chrono::nanoseconds((long long)(rtt_us * 1000.0));
        while (std::chrono::steady_clock::now() < deadline) { /* spin */ }
        exch_us += std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - t0).count();
    };

    std::vector<uint64_t> sumC(B), sumT(B), CW(B), g((size_t)gN);
    for (int b = 0; b < B; ++b) CW[b] = (0xABCDEFULL * (b + 1)) % P;

    cudaEvent_t e0,e1,e2,e3; CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
    CK(cudaEventCreate(&e2));CK(cudaEventCreate(&e3));
    int WARM = (iters > 1) ? 1 : 0;
    double exp_ms=0, sum_ms=0, sca_ms=0;
    const DpfBlk* d_leaves = nullptr;
    for (int it = 0; it < WARM + iters; ++it) {
        CK(cudaEventRecord(e0));
        d_leaves = pcg_cuda::dpf_gpu_batch_full_eval_device(
            B, n, w, roots.data(), dltsft.data(), round_keys.data(), 0,
            exchange, pcg_cuda::DpfGpuPrg::CHACHA8);
        CK(cudaEventRecord(e1));
        pcg_cuda::dpf_out_sums(d_leaves, B, D, P, sumC.data(), sumT.data());
        CK(cudaEventRecord(e2));
        pcg_cuda::dpf_out_scatter_g(d_leaves, B, D, t, 0, P, CW.data(),
                                    g.data(), gN);
        CK(cudaEventRecord(e3)); CK(cudaEventSynchronize(e3));
        float a=0,b2=0,c=0;
        CK(cudaEventElapsedTime(&a,e0,e1)); CK(cudaEventElapsedTime(&b2,e1,e2));
        CK(cudaEventElapsedTime(&c,e2,e3));
        if (it >= WARM) { exp_ms+=a; sum_ms+=b2; sca_ms+=c; }
    }
    exp_ms/=iters; sum_ms/=iters; sca_ms/=iters;
    uint64_t cks = 0;
    for (int b = 0; b < B; ++b) cks ^= sumC[b] ^ sumT[b];
    for (int i = 0; i < gN; i += 97) cks ^= g[i];
    double tot = exp_ms + sum_ms + sca_ms;
    printf("REAL-KERNEL DPF+convert  n=%d B=%d t=%d w=%d prg=chacha8  leaves=%ld  gN=%d  checksum=%016llx\n",
           n, B, t, w, total, gN, (unsigned long long)cks);
    printf("  tier=phase-wall (cudaEvent; use nsys for kernel-only):\n");
    printf("  expand (hash+expand kernels + CW D2H/H2D)   : %8.3f ms  %.4f ns/leaf\n", exp_ms, exp_ms*1e6/total);
    printf("  out_sums   (per-leaf ChaCha8 H' + mod-P)    : %8.3f ms  %.4f ns/leaf\n", sum_ms, sum_ms*1e6/total);
    printf("  out_scatter (y=C+tau?CW + negacyclic fold)  : %8.3f ms  %.4f ns/leaf\n", sca_ms, sca_ms*1e6/total);
    printf("  TOTAL DPF+convert                           : %8.3f ms  %.4f ns/leaf\n", tot, tot*1e6/total);
    printf("  network: rtt=%.1f us  exchanges=%lld  exposed=%.3f ms (%.1f%% of TOTAL)\n",
           rtt_us, exch_n / (long long)(iters + WARM),
           exch_us / 1e3 / (iters + WARM),
           tot > 0 ? 100.0 * (exch_us / 1e3 / (iters + WARM)) / tot : 0.0);
    return 0;
}
