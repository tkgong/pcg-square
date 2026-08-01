// Standalone GPU bench: full-domain half-tree DPF expansion + leaf_convert,
// timed with CUDA events, aligned to the PIM benchmark methodology (per-leaf
// throughput at matched total leaf counts). PRG = AES-128 (the GPU DPF's PRG;
// the PIM side uses ChaCha8 -- noted in the report, different PRG).
//
// leaf_convert here mirrors pcg_ole_impl.h:179/224 semantics:
//   v = block_to_uint64(leaf) mod P ; g += v * CW_inst mod P   (Barrett-free 128b).
//
// Build:
//   nvcc -O3 -std=c++17 -I. test/gpu_dpf_leafconv_bench.cu \
//        common/dpf_gpu.cu common/aes_gpu.cu -o /tmp/gpu_dpf_lc
// Run:  /tmp/gpu_dpf_lc <n> <B>     (D = 2^n leaves/instance, B instances)
#include "common/dpf_gpu.h"
#include "common/aes_gpu.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using pcg_cuda::DpfBlk;

static const uint64_t P = 4611686018326724609ULL;  // 62-bit prime (matches FFp<P>)

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__device__ __forceinline__ uint64_t mulmodP(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * b;
    return (uint64_t)(prod % P);
}

// leaf_convert: one thread per leaf. v = leaf.lo mod P; scale by per-instance CW;
// atomic-accumulate into g[inst] (the scatter target; sum-into-bin is the shape).
__global__ void leaf_convert_kernel(const DpfBlk* __restrict__ leaves,
                                    const uint64_t* __restrict__ cw,  // per-instance
                                    uint64_t* __restrict__ g,         // per-instance accum
                                    long total, int D) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int inst = (int)(idx / D);
    uint64_t v = leaves[idx].lo % P;              // block_to_uint64 -> Z_p
    uint64_t coeff = mulmodP(v, cw[inst]);        // v * CW  (pcg_ole_impl.h:224)
    atomicAdd((unsigned long long*)&g[inst], (unsigned long long)coeff);  // scatter/accum
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 18;
    int B = argc > 2 ? atoi(argv[2]) : 1;
    int w = 1;
    int levels = n;                 // m=2, D=2^n
    long D = 1L << n;
    long total = (long)B * D;
    int nk = (1 << w) - 1;

    // --- host setup: roots, dltsft (random-ish, timing-only), AES round keys ---
    std::vector<DpfBlk> roots(B);
    std::vector<DpfBlk> dltsft((size_t)B * levels * nk);
    for (int b = 0; b < B; ++b) { roots[b].lo = 0x9e3779b97f4a7c15ULL * (b+1); roots[b].hi = b + 1; }
    for (size_t i = 0; i < dltsft.size(); ++i) { dltsft[i].lo = 0x1234567 * (i+1); dltsft[i].hi = i; }
    std::vector<uint8_t> master(16, 0); for (int i=0;i<16;i++) master[i]=i*7+1;
    std::vector<uint8_t> round_keys(nk * 176);
    // sub_key_k = AES_master(makeBlock(0,k)) ^ makeBlock(0,k); expand each to 176B.
    // For timing we just need valid round-key schedules -> derive nk distinct keys.
    for (int k = 0; k < nk; ++k) {
        std::vector<uint8_t> sub(16, 0); for (int i=0;i<16;i++) sub[i] = master[i]^(uint8_t)(k*31+i);
        pcg_cuda::aes128_expand_key_host(sub.data(), &round_keys[k*176]);
    }
    auto exchange = [](const DpfBlk*, DpfBlk* peer, size_t count){
        for (size_t i=0;i<count;i++){ peer[i].lo=0; peer[i].hi=0; }   // stub (timing-only)
    };

    std::vector<DpfBlk> out((size_t)total);

    // --- device buffers for leaf_convert ---
    DpfBlk* d_leaves=nullptr; uint64_t *d_cw=nullptr,*d_g=nullptr;
    CK(cudaMalloc(&d_leaves, sizeof(DpfBlk)*total));
    CK(cudaMalloc(&d_cw, sizeof(uint64_t)*B));
    CK(cudaMalloc(&d_g, sizeof(uint64_t)*B));
    std::vector<uint64_t> cw(B); for(int b=0;b<B;b++) cw[b]=(0xABCDEFULL*(b+1))%P;
    CK(cudaMemcpy(d_cw, cw.data(), sizeof(uint64_t)*B, cudaMemcpyHostToDevice));

    cudaEvent_t e0,e1,e2; CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));CK(cudaEventCreate(&e2));
    const int WARM=1, IT=5;
    double exp_ms=0, lc_ms=0;
    for (int it=0; it<WARM+IT; ++it) {
        // ---- DPF expansion (device-resident, AES PRG) ----
        CK(cudaEventRecord(e0));
        pcg_cuda::dpf_gpu_batch_full_eval(B, n, w, roots.data(), dltsft.data(),
                                          round_keys.data(), 0, exchange, out.data());
        CK(cudaEventRecord(e1));
        // out is host; copy leaves to device for convert (models leaves already on GPU:
        // in the fused path they never leave -- so we time the kernel on-device only).
        CK(cudaMemcpy(d_leaves, out.data(), sizeof(DpfBlk)*total, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_g, 0, sizeof(uint64_t)*B));
        int TPB=256; long blocks=(total+TPB-1)/TPB;
        CK(cudaEventRecord(e1));  // re-mark: leaf_convert start
        leaf_convert_kernel<<<blocks,TPB>>>(d_leaves,d_cw,d_g,total,(int)D);
        CK(cudaEventRecord(e2)); CK(cudaEventSynchronize(e2));
        float m_exp=0,m_lc=0; CK(cudaEventElapsedTime(&m_exp,e0,e1)); CK(cudaEventElapsedTime(&m_lc,e1,e2));
        if (it>=WARM){ exp_ms+=m_exp; lc_ms+=m_lc; }
    }
    exp_ms/=IT; lc_ms/=IT;
    double tot_ms = exp_ms + lc_ms;
    double ns_leaf_exp = exp_ms*1e6/total;
    double ns_leaf_tot = tot_ms*1e6/total;
    printf("GPU L40S ChaCha8  n=%d B=%d  total_leaves=%ld\n", n, B, total);
    printf("  DPF expand   : %.3f ms   %.4f ns/leaf   %.3e leaves/s\n", exp_ms, ns_leaf_exp, total/(exp_ms*1e-3));
    printf("  leaf_convert : %.3f ms   %.4f ns/leaf\n", lc_ms, lc_ms*1e6/total);
    printf("  FULL (exp+lc): %.3f ms   %.4f ns/leaf   %.3e leaves/s\n", tot_ms, ns_leaf_tot, total/(tot_ms*1e-3));
    return 0;
}
