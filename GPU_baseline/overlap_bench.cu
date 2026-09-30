// overlap_bench: is "GPU DPF + GPU NTT" a fair GPU-only baseline, or can a
// GPU-only implementation overlap the two (reviewer D)?  Runs the paper's
// measured DPF pipeline (expand + out_sums + scatter, default stream) for K
// blocks and the GPU-NTT merge poly-muls the e2e attributes to those blocks
// (2 per block) on a non-blocking stream, and times: each alone, and both
// launched together.  Reports t_both / (t_dpf + t_ntt) and t_both / max(...).
//
//   ./overlap_bench --c 4 --t 16 --logN 20 --blocks 4 --iters 3
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "common/aes_gpu.h"
#include "common/dpf_gpu.h"
#include "common/leaf_convert_cuda.h"
#include "fused4/fused4_ntt.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/common/nttparameters.cuh"
using namespace gpuntt;
using pcg_cuda::DpfBlk;
static const uint64_t P = 4611686018326724609ULL;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

struct Merge {
    int logN; NTTParameters<Data64> params; Modulus<Data64> modulus; Root<Data64>* fwd; Root<Data64>* inv;
    ntt_configuration<Data64> cf{}, ci{};
    static NTTParameters<Data64> make(int lg) {
        const int N = 1 << lg; const uint64_t psi = pcg_cuda::negacyclic_psi(P, N);
        NTTFactors<Data64> f(Modulus<Data64>(P), pcg_cuda::mod_mul_u64(psi, psi, P), psi);
        return NTTParameters<Data64>(lg, f, ReductionPolynomial::X_N_plus);
    }
    Merge(int lg, cudaStream_t s) : logN(lg), params(make(lg)), modulus(params.modulus) {
        auto f = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
        auto i = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);
        size_t b = (size_t)params.root_of_unity_size * sizeof(Root<Data64>);
        CK(cudaMalloc(&fwd, b)); CK(cudaMalloc(&inv, b));
        CK(cudaMemcpy(fwd, f.data(), b, cudaMemcpyHostToDevice)); CK(cudaMemcpy(inv, i.data(), b, cudaMemcpyHostToDevice));
        cf.n_power = lg; cf.ntt_type = FORWARD; cf.ntt_layout = PerPolynomial; cf.reduction_poly = ReductionPolynomial::X_N_plus;
        cf.zero_padding = false; cf.stream = s; ci = cf; ci.ntt_type = INVERSE; ci.mod_inverse = params.n_inv;
    }
    void mul(Data64* a, Data64* b, int batch, cudaStream_t s) {
        GPU_NTT_Inplace<Data64>(a, fwd, modulus, cf, batch); GPU_NTT_Inplace<Data64>(b, fwd, modulus, cf, batch);
        size_t tot = (size_t)batch << logN;
        fused4::pointwise_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(a, b, modulus, tot);
        GPU_INTT_Inplace<Data64>(a, inv, modulus, ci, batch);
    }
};

int main(int argc, char** argv) {
    int c = 4, t = 16, lg = 20, K = 4, iters = 3;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--c")) c = atoi(argv[++i]); else if (!strcmp(argv[i], "--t")) t = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logN")) lg = atoi(argv[++i]); else if (!strcmp(argv[i], "--blocks")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    }
    const int N = 1 << lg; int tlog = 0; while ((1 << tlog) < t) ++tlog;
    const int n = lg + 1 - tlog; const long D = 1L << n; const int B = t * t; const int gN = (int)(t * (D / 2));
    const int w = 1, levels = n, nk = 1;
    std::vector<DpfBlk> roots(B), dltsft((size_t)B * levels * nk);
    for (int b = 0; b < B; ++b) { roots[b].lo = 0x9e3779b97f4a7c15ULL * (b + 1); roots[b].hi = b + 1; }
    for (size_t i = 0; i < dltsft.size(); ++i) { dltsft[i].lo = 0x1234567ULL * (i + 1); dltsft[i].hi = i; }
    std::vector<uint8_t> master(16), round_keys((size_t)nk * 176); for (int i = 0; i < 16; i++) master[i] = i * 7 + 1;
    pcg_cuda::aes128_expand_key_host(master.data(), round_keys.data());
    auto exchange = [&](const DpfBlk*, DpfBlk* peer, size_t cnt) { for (size_t i = 0; i < cnt; i++) { peer[i].lo = 0; peer[i].hi = 0; } };
    std::vector<uint64_t> sumC(B), sumT(B), CW(B), g((size_t)gN);
    for (int b = 0; b < B; ++b) CW[b] = (0xABCDEFULL * (b + 1)) % P;

    cudaStream_t sn; CK(cudaStreamCreateWithFlags(&sn, cudaStreamNonBlocking));
    Merge mp(lg, sn);
    const int batch = c * c;                                  // e2e: 2 batched calls of batch c^2 per c^2 blocks -> 2 muls per block
    Data64 *a, *b; CK(cudaMalloc(&a, (size_t)batch * N * 8)); CK(cudaMalloc(&b, (size_t)batch * N * 8));
    CK(cudaMemset(a, 1, (size_t)batch * N * 8)); CK(cudaMemset(b, 2, (size_t)batch * N * 8));
    const int ntt_calls = (2 * K + batch - 1) / batch;        // 2K muls, in batch-c^2 calls

    auto dpf_blocks = [&]() {
        for (int blk = 0; blk < K; ++blk) {
            const DpfBlk* dl = pcg_cuda::dpf_gpu_batch_full_eval_device(B, n, w, roots.data(), dltsft.data(), round_keys.data(), 0, exchange, pcg_cuda::DpfGpuPrg::CHACHA8);
            pcg_cuda::dpf_out_sums(dl, B, D, P, sumC.data(), sumT.data());
            pcg_cuda::dpf_out_scatter_g(dl, B, D, t, 0, P, CW.data(), g.data(), gN);
        }
    };
    auto ntt_launch = [&]() { for (int i = 0; i < ntt_calls; ++i) mp.mul(a, b, batch, sn); };
    auto ms = [](auto x, auto y) { return std::chrono::duration<double, std::milli>(y - x).count(); };
    dpf_blocks(); ntt_launch(); CK(cudaDeviceSynchronize());   // warm-up
    double td = 0, tn = 0, tb = 0;
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now(); dpf_blocks(); CK(cudaDeviceSynchronize()); auto t1 = std::chrono::steady_clock::now();
        ntt_launch(); CK(cudaStreamSynchronize(sn)); auto t2 = std::chrono::steady_clock::now();
        ntt_launch(); dpf_blocks(); CK(cudaStreamSynchronize(sn)); CK(cudaDeviceSynchronize()); auto t3 = std::chrono::steady_clock::now();
        td += ms(t0, t1); tn += ms(t1, t2); tb += ms(t2, t3);
    }
    td /= iters; tn /= iters; tb /= iters;
    printf("c=%d t=%d logN=%d blocks=%d | GPU DPF %.1f ms, GPU merge NTT %.1f ms | serial sum %.1f | concurrent %.1f | concurrent/sum %.3f  concurrent/max %.3f\n",
           c, t, lg, K, td, tn, td + tn, tb, tb / (td + tn), tb / std::max(td, tn));
    return 0;
}
