// GPU-NTT merge forward NTT at logN 20..28 (GPU-NTT's built-in primes; kernel time is prime-independent):
// ms per transform, ps per element per global pass, kernels per transform; plus a contiguous-row reference
// (the same bytes transformed as rows of 2^14 elements, PerPolynomial layout, GPU-NTT's 2-kernel schedule).
//   ./merge_scale_bench --logN 20,22,24,25,26,27,28 --iters 5
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"
using namespace gpuntt;
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }
struct Fwd {
    int lg; NTTParameters<Data64> params; Modulus<Data64> modulus; Root<Data64>* fwd = nullptr; ntt_configuration<Data64> cfg{};
    explicit Fwd(int l) : lg(l), params(l, ReductionPolynomial::X_N_plus), modulus(params.modulus) {
        auto f = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
        const size_t bytes = static_cast<size_t>(params.root_of_unity_size) * sizeof(Root<Data64>);
        ck(cudaMalloc(&fwd, bytes), "malloc fwd"); ck(cudaMemcpy(fwd, f.data(), bytes, cudaMemcpyHostToDevice), "cp fwd");
        cfg.n_power = l; cfg.ntt_type = FORWARD; cfg.ntt_layout = PerPolynomial; cfg.reduction_poly = ReductionPolynomial::X_N_plus;
        cfg.zero_padding = false; cfg.stream = 0;
    }
    ~Fwd() { cudaFree(fwd); }
    void run(Data64* x, int batch) const { GPU_NTT_Inplace<Data64>(x, fwd, modulus, cfg, batch); }
};
static std::vector<int> parse(const char* s) { std::vector<int> v; for (const char* p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } return v; }
int main(int argc, char** argv) {
    std::vector<int> lgs = {20, 22, 24, 25, 26, 27, 28}; int iters = 5; bool ref = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--logN") && i + 1 < argc) lgs = parse(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noref")) ref = false;
    }
    printf("logN,batch,total_MB,merge_ms_per_transform,merge_ps_per_elem,merge_kernels,ref14_ms_same_bytes,ref14_ps_per_elem,ref14_kernels\n");
    for (int lg : lgs) {
        const int batch = lg <= 26 ? (1 << (26 - lg)) * 2 : (lg == 27 ? 1 : 1);   // 1 GB of data up to 2^26, then one polynomial
        const size_t total = static_cast<size_t>(batch) << lg;
        Data64* x; ck(cudaMalloc(&x, total * 8), "malloc x"); ck(cudaMemset(x, 1, total * 8), "memset");
        Fwd m(lg); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        m.run(x, batch); ck(cudaDeviceSynchronize(), "warm");
        cudaEventRecord(a); for (int i = 0; i < iters; ++i) m.run(x, batch); cudaEventRecord(b); ck(cudaEventSynchronize(b), "sync");
        float ms; cudaEventElapsedTime(&ms, a, b); const double t_m = ms / iters / batch;
        const int km = (int) CreateForwardNTTKernel<Data64>()[lg].size();
        double t_r = 0; int kr = 0;
        if (ref) {   // same bytes as 2^14-element rows
            const int rows = (int) (total >> 14); Fwd r(14); kr = (int) CreateForwardNTTKernel<Data64>()[14].size();
            r.run(x, rows); ck(cudaDeviceSynchronize(), "warm r");
            cudaEventRecord(a); for (int i = 0; i < iters; ++i) r.run(x, rows); cudaEventRecord(b); ck(cudaEventSynchronize(b), "sync r");
            cudaEventElapsedTime(&ms, a, b); t_r = ms / iters / batch;
        }
        const double n = (double) (1ull << lg);
        printf("%d,%d,%.0f,%.4f,%.1f,%d,%.4f,%.1f,%d\n", lg, batch, total * 8 / 1048576.0, t_m, t_m * 1e9 / n / km, km, t_r, kr ? t_r * 1e9 / n / kr : 0.0, kr);
        fflush(stdout); cudaFree(x);
    }
    return 0;
}
