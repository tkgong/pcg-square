// GPU-NTT merge backend (HEonGPU's NTT core, X_N_plus negacyclic), host-pointer interface identical to
// poly_mul_u64_gpuntt_square: copy in, 2 forward merge NTTs, pointwise, 1 inverse, copy out. Per-N plan cached.
#include "common/poly_mul_cuda.h"
#include "common/ntt_roots.h"
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/common/nttparameters.cuh"
using namespace gpuntt;
namespace pcg_cuda {
namespace {
void chk(cudaError_t e, const char* w) { if (e != cudaSuccess) throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e)); }
__global__ void pw_kernel(Data64* __restrict__ a, const Data64* __restrict__ b, Modulus<Data64> mod, size_t total) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) a[i] = OPERATOR_GPU<Data64>::mult(a[i], b[i], mod);
}
struct MergePlan {
    int logN; uint64_t p; NTTParameters<Data64> params; Modulus<Data64> modulus;
    Root<Data64>* fwd = nullptr; Root<Data64>* inv = nullptr; Data64* d_a = nullptr; Data64* d_b = nullptr; size_t cap = 0;
    ntt_configuration<Data64> cfg_fwd{}, cfg_inv{};
    static NTTParameters<Data64> make(uint64_t p, int lg) {
        const int N = 1 << lg; const uint64_t psi = negacyclic_psi(p, N); const uint64_t omega = mod_mul_u64(psi, psi, p);
        NTTFactors<Data64> f(Modulus<Data64>(p), omega, psi);
        return NTTParameters<Data64>(lg, f, ReductionPolynomial::X_N_plus);
    }
    MergePlan(uint64_t p_, int lg) : logN(lg), p(p_), params(make(p_, lg)), modulus(params.modulus) {
        auto f = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
        auto i = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);
        const size_t bytes = static_cast<size_t>(params.root_of_unity_size) * sizeof(Root<Data64>);
        chk(cudaMalloc(&fwd, bytes), "malloc fwd"); chk(cudaMalloc(&inv, bytes), "malloc inv");
        chk(cudaMemcpy(fwd, f.data(), bytes, cudaMemcpyHostToDevice), "copy fwd"); chk(cudaMemcpy(inv, i.data(), bytes, cudaMemcpyHostToDevice), "copy inv");
        cfg_fwd.n_power = lg; cfg_fwd.ntt_type = FORWARD; cfg_fwd.ntt_layout = PerPolynomial; cfg_fwd.reduction_poly = ReductionPolynomial::X_N_plus;
        cfg_fwd.zero_padding = false; cfg_fwd.stream = 0; cfg_inv = cfg_fwd; cfg_inv.ntt_type = INVERSE; cfg_inv.mod_inverse = params.n_inv;
    }
    void reserve(size_t n) { if (n <= cap) return; if (d_a) { cudaFree(d_a); cudaFree(d_b); } chk(cudaMalloc(&d_a, n * 8), "malloc a"); chk(cudaMalloc(&d_b, n * 8), "malloc b"); cap = n; }
    void multiply(uint64_t* out, const uint64_t* a, const uint64_t* b, int batch, PolyMulStats* stats) {
        const size_t total = static_cast<size_t>(batch) << logN; reserve(total); const size_t bytes = total * 8;
        chk(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "copy a"); chk(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "copy b");
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
        GPU_NTT_Inplace<Data64>(d_a, fwd, modulus, cfg_fwd, batch);
        GPU_NTT_Inplace<Data64>(d_b, fwd, modulus, cfg_fwd, batch);
        pw_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(d_a, d_b, modulus, total);
        GPU_INTT_Inplace<Data64>(d_a, inv, modulus, cfg_inv, batch);
        cudaEventRecord(e); chk(cudaEventSynchronize(e), "ntt"); float ms = 0; cudaEventElapsedTime(&ms, s, e); cudaEventDestroy(s); cudaEventDestroy(e);
        if (stats) stats->device_ms = ms;
        chk(cudaMemcpy(out, d_a, bytes, cudaMemcpyDeviceToHost), "copy out");
    }
};
std::mutex g_mu; std::map<std::pair<uint64_t, int>, std::unique_ptr<MergePlan>> g_plans;
MergePlan& plan_for(uint64_t p, int N) {
    int lg = 0; while ((1 << lg) < N) ++lg;
    std::lock_guard<std::mutex> lock(g_mu); auto& e = g_plans[{p, lg}]; if (!e) e.reset(new MergePlan(p, lg)); return *e;
}
}  // namespace
void poly_mul_u64_gpuntt_merge(uint64_t* out, const uint64_t* a, const uint64_t* b, int batch, int N, uint64_t prime, PolyMulStats* stats) {
    plan_for(prime, N).multiply(out, a, b, batch, stats);
}
}  // namespace pcg_cuda
