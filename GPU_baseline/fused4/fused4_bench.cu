// fused4_bench: fused four-step (SM lane + DRU lane) vs GPU-NTT merge, device-resident.
//
// Per (logN, batch) it reports, per negacyclic poly-mul (2 forward + pointwise +
// 1 inverse, both operands transformed, exactly the baseline's work unit):
//   merge   GPU-NTT merge backend (X_N_plus), 10 kernels, no transposes
//   f4_full fused four-step with its transposes executed on the SMs
//           (= a GPU-only fused four-step, no DRU)
//   f4_sm   fused four-step SM lane only: 7 kernels (the SM timeline when the
//           6 transposes run on the channel-level DRUs, fully overlapped)
//   f4_dru  the 6 transposes alone, as a 32x33 shared-memory tile transpose on
//           the GPU (the data volume the DRUs absorb)
//   f4g_full / f4g_sm  the same fused four-step with every sub-transform on
//           GPU-NTT's own merge kernels (fused4_gpuntt.cuh; logN 20..24 only):
//           transposes on the SMs / SM lane only (transposes on the DRU)
// and checks the fused result bit-exactly against merge on the same inputs.
// --selftest additionally checks fused4 against a CPU schoolbook product.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_XX -I.. -I$GPUNTT_INC fused4_bench.cu \
//        -L$GPUNTT_LIB -lntt-1.0 -o fused4_bench
// Run:
//   ./fused4_bench --logN 20,21,22,23,24 --batch 4,16,64 --iters 5 [--selftest]

#include "fused4/fused4_ntt.cuh"
#include "fused4/fused4_gpuntt.cuh"

#include "gpuntt/common/nttparameters.cuh"

#include <algorithm>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace gpuntt;

static const uint64_t P = 4611686018326724609ULL;   // 2^62 - 3*2^25 + 1

__global__ void fill_kernel(Data64* x, size_t total, uint64_t seed, uint64_t p) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL * (i + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    x[i] = z % p;
}

__global__ void diff_kernel(const Data64* a, const Data64* b, size_t total,
                            unsigned long long* bad) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total && a[i] != b[i]) atomicAdd(bad, 1ULL);
}

static void fill(Data64* x, size_t total, uint64_t seed) {
    fill_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(x, total, seed, P);
    fused4::check(cudaGetLastError(), "fill_kernel");
}

static unsigned long long count_diff(const Data64* a, const Data64* b, size_t total) {
    unsigned long long* d = nullptr;
    unsigned long long h = 0;
    fused4::check(cudaMalloc(&d, sizeof(*d)), "cudaMalloc diff");
    fused4::check(cudaMemset(d, 0, sizeof(*d)), "cudaMemset diff");
    diff_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(a, b, total, d);
    fused4::check(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost), "cudaMemcpy diff");
    cudaFree(d);
    return h;
}

// ---- GPU-NTT merge reference (same setup as common/poly_mul_gpuntt.cu) ----
struct MergePlan {
    int logN;
    NTTParameters<Data64> params;
    Modulus<Data64> modulus;
    Root<Data64>* fwd = nullptr;
    Root<Data64>* inv = nullptr;
    ntt_configuration<Data64> cfg_fwd{}, cfg_inv{};

    static NTTParameters<Data64> make(int lg) {
        const int N = 1 << lg;
        const uint64_t psi = pcg_cuda::negacyclic_psi(P, N);
        const uint64_t omega = pcg_cuda::mod_mul_u64(psi, psi, P);
        NTTFactors<Data64> f(Modulus<Data64>(P), omega, psi);
        return NTTParameters<Data64>(lg, f, ReductionPolynomial::X_N_plus);
    }
    explicit MergePlan(int lg) : logN(lg), params(make(lg)), modulus(params.modulus) {
        auto f = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
        auto i = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);
        const size_t bytes = static_cast<size_t>(params.root_of_unity_size) * sizeof(Root<Data64>);
        fused4::check(cudaMalloc(&fwd, bytes), "cudaMalloc merge fwd");
        fused4::check(cudaMalloc(&inv, bytes), "cudaMalloc merge inv");
        fused4::check(cudaMemcpy(fwd, f.data(), bytes, cudaMemcpyHostToDevice), "memcpy fwd");
        fused4::check(cudaMemcpy(inv, i.data(), bytes, cudaMemcpyHostToDevice), "memcpy inv");
        cfg_fwd.n_power = lg;
        cfg_fwd.ntt_type = FORWARD;
        cfg_fwd.ntt_layout = PerPolynomial;
        cfg_fwd.reduction_poly = ReductionPolynomial::X_N_plus;
        cfg_fwd.zero_padding = false;
        cfg_fwd.stream = 0;
        cfg_inv = cfg_fwd;
        cfg_inv.ntt_type = INVERSE;
        cfg_inv.mod_inverse = params.n_inv;
    }
    ~MergePlan() { cudaFree(fwd); cudaFree(inv); }

    void multiply(Data64* a, Data64* b, int batch) const {
        GPU_NTT_Inplace<Data64>(a, fwd, modulus, cfg_fwd, batch);
        GPU_NTT_Inplace<Data64>(b, fwd, modulus, cfg_fwd, batch);
        const size_t total = static_cast<size_t>(batch) << logN;
        fused4::pointwise_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
            a, b, modulus, total);
        GPU_INTT_Inplace<Data64>(a, inv, modulus, cfg_inv, batch);
    }
};

template <typename F>
static double time_ms(F&& fn, int iters) {
    fn();                                            // warm-up
    fused4::check(cudaDeviceSynchronize(), "warm-up");
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(e);
    fused4::check(cudaEventSynchronize(e), "timing");
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms / iters;
}

static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

// CPU schoolbook negacyclic product, for the self-test.
static std::vector<uint64_t> schoolbook(const std::vector<uint64_t>& a,
                                        const std::vector<uint64_t>& b) {
    const size_t N = a.size();
    std::vector<uint64_t> c(N, 0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
            const uint64_t t = pcg_cuda::mod_mul_u64(a[i], b[j], P);
            const size_t k = i + j;
            if (k < N) c[k] = (c[k] + t) % P;
            else       c[k - N] = (c[k - N] + P - t) % P;
        }
    return c;
}

static bool selftest() {
    bool ok = true;
    for (int lg : {8, 9, 10, 11}) {
        const int N = 1 << lg, batch = 2;
        const size_t total = static_cast<size_t>(batch) * N;
        Data64 *a, *b, *ta, *tb;
        cudaMalloc(&a, total * 8); cudaMalloc(&b, total * 8);
        cudaMalloc(&ta, total * 8); cudaMalloc(&tb, total * 8);
        fill(a, total, 11); fill(b, total, 22);
        std::vector<uint64_t> ha(total), hb(total), hc(total);
        cudaMemcpy(ha.data(), a, total * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), b, total * 8, cudaMemcpyDeviceToHost);
        fused4::Plan plan(lg, P);
        plan.multiply(a, b, ta, tb, batch);
        fused4::check(cudaMemcpy(hc.data(), a, total * 8, cudaMemcpyDeviceToHost), "selftest copy");
        bool good = true;
        for (int q = 0; q < batch && good; ++q) {
            std::vector<uint64_t> x(ha.begin() + q * N, ha.begin() + (q + 1) * N);
            std::vector<uint64_t> y(hb.begin() + q * N, hb.begin() + (q + 1) * N);
            auto ref = schoolbook(x, y);
            good = std::equal(ref.begin(), ref.end(), hc.begin() + q * N);
        }
        printf("selftest logN=%d vs CPU schoolbook: %s\n", lg, good ? "BITEXACT" : "MISMATCH");
        ok &= good;
        cudaFree(a); cudaFree(b); cudaFree(ta); cudaFree(tb);
    }
    return ok;
}

int main(int argc, char** argv) {
    std::vector<int> logNs = {20, 21, 22, 23, 24}, batches = {4, 16, 64};
    int iters = 5;
    bool do_self = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--logN") && i + 1 < argc) logNs = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batches = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--selftest")) do_self = true;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string gpu = prop.name;
    for (auto& ch : gpu) if (ch == ',' || ch == ' ') ch = '_';

    if (do_self && !selftest()) { fprintf(stderr, "SELFTEST FAILED\n"); return 2; }

    printf("gpu,logN,batch,merge_ms_mul,f4_full_ms_mul,f4_sm_ms_mul,f4_dru_ms_mul,"
           "merge_over_f4sm,merge_over_f4full,check,f4g_full_ms_mul,f4g_sm_ms_mul,"
           "merge_over_f4gsm,merge_over_f4gfull,check_g\n");
    for (int lg : logNs) {
        fused4::Plan plan(lg, P);
        MergePlan merge(lg);
        std::unique_ptr<fused4g::PlanG> plang;
        if (lg >= 20 && lg <= 24) plang.reset(new fused4g::PlanG(lg, P));
        for (int batch : batches) {
            const size_t total = static_cast<size_t>(batch) << lg;
            size_t free_b = 0, tot_b = 0;
            cudaMemGetInfo(&free_b, &tot_b);
            if (4 * total * sizeof(Data64) + (256u << 20) > free_b) {
                printf("%s,%d,%d,,,,,,,SKIP_MEM\n", gpu.c_str(), lg, batch);
                continue;
            }
            Data64 *x, *y, *tx, *ty;
            fused4::check(cudaMalloc(&x, total * 8), "cudaMalloc x");
            fused4::check(cudaMalloc(&y, total * 8), "cudaMalloc y");
            fused4::check(cudaMalloc(&tx, total * 8), "cudaMalloc tx");
            fused4::check(cudaMalloc(&ty, total * 8), "cudaMalloc ty");

            // correctness: fused4 in x, merge in tx, same inputs
            fill(x, total, 1234 + lg); fill(y, total, 5678 + lg);
            plan.multiply(x, y, tx, ty, batch, 3);
            fill(tx, total, 1234 + lg); fill(ty, total, 5678 + lg);
            merge.multiply(tx, ty, batch);
            fused4::check(cudaDeviceSynchronize(), "correctness run");
            const unsigned long long bad = count_diff(x, tx, total);
            unsigned long long bad_g = ~0ULL;
            if (plang) {    // GPU-NTT-kernel variant vs merge on the same inputs
                fill(x, total, 1234 + lg); fill(y, total, 5678 + lg);
                plang->multiply(x, y, tx, ty, batch, 3);
                Data64* keep = x;
                fill(tx, total, 1234 + lg); fill(ty, total, 5678 + lg);
                merge.multiply(tx, ty, batch);
                fused4::check(cudaDeviceSynchronize(), "correctness run g");
                bad_g = count_diff(keep, tx, total);
            }

            // timing (values are irrelevant to the op count; buffers are reused)
            const double t_merge = time_ms([&] { merge.multiply(x, y, batch); }, iters);
            const double t_full = time_ms([&] { plan.multiply(x, y, tx, ty, batch, 3); }, iters);
            const double t_sm = time_ms([&] { plan.multiply(x, y, tx, ty, batch, 1); }, iters);
            const double t_dru = time_ms([&] { plan.multiply(x, y, tx, ty, batch, 2); }, iters);
            double t_gfull = 0, t_gsm = 0;
            if (plang) {
                t_gfull = time_ms([&] { plang->multiply(x, y, tx, ty, batch, 3); }, iters);
                t_gsm = time_ms([&] { plang->multiply(x, y, tx, ty, batch, 1); }, iters);
            }

            const std::string chk = bad == 0 ? "BITEXACT" : "MISMATCH_" + std::to_string(bad);
            const std::string chk_g = !plang ? "NA" : (bad_g == 0 ? "BITEXACT" : "MISMATCH_" + std::to_string(bad_g));
            printf("%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%s,%.4f,%.4f,%.3f,%.3f,%s\n", gpu.c_str(), lg, batch,
                   t_merge / batch, t_full / batch, t_sm / batch, t_dru / batch,
                   t_merge / t_sm, t_merge / t_full, chk.c_str(),
                   t_gfull / batch, t_gsm / batch,
                   plang ? t_merge / t_gsm : 0.0, plang ? t_merge / t_gfull : 0.0, chk_g.c_str());
            fflush(stdout);
            cudaFree(x); cudaFree(y); cudaFree(tx); cudaFree(ty);
        }
    }
    return 0;
}
