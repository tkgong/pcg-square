// interfere_ntt: DRAM-interference sensitivity of three NTT poly-mul implementations
// on silicon, with no model of their access patterns: GPU-NTT merge (SOTA), the
// fused four-step on GPU-NTT kernels with its SM lane only (transposes offloaded,
// the DRU design) and with the transposes on the SMs. Each victim runs next to a
// persistent aggressor (interfere_bench's kernels): a streaming aggressor, an
// SPU-like row-burst aggressor (random 8 KB chunk, contiguous), and a sleep-only
// control with the same grid (SM occupancy). net = victim(stream|rowburst) / victim(control).
//
//   ./interfere_ntt --logN 22 --batch 16 --blocks 8,16,32 --iters 10
//
// CSV: gpu,victim,logN,batch,blocks,sleep_ns,mode,agg_GBps,victim_ms,slowdown_raw
#include "fused4/fused4_ntt.cuh"
#include "fused4/fused4_gpuntt.cuh"
#include "gpuntt/common/nttparameters.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
using namespace gpuntt;
static const uint64_t P = 4611686018326724609ULL;

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

__global__ void fill_kernel(Data64* x, size_t total, uint64_t seed) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) x[i] = (seed + 0x9e3779b97f4a7c15ULL * (i + 1)) % P;
}

// mode 0: stream (read 16 B, write 16 B per step); mode 1: sleep only (control);
// mode 2: random rows -- same 16 B read+write but at hashed addresses, so nearly
// every access opens a new DRAM row (row-conflict pressure, like the SPU's
// all-bank activates closing the GPU's rows) rather than streaming row hits;
// mode 3: random rows, bursty -- each block jumps to a random 8 KB chunk and
// reads+writes it contiguously (the SPU's pattern: ~43 columns per activate),
// so row switches are frequent but L2/TLB pollution per byte is low
__global__ void aggressor(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16,
                          volatile int* stop, unsigned sleep_ns, int mode,
                          unsigned long long* bytes) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    __shared__ int s_stop;
    // one thread per block polls the device-memory flag; everyone else reads shared
    for (;;) {
        if (threadIdx.x == 0) s_stop = *stop;
        __syncthreads();
        if (s_stop) break;
        if (mode == 0) {
#pragma unroll 4
            for (int k = 0; k < 16; ++k) {
                uint4 v = __ldcs(src + i);
                __stcs(dst + i, v);
                i += stride;
                if (i >= n16) i -= n16;
            }
            local += 16 * 32;
        } else if (mode == 2) {
#pragma unroll 4
            for (int k = 0; k < 16; ++k) {
                const size_t h = (i * 0x9E3779B97F4A7C15ULL + k * 0xBF58476D1CE4E5B9ULL) % n16;
                uint4 v = __ldcs(src + h);
                __stcs(dst + h, v);
                i += stride;
                if (i >= n16) i -= n16;
            }
            local += 16 * 32;
        }
        else if (mode == 3) {
            const size_t nchunk = n16 / blockDim.x;
            for (int k = 0; k < 16; ++k) {
                const size_t c = ((static_cast<size_t>(blockIdx.x) + 1) * 0x9E3779B97F4A7C15ULL +
                                  (local + k) * 0xBF58476D1CE4E5B9ULL) % nchunk;
                const size_t h = c * blockDim.x + threadIdx.x;
                uint4 v = __ldcs(src + h);
                __stcs(dst + h, v);
            }
            local += 16 * 32;
        }
        if (sleep_ns) __nanosleep(sleep_ns);
        else if (mode == 1) __nanosleep(1000);
        __syncthreads();
    }
    atomicAdd(bytes, local);
}

struct Agg {
    uint4 *src = nullptr, *dst = nullptr;
    size_t n16 = 0;
    int* d_stop = nullptr;
    cudaStream_t ctl;
    unsigned long long* d_bytes = nullptr;
    cudaStream_t s;
    std::chrono::steady_clock::time_point t0;
    explicit Agg(size_t bytes) {
        n16 = bytes / 16;
        fused4::check(cudaMalloc(&src, n16 * 16), "agg src");
        fused4::check(cudaMalloc(&dst, n16 * 16), "agg dst");
        cudaMemset(src, 1, n16 * 16);
        fused4::check(cudaMalloc(&d_stop, sizeof(int)), "stop flag");
        fused4::check(cudaStreamCreateWithFlags(&ctl, cudaStreamNonBlocking), "ctl stream");
        fused4::check(cudaMalloc(&d_bytes, sizeof(unsigned long long)), "bytes");
        fused4::check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    }
    void start(int blocks, unsigned sleep_ns, int mode) {
        cudaMemsetAsync(d_stop, 0, sizeof(int), s);
        cudaMemsetAsync(d_bytes, 0, sizeof(unsigned long long), s);
        aggressor<<<blocks, 512, 0, s>>>(src, dst, n16, d_stop, sleep_ns, mode, d_bytes);
        fused4::check(cudaGetLastError(), "aggressor launch");
        t0 = std::chrono::steady_clock::now();
    }
    double stop_GBps() {   // stop, return achieved GB/s over the run
        static const int one = 1;
        cudaMemcpyAsync(d_stop, &one, sizeof(int), cudaMemcpyHostToDevice, ctl);
        fused4::check(cudaStreamSynchronize(s), "agg sync");
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        unsigned long long b = 0;
        cudaMemcpy(&b, d_bytes, sizeof(b), cudaMemcpyDeviceToHost);
        return b / sec / 1e9;
    }
};


int main(int argc, char** argv) {
    int lg = 22, batch = 16, iters = 10;
    std::vector<int> blocks_list = {8, 16, 32};
    std::vector<unsigned> sleeps = {0, 10000, 30000};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--logN")) lg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch")) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks")) {
            blocks_list.clear();
            for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) blocks_list.push_back(atoi(t));
        }
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string gpu = prop.name;
    for (auto& ch : gpu) if (ch == ' ') ch = '_';
    MergePlan mp(lg);
    fused4g::PlanG pg(lg, P);
    const size_t total = static_cast<size_t>(batch) << lg;
    Data64 *x, *y, *tx, *ty, *x0, *y0;
    fused4::check(cudaMalloc(&x, total * 8), "x");
    fused4::check(cudaMalloc(&y, total * 8), "y");
    fused4::check(cudaMalloc(&tx, total * 8), "tx");
    fused4::check(cudaMalloc(&ty, total * 8), "ty");
    fused4::check(cudaMalloc(&x0, total * 8), "x0");
    fused4::check(cudaMalloc(&y0, total * 8), "y0");
    fill_kernel<<<(total + 255) / 256, 256>>>(x0, total, 1);
    fill_kernel<<<(total + 255) / 256, 256>>>(y0, total, 2);
    cudaDeviceSynchronize();
    Agg agg(static_cast<size_t>(1) << 30);
    auto victim_ms = [&](int v) {
        auto once = [&]() {
            cudaMemcpyAsync(x, x0, total * 8, cudaMemcpyDeviceToDevice, 0);   // same work every call
            cudaMemcpyAsync(y, y0, total * 8, cudaMemcpyDeviceToDevice, 0);
            if (v == 0) mp.multiply(x, y, batch);
            else pg.multiply(x, y, tx, ty, batch, v == 1 ? 1 : 3);
        };
        cudaEvent_t s, e, c0, c1;
        cudaEventCreate(&s); cudaEventCreate(&e); cudaEventCreate(&c0); cudaEventCreate(&c1);
        once();
        float ms = 0, cp = 0;
        for (int i = 0; i < iters; ++i) {
            cudaEventRecord(c0, 0);
            cudaMemcpyAsync(x, x0, total * 8, cudaMemcpyDeviceToDevice, 0);
            cudaMemcpyAsync(y, y0, total * 8, cudaMemcpyDeviceToDevice, 0);
            cudaEventRecord(c1, 0);
            if (v == 0) mp.multiply(x, y, batch);
            else pg.multiply(x, y, tx, ty, batch, v == 1 ? 1 : 3);
            cudaEventRecord(e, 0);
            cudaEventSynchronize(e);
            float a = 0; cudaEventElapsedTime(&a, c1, e); ms += a;
        }
        cudaEventDestroy(s); cudaEventDestroy(e); cudaEventDestroy(c0); cudaEventDestroy(c1);
        return ms / iters / batch;
    };
    const char* names[3] = {"merge", "f4g_sm", "f4g_full"};
    printf("gpu,victim,logN,batch,blocks,sleep_ns,mode,agg_GBps,victim_ms,slowdown_raw\n");
    for (int v = 0; v < 3; ++v) {
        const double base = victim_ms(v);
        printf("%s,%s,%d,%d,0,-,none,0,%.4f,1.000\n", gpu.c_str(), names[v], lg, batch, base);
        for (int b : blocks_list)
            for (int mode : {1, 0, 3})
                for (unsigned sl : sleeps) {
                    if (mode == 1 && sl != 0) continue;
                    agg.start(b, sl, mode);
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    const double t = victim_ms(v);
                    const double bw = agg.stop_GBps();
                    printf("%s,%s,%d,%d,%d,%u,%s,%.1f,%.4f,%.3f\n", gpu.c_str(), names[v], lg, batch, b, sl,
                           mode == 0 ? "stream" : mode == 3 ? "rowburst" : "sleep_only", bw, t, t / base);
                    fflush(stdout);
                }
    }
    return 0;
}
