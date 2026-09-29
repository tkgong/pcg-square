// interfere_bench: silicon calibration/validation of the GPU-PIM contention model.
//
// A persistent "aggressor" kernel streams reads+writes over a buffer much larger
// than L2 (__ldcs/__stcs) on a few SMs, throttled by __nanosleep, while the
// victim -- the fused four-step NTT (full pipeline, or the SM lane alone) -- runs
// on the default stream. Each point reports the aggressor's achieved DRAM
// bandwidth and the victim's slowdown. A sleep-only aggressor with the same grid
// is the control for the SMs the aggressor occupies. The same aggressor
// bandwidths are replayed in Ramulator2 as a class-3 (AGG) host stream, and the
// two slowdown curves are compared.
//
//   ./interfere_bench --logN 22 --batch 16 --blocks 16 --iters 10
//
// Output CSV: gpu,victim,logN,batch,blocks,sleep_ns,mode,agg_GBps,agg_frac,victim_ms,slowdown

#include "fused4/fused4_ntt.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const uint64_t P = 4611686018326724609ULL;

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
    int lg = 22, batch = 16, blocks = 16, iters = 10, agg_mb = 2048;
    std::vector<unsigned> sleeps = {0, 100, 300, 1000, 3000, 10000, 30000};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--logN")) lg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch")) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks")) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--agg-mb")) agg_mb = atoi(argv[++i]);
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string gpu = prop.name;
    for (auto& ch : gpu) if (ch == ' ') ch = '_';
    int bus_bits = 0, mem_khz = 0;
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0);
    const double peak_GBps = 2.0 * mem_khz * 1e3 * (bus_bits / 8) / 1e9;   // DDR, nominal

    fused4::Plan plan(lg, P);
    const size_t total = static_cast<size_t>(batch) << lg;
    Data64 *x, *y, *tx, *ty;
    fused4::check(cudaMalloc(&x, total * 8), "x");
    fused4::check(cudaMalloc(&y, total * 8), "y");
    fused4::check(cudaMalloc(&tx, total * 8), "tx");
    fused4::check(cudaMalloc(&ty, total * 8), "ty");
    fill_kernel<<<(total + 255) / 256, 256>>>(x, total, 1);
    fill_kernel<<<(total + 255) / 256, 256>>>(y, total, 2);
    cudaDeviceSynchronize();
    Agg agg(static_cast<size_t>(agg_mb) << 20);   // per side; default 2 GiB >> L2

    auto victim_ms = [&](int lanes) {
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        plan.multiply(x, y, tx, ty, batch, lanes);            // warm-up
        cudaEventRecord(s, 0);
        for (int i = 0; i < iters; ++i) plan.multiply(x, y, tx, ty, batch, lanes);
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);                               // never a device-wide sync
        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s);
        cudaEventDestroy(e);
        return ms / iters / batch;
    };

    printf("# peak(nominal)=%.0f GB/s\n", peak_GBps);
    printf("gpu,victim,logN,batch,blocks,sleep_ns,mode,agg_GBps,agg_frac,victim_ms,slowdown\n");
    for (int lanes : {3, 1}) {
        const char* vname = lanes == 3 ? "f4_full" : "f4_sm";
        const double base = victim_ms(lanes);
        printf("%s,%s,%d,%d,0,-,none,0,0,%.4f,1.000\n", gpu.c_str(), vname, lg, batch, base);
        for (int mode : {1, 0, 2, 3}) {
            for (unsigned sl : sleeps) {
                if (mode == 1 && sl != 0) continue;       // one control point
                agg.start(blocks, sl, mode);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                const double v = victim_ms(lanes);
                const double bw = agg.stop_GBps();
                printf("%s,%s,%d,%d,%d,%u,%s,%.1f,%.3f,%.4f,%.3f\n", gpu.c_str(), vname, lg, batch, blocks,
                       sl, mode == 0 ? "stream" : mode == 2 ? "random" : mode == 3 ? "rowburst" : "sleep_only", bw, bw / peak_GBps, v, v / base);
                fflush(stdout);
            }
        }
    }
    // aggressor alone (no victim): its uncontended bandwidth per setting
    for (int mode : {0, 2, 3})
        for (unsigned sl : sleeps) {
            agg.start(blocks, sl, mode);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const double bw = agg.stop_GBps();
            printf("%s,none,%d,%d,%d,%u,%s_alone,%.1f,%.3f,0,0\n", gpu.c_str(), lg, batch, blocks, sl,
                   mode == 0 ? "stream" : mode == 2 ? "random" : "rowburst", bw, bw / peak_GBps);
        }
    return 0;
}
