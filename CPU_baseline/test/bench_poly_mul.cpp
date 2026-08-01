#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/ffp.h"
#include "common/ntt.h"

#ifdef PCG_ENABLE_CUDA
#include "common/poly_mul_cuda.h"
#endif

namespace {

constexpr uint64_t PRIME = 4611686018326724609ULL;
using F = FFp<PRIME>;

struct Args {
    int N = 128;
    int batch = 1;
    int iters = 20;
    int warmup = 5;
    bool csv_header = false;
};

void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --N <degree> --batch <count> [--iters n] [--warmup n] [--csv-header]\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) args.N = std::atoi(argv[++i]);
        else if (arg == "--batch" && i + 1 < argc) args.batch = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) args.iters = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) args.warmup = std::atoi(argv[++i]);
        else if (arg == "--csv-header") args.csv_header = true;
        else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.N <= 0 || args.batch <= 0 || args.iters <= 0 || args.warmup < 0) {
        usage(argv[0]);
        std::exit(1);
    }
    return args;
}

#ifdef PCG_ENABLE_CUDA
struct Stats {
    double mean_ms = 0.0;
    double std_ms = 0.0;
};

std::vector<uint64_t> random_polys(int N, int batch, uint64_t seed) {
    std::vector<uint64_t> out(static_cast<size_t>(N) * batch);
    std::mt19937_64 rng(seed);
    for (auto& x : out) x = rng() % PRIME;
    return out;
}

// NTT plan (twiddles, primitive-root search) is constructed once and reused
// across calls — apples-to-apples with the cached CUDA plans.
void cpu_poly_mul(const NTT<PRIME>& ntt, uint64_t* out,
                  const uint64_t* a, const uint64_t* b,
                  int N, int batch) {
    std::vector<F> aa(N), bb(N), cc(N);
    for (int k = 0; k < batch; ++k) {
        for (int i = 0; i < N; ++i) {
            aa[i] = F(a[static_cast<size_t>(k) * N + i]);
            bb[i] = F(b[static_cast<size_t>(k) * N + i]);
        }
        ntt.poly_mul(cc.data(), aa.data(), bb.data());
        for (int i = 0; i < N; ++i)
            out[static_cast<size_t>(k) * N + i] = cc[i].val();
    }
}

Stats summarize(const std::vector<double>& samples) {
    Stats s;
    if (samples.empty()) return s;
    s.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double var = 0.0;
    for (double x : samples) {
        double d = x - s.mean_ms;
        var += d * d;
    }
    s.std_ms = std::sqrt(var / samples.size());
    return s;
}

using GpuFn = void (*)(uint64_t*, const uint64_t*, const uint64_t*, int, int,
                       uint64_t, pcg_cuda::PolyMulStats*);

// Times one GPU backend. Returns false (and leaves api/dev/correct untouched)
// if the backend does not support this N.
struct GpuResult {
    bool ran = false;
    Stats api;
    Stats dev;
    bool correct = false;
};

GpuResult time_gpu(GpuFn fn, bool supported, const std::vector<uint64_t>& a,
                   const std::vector<uint64_t>& b,
                   const std::vector<uint64_t>& cpu_out,
                   int N, int batch, int warmup, int iters) {
    GpuResult r;
    if (!supported) return r;
    std::vector<uint64_t> gpu_out(a.size());

    for (int i = 0; i < warmup; ++i)
        fn(gpu_out.data(), a.data(), b.data(), batch, N, PRIME, nullptr);

    std::vector<double> api_samples, dev_samples;
    for (int i = 0; i < iters; ++i) {
        pcg_cuda::PolyMulStats stats;
        auto t0 = std::chrono::steady_clock::now();
        fn(gpu_out.data(), a.data(), b.data(), batch, N, PRIME, &stats);
        auto t1 = std::chrono::steady_clock::now();
        api_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        dev_samples.push_back(stats.device_ms);
    }
    r.ran = true;
    r.api = summarize(api_samples);
    r.dev = summarize(dev_samples);
    r.correct = (cpu_out == gpu_out);
    return r;
}

void emit_gpu(const GpuResult& r) {
    if (r.ran)
        std::cout << r.api.mean_ms << ',' << r.api.std_ms << ','
                  << r.dev.mean_ms << ',' << r.dev.std_ms << ','
                  << (r.correct ? 1 : 0) << ',';
    else
        std::cout << "-1,-1,-1,-1,-1,";  // backend not applicable at this N
}
#endif

std::string env_or_unknown(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string("unknown");
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.csv_header) {
        // gpuntt_* columns are the GPU-NTT "merge" backend; fourstep_* are the
        // GPU-NTT "4-step" backend (emits the -1 sentinel for N < 4096).
        std::cout << "device,N,batch,warmup,iters,cpu_mean_ms,cpu_std_ms,"
                  << "v1_api_mean_ms,v1_api_std_ms,v1_dev_mean_ms,v1_dev_std_ms,v1_correct,"
                  << "gpuntt_api_mean_ms,gpuntt_api_std_ms,gpuntt_dev_mean_ms,"
                  << "gpuntt_dev_std_ms,gpuntt_correct,"
                  << "fourstep_api_mean_ms,fourstep_api_std_ms,fourstep_dev_mean_ms,"
                  << "fourstep_dev_std_ms,fourstep_correct,"
                  << "gpu_model,cuda_runtime,compiler,build_flags,commit\n";
    }

#ifndef PCG_ENABLE_CUDA
    std::cout << "cuda_disabled," << args.N << ',' << args.batch << ','
              << args.warmup << ',' << args.iters
              << ",0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,unavailable,unavailable,"
              << __VERSION__ << ",unknown," << env_or_unknown("PCG_GIT_COMMIT") << "\n";
    return 77;
#else
    if (!pcg_cuda::is_cuda_available()) {
        std::cerr << "CUDA unavailable; skipping benchmark\n";
        return 77;
    }

    auto a = random_polys(args.N, args.batch, 0x12345678ULL);
    auto b = random_polys(args.N, args.batch, 0x87654321ULL);
    std::vector<uint64_t> cpu_out(a.size());

    // Cache the CPU NTT plan outside the timed region so its construction
    // (primitive-root search + twiddle precompute) is amortized across runs,
    // mirroring the cached CUDA plans.
    NTT<PRIME> cpu_plan(args.N);

    // PCG_BENCH_SKIP_CPU=1 skips the CPU reference (timing AND correctness
    // check) for device-only sweeps at sizes where the CPU pass takes minutes.
    // cpu_out is computed once (not timed) unless skipped; with skip on, the
    // *_correct columns compare GPU output against zeros and must be ignored.
    const bool skip_cpu = [] {
        const char* e = std::getenv("PCG_BENCH_SKIP_CPU");
        return e && e[0] == '1';
    }();

    Stats cpu;
    if (!skip_cpu) {
        for (int i = 0; i < args.warmup; ++i)
            cpu_poly_mul(cpu_plan, cpu_out.data(), a.data(), b.data(), args.N, args.batch);

        std::vector<double> cpu_samples;
        for (int i = 0; i < args.iters; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            cpu_poly_mul(cpu_plan, cpu_out.data(), a.data(), b.data(), args.N, args.batch);
            auto t1 = std::chrono::steady_clock::now();
            cpu_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        cpu = summarize(cpu_samples);
    } else {
        cpu.mean_ms = -1;
        cpu.std_ms = -1;
    }

    bool v1_ok = pcg_cuda::poly_mul_supported(PRIME, args.N, nullptr);
    bool gpuntt_ok = pcg_cuda::poly_mul_gpuntt_supported(PRIME, args.N, nullptr);
    bool fourstep_ok = pcg_cuda::poly_mul_gpuntt_4step_supported(PRIME, args.N, nullptr);

    GpuResult v1 = time_gpu(&pcg_cuda::poly_mul_u64, v1_ok, a, b, cpu_out,
                            args.N, args.batch, args.warmup, args.iters);
    GpuResult gpuntt = time_gpu(&pcg_cuda::poly_mul_u64_gpuntt, gpuntt_ok, a, b, cpu_out,
                                args.N, args.batch, args.warmup, args.iters);
    GpuResult fourstep = time_gpu(&pcg_cuda::poly_mul_u64_gpuntt_4step, fourstep_ok,
                                  a, b, cpu_out, args.N, args.batch, args.warmup, args.iters);

    std::cout << "cuda," << args.N << ',' << args.batch << ','
              << args.warmup << ',' << args.iters << ','
              << cpu.mean_ms << ',' << cpu.std_ms << ',';
    emit_gpu(v1);
    emit_gpu(gpuntt);
    emit_gpu(fourstep);
    std::cout << '"' << pcg_cuda::device_name() << '"' << ','
              << pcg_cuda::runtime_version() << ','
              << '"' << __VERSION__ << '"' << ','
              << (std::string(
#ifdef NDEBUG
                  "Release"
#else
                  "Debug"
#endif
              )) << ','
              << env_or_unknown("PCG_GIT_COMMIT") << "\n";

    bool correct = (!v1.ran || v1.correct) && (!gpuntt.ran || gpuntt.correct) &&
                   (!fourstep.ran || fourstep.correct);
    return correct ? 0 : 2;
#endif
}
