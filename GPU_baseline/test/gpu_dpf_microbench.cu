#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef FUSES_MATMUL
#define FUSES_MATMUL 0
#endif
#ifndef MM
#define MM 1
#endif

#include "dpf_base/dpf.h"
#include "dpf_gpu/dpf/dpf_hybrid.cu"

namespace {

struct Args {
    int entries = 64;
    int batch = 16;
    int warmup = 3;
    int iters = 10;
    int prf = DUMMY;
    bool csv_header = false;
    bool skip_validation = false;
};

struct Stats {
    double mean_ms = 0.0;
    double std_ms = 0.0;
};

void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --entries <pow2> --batch <n> [--prf dummy|aes|salsa|chacha]"
              << " [--warmup n] [--iters n] [--csv-header] [--skip-validation]\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--entries" && i + 1 < argc) args.entries = std::atoi(argv[++i]);
        else if (arg == "--batch" && i + 1 < argc) args.batch = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) args.warmup = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) args.iters = std::atoi(argv[++i]);
        else if (arg == "--skip-validation") args.skip_validation = true;
        else if (arg == "--csv-header") args.csv_header = true;
        else if (arg == "--prf" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "dummy") args.prf = DUMMY;
            else if (p == "aes") args.prf = AES128;
            else if (p == "salsa") args.prf = SALSA;
            else if (p == "chacha") args.prf = CHACHA;
            else {
                usage(argv[0]);
                std::exit(1);
            }
        } else {
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.entries <= 1 || (args.entries & (args.entries - 1)) != 0 ||
        args.batch <= 0 || args.warmup < 0 || args.iters <= 0) {
        usage(argv[0]);
        std::exit(1);
    }
    if (args.entries < Z) {
        std::cerr << "GPU-DPF hybrid requires at least " << Z
                  << " table entries; requested " << args.entries << "\n";
        std::exit(77);
    }
    return args;
}

const char* prf_name(int prf) {
    if (prf == DUMMY) return "dummy";
    if (prf == AES128) return "aes128";
    if (prf == SALSA) return "salsa";
    if (prf == CHACHA) return "chacha";
    return "unknown";
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

std::vector<SeedsCodewordsFlat>
gen_codewords(int entries, int batch, int prf, SeedsCodewordsFlatGPU** cw_gpu) {
    std::vector<SeedsCodewordsFlat> cw_cpu(batch);
    std::vector<SeedsCodewordsFlatGPU> cw_intermediate(batch);
    for (int i = 0; i < batch; ++i) {
        std::mt19937 gen(1000 + i);
        int alpha = (17 + 31 * i) & (entries - 1);
        uint128_t beta = static_cast<uint128_t>(1 + i);
        SeedsCodewords* s = GenerateSeedsAndCodewordsLog(alpha, beta, entries, gen, prf);
        FlattenCodewords(s, 0, &cw_cpu[i]);
        cw_intermediate[i] = SeedsCodewordsFlatGPUFromCPU(cw_cpu[i]);
        // GPU-DPF's upstream FreeSeedsCodewords uses free() for new'd objects, so
        // this short-lived benchmark intentionally lets this tiny setup memory go.
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(cw_gpu),
                          sizeof(SeedsCodewordsFlatGPU) * batch));
    CUDA_CHECK(cudaMemcpy(*cw_gpu, cw_intermediate.data(),
                          sizeof(SeedsCodewordsFlatGPU) * batch,
                          cudaMemcpyHostToDevice));
    return cw_cpu;
}

template <int Prf>
void run_dpf(SeedsCodewordsFlatGPU* cw_gpu, uint128_t_gpu* out,
             int batch, int entries, cudaStream_t stream) {
    dpf_hybrid<Prf>(cw_gpu, out, nullptr, batch, entries, stream);
}

void run_dpf_dynamic(int prf, SeedsCodewordsFlatGPU* cw_gpu, uint128_t_gpu* out,
                     int batch, int entries, cudaStream_t stream) {
    if (prf == DUMMY) run_dpf<DUMMY>(cw_gpu, out, batch, entries, stream);
    else if (prf == AES128) run_dpf<AES128>(cw_gpu, out, batch, entries, stream);
    else if (prf == SALSA) run_dpf<SALSA20>(cw_gpu, out, batch, entries, stream);
    else if (prf == CHACHA) run_dpf<CHACHA20>(cw_gpu, out, batch, entries, stream);
    else assert(false);
}

bool validate(const std::vector<SeedsCodewordsFlat>& cw_cpu,
              const std::vector<uint128_t_gpu>& gpu,
              int batch, int entries, int prf) {
    int log_entries = static_cast<int>(std::log2(static_cast<double>(entries)));
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < entries; ++j) {
            int permuted = static_cast<int>(brev_cpu(static_cast<uint32_t>(j)) >>
                                            (32 - log_entries));
            uint128_t expected = EvaluateFlat(&cw_cpu[b], j, prf);
            uint128_t got = uint128_from_gpu(gpu[static_cast<size_t>(b) * entries + permuted]);
            if (got != expected) {
                std::cerr << "GPU-DPF mismatch batch=" << b
                          << " index=" << j
                          << " permuted=" << permuted << "\n";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "CUDA device unavailable; skipping GPU-DPF benchmark\n";
        return 77;
    }

    if (args.csv_header) {
        std::cout << "device,entries,batch,prf,warmup,iters,mean_ms,std_ms,"
                  << "validated,ordering,note\n";
    }

    SeedsCodewordsFlatGPU* cw_gpu = nullptr;
    auto cw_cpu = gen_codewords(args.entries, args.batch, args.prf, &cw_gpu);
    uint128_t_gpu* out_gpu = nullptr;
    size_t out_count = static_cast<size_t>(args.entries) * args.batch;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&out_gpu),
                          sizeof(uint128_t_gpu) * out_count));
    CUDA_CHECK(cudaMemset(out_gpu, 0, sizeof(uint128_t_gpu) * out_count));
    dpf_hybrid_initialize(args.batch, args.entries);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    for (int i = 0; i < args.warmup; ++i) {
        run_dpf_dynamic(args.prf, cw_gpu, out_gpu, args.batch, args.entries, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<double> samples;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < args.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        run_dpf_dynamic(args.prf, cw_gpu, out_gpu, args.batch, args.entries, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    std::vector<uint128_t_gpu> out_cpu(out_count);
    CUDA_CHECK(cudaMemcpy(out_cpu.data(), out_gpu, sizeof(uint128_t_gpu) * out_count,
                          cudaMemcpyDeviceToHost));
    bool valid = args.skip_validation ||
                 validate(cw_cpu, out_cpu, args.batch, args.entries, args.prf);
    Stats stats = summarize(samples);

    std::cout << "gpu_dpf_hybrid," << args.entries << ',' << args.batch << ','
              << prf_name(args.prf) << ',' << args.warmup << ',' << args.iters << ','
              << stats.mean_ms << ',' << stats.std_ms << ','
              << (valid ? 1 : 0)
              << ",bit_reversed_full_expansion,"
              << '"' << "separate binary-DPF microbenchmark; not integrated into HalfTreeDPF" << '"'
              << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    dpf_hybrid_deinitialize();
    cudaFree(out_gpu);
    cudaFree(cw_gpu);
    return valid ? 0 : 2;
}
