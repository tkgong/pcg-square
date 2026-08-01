#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/ffp.h"
#include "common/ntt.h"
#include "common/poly_mul_cuda.h"

namespace {

constexpr uint64_t PRIME = 4611686018326724609ULL;
using F = FFp<PRIME>;

enum class Backend { V1, GPUNTT_MERGE, GPUNTT_4STEP };

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::V1:           return "v1";
        case Backend::GPUNTT_MERGE: return "gpuntt-merge";
        case Backend::GPUNTT_4STEP: return "gpuntt-4step";
    }
    return "unknown";
}

bool backend_supported(Backend b, int N, std::string* reason) {
    switch (b) {
        case Backend::V1:           return pcg_cuda::poly_mul_supported(PRIME, N, reason);
        case Backend::GPUNTT_MERGE: return pcg_cuda::poly_mul_gpuntt_supported(PRIME, N, reason);
        case Backend::GPUNTT_4STEP: return pcg_cuda::poly_mul_gpuntt_4step_supported(PRIME, N, reason);
    }
    return false;
}

// A power-of-two N this backend definitely supports, used to probe availability
// (the 4-step backend only supports N >= 4096, so probing with 128 would wrongly
// skip it entirely).
int probe_N(Backend b) {
    return b == Backend::GPUNTT_4STEP ? 4096 : 128;
}

void call_backend(Backend b, uint64_t* out, const uint64_t* a, const uint64_t* bb,
                  int batch, int N, pcg_cuda::PolyMulStats* stats) {
    switch (b) {
        case Backend::V1:
            pcg_cuda::poly_mul_u64(out, a, bb, batch, N, PRIME, stats); return;
        case Backend::GPUNTT_MERGE:
            pcg_cuda::poly_mul_u64_gpuntt(out, a, bb, batch, N, PRIME, stats); return;
        case Backend::GPUNTT_4STEP:
            pcg_cuda::poly_mul_u64_gpuntt_4step(out, a, bb, batch, N, PRIME, stats); return;
    }
}

std::vector<uint64_t> make_pattern(int N, int batch, const std::string& pattern,
                                   uint64_t seed) {
    std::vector<uint64_t> out(static_cast<size_t>(N) * batch);
    std::mt19937_64 rng(seed);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < N; ++i) {
            uint64_t v = 0;
            if (pattern == "one") v = 1;
            else if (pattern == "impulse") v = (i == (b % N)) ? 1 : 0;
            else if (pattern == "max") v = PRIME - 1;
            else if (pattern == "random") v = rng() % PRIME;
            out[static_cast<size_t>(b) * N + i] = v;
        }
    }
    return out;
}

std::vector<uint64_t> cpu_reference(const std::vector<uint64_t>& a,
                                    const std::vector<uint64_t>& b,
                                    int N, int batch) {
    NTT<PRIME> ntt(N);
    std::vector<uint64_t> out(static_cast<size_t>(N) * batch);
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
    return out;
}

bool run_case(Backend backend, int N, int batch, const std::string& lhs_pattern,
              const std::string& rhs_pattern) {
    auto a = make_pattern(N, batch, lhs_pattern, 1234 + N + batch);
    auto b = make_pattern(N, batch, rhs_pattern, 5678 + 3 * N + batch);
    auto expected = cpu_reference(a, b, N, batch);
    std::vector<uint64_t> got(expected.size());

    pcg_cuda::PolyMulStats stats;
    call_backend(backend, got.data(), a.data(), b.data(), batch, N, &stats);

    if (got != expected) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != expected[i]) {
                std::cerr << "Mismatch [" << backend_name(backend) << "] N=" << N
                          << " batch=" << batch
                          << " case=" << lhs_pattern << "*" << rhs_pattern
                          << " index=" << i
                          << " expected=" << expected[i]
                          << " got=" << got[i] << "\n";
                break;
            }
        }
        return false;
    }
    std::cout << "PASS [" << backend_name(backend) << "] N=" << N << " batch=" << batch
              << " " << lhs_pattern << "*" << rhs_pattern
              << " device_ms=" << stats.device_ms << "\n";
    return true;
}

bool run_alias_case(Backend backend) {
    // 4-step only supports N >= 4096; use a supported N for the alias check.
    const int N = backend == Backend::GPUNTT_4STEP ? 4096 : 256;
    const int batch = 4;
    auto a = make_pattern(N, batch, "random", 111);
    auto b = make_pattern(N, batch, "random", 222);
    auto expected = cpu_reference(a, b, N, batch);
    call_backend(backend, a.data(), a.data(), b.data(), batch, N, nullptr);
    if (a != expected) {
        std::cerr << "Mismatch in alias case [" << backend_name(backend) << "]\n";
        return false;
    }
    std::cout << "PASS [" << backend_name(backend) << "] alias output==lhs\n";
    return true;
}

bool run_backend(Backend backend, const std::vector<int>& sizes) {
    std::string reason;
    if (!backend_supported(backend, probe_N(backend), &reason)) {
        std::cout << "backend " << backend_name(backend)
                  << " unavailable; skipping: " << reason << "\n";
        return true;  // not a failure — backend simply not applicable
    }
    bool ok = true;
    for (int N : sizes) {
        if (!backend_supported(backend, N, &reason)) continue;
        int batch = (N <= 1024) ? 4 : 2;
        ok = run_case(backend, N, batch, "zero", "zero") && ok;
        ok = run_case(backend, N, batch, "one", "one") && ok;
        ok = run_case(backend, N, batch, "impulse", "one") && ok;
        ok = run_case(backend, N, batch, "max", "max") && ok;
        ok = run_case(backend, N, batch, "random", "random") && ok;
    }
    ok = run_alias_case(backend) && ok;
    return ok;
}

}  // namespace

int main() {
    std::string reason;
    if (!pcg_cuda::is_cuda_available()) {
        std::cout << "CUDA unavailable; skipping\n";
        return 77;
    }

    bool ok = true;
    // v1 hand-rolled kernel is capped at N <= 32768.
    ok = run_backend(Backend::V1,
                     {128, 256, 512, 1024, 8192, 16384, 32768}) && ok;
    // GPU-NTT merge supports larger N — exercise up to 2^20.
    ok = run_backend(Backend::GPUNTT_MERGE,
                     {128, 256, 512, 1024, 8192, 16384, 32768,
                      1 << 18, 1 << 20}) && ok;
    // GPU-NTT 4-step only supports N in [4096, 2^24]; exercise 4096..2^20.
    ok = run_backend(Backend::GPUNTT_4STEP,
                     {4096, 8192, 16384, 32768, 65536,
                      1 << 18, 1 << 20}) && ok;
    return ok ? 0 : 1;
}
