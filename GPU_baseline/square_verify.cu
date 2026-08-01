// Correctness gate for the balanced ("square") four-step backend.
// Compares poly_mul_u64_gpuntt_square against the schoolbook negacyclic product
// computed in __int128 on the host. Any mismatch aborts; no timing is printed,
// because a wrong transform has no meaningful performance.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "common/poly_mul_cuda.h"

static const uint64_t kPrime = 4611686018326724609ULL;

static uint64_t addm(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t s = a + b; return s >= p ? s - p : s;
}
static uint64_t subm(uint64_t a, uint64_t b, uint64_t p) {
    return a >= b ? a - b : a + p - b;
}
static uint64_t mulm(uint64_t a, uint64_t b, uint64_t p) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % p);
}

// reference negacyclic convolution: c = a*b mod (X^N + 1)
static void ref_mul(std::vector<uint64_t>& c, const uint64_t* a, const uint64_t* b,
                    int N, uint64_t p) {
    c.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        if (!a[i]) continue;
        for (int j = 0; j < N; ++j) {
            const uint64_t t = mulm(a[i], b[j], p);
            const int k = i + j;
            if (k < N) c[k] = addm(c[k], t, p);
            else       c[k - N] = subm(c[k - N], t, p);
        }
    }
}

int main(int argc, char** argv) {
    // Small N only: the reference is O(N^2). The transform's index conventions
    // are N-independent, so passing at several small N validates the structure.
    std::vector<int> logs = {8, 9, 10, 11, 12};
    if (argc > 1) { logs.clear(); for (int i = 1; i < argc; ++i) logs.push_back(atoi(argv[i])); }

    std::mt19937_64 rng(12345);
    int failures = 0;
    for (int lg : logs) {
        const int N = 1 << lg;
        std::string why;
        if (!pcg_cuda::poly_mul_gpuntt_square_supported(kPrime, N, &why)) {
            std::printf("  logN=%2d SKIP (%s)\n", lg, why.c_str());
            continue;
        }
        for (int batch : {1, 3}) {
            std::vector<uint64_t> a(static_cast<size_t>(batch) * N),
                                  b(static_cast<size_t>(batch) * N),
                                  got(static_cast<size_t>(batch) * N);
            for (auto& v : a) v = rng() % kPrime;
            for (auto& v : b) v = rng() % kPrime;
            pcg_cuda::poly_mul_u64_gpuntt_square(got.data(), a.data(), b.data(),
                                                 batch, N, kPrime, nullptr);
            int bad = 0, first = -1, revfull = 0;
            auto brv = [](int i, int b) { int o = 0; for (int k = 0; k < b; ++k) { o = (o << 1) | (i & 1); i >>= 1; } return o; };
            for (int t = 0; t < batch; ++t) {
                std::vector<uint64_t> want;
                ref_mul(want, a.data() + static_cast<size_t>(t) * N,
                        b.data() + static_cast<size_t>(t) * N, N, kPrime);
                for (int i = 0; i < N; ++i) {
                    if (want[i] != got[static_cast<size_t>(t) * N + i]) {
                        if (first < 0) first = t * N + i;
                        ++bad;
                    }
                    if (want[brv(i, lg)] == got[static_cast<size_t>(t) * N + i]) ++revfull;
                }
            }
            std::printf("  logN=%2d batch=%d : %s (%d/%d mismatched, full-bitrev matches %d/%d%s)\n",
                        lg, batch,
                        bad == 0 ? "BITEXACT" : "FAIL", bad, batch * N, revfull, batch * N,
                        first >= 0 ? (", first at " + std::to_string(first)).c_str() : "");
            if (bad) ++failures;
        }
    }
    std::printf("%s\n", failures ? "GATE FAIL" : "GATE PASS");
    return failures ? 1 : 0;
}
