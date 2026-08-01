// Isolation probe: does a SINGLE cyclic merge-NTT of length L, driven by the
// root table we build ourselves, match a CPU DFT with the same root?
// This separates "root table / cfg convention wrong" from "four-step
// composition wrong". No timing; correctness only.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/common/nttparameters.cuh"
#include "common/ntt_roots.h"

using namespace gpuntt;
using pcg_cuda::mod_mul_u64;
using pcg_cuda::mod_pow_u64;
using pcg_cuda::negacyclic_psi;
static const uint64_t Q = 4611686018326724609ULL;

static int lg2(int n) { int o = 0; while (n > 1) { n >>= 1; ++o; } return o; }
static int brev(int i, int b) { int o = 0; for (int k = 0; k < b; ++k) { o = (o << 1) | (i & 1); i >>= 1; } return o; }

// table[i] = base^{bitrev(i, log2(L/2))}, size L/2  (X_N_minus convention)
static std::vector<uint64_t> tbl(int L, uint64_t base) {
    const int half = L / 2, b = lg2(half);
    std::vector<uint64_t> t(half), o(half);
    uint64_t c = 1;
    for (int i = 0; i < half; ++i) { t[i] = c; c = mod_mul_u64(c, base, Q); }
    for (int i = 0; i < half; ++i) o[i] = t[brev(i, b)];
    return o;
}

int main(int argc, char** argv) {
    const int L = (argc > 1) ? atoi(argv[1]) : 1024;
    const int lg = lg2(L);
    const uint64_t psi = negacyclic_psi(Q, L);
    const uint64_t w = mod_mul_u64(psi, psi, Q);      // L-th root of unity

    std::vector<uint64_t> h(L);
    std::mt19937_64 rng(7);
    for (auto& v : h) v = rng() % Q;

    // CPU cyclic DFT: X[k] = sum_n x[n] w^{nk}
    std::vector<uint64_t> want(L, 0);
    for (int k = 0; k < L; ++k) {
        uint64_t acc = 0;
        for (int n = 0; n < L; ++n) {
            const uint64_t t = mod_mul_u64(h[n], mod_pow_u64(w, (uint64_t)n * k % L, Q), Q);
            acc = (acc + t) % Q;
        }
        want[k] = acc;
    }

    Data64 *d = nullptr, *dr = nullptr;
    cudaMalloc(&d, L * sizeof(Data64));
    cudaMemcpy(d, h.data(), L * sizeof(Data64), cudaMemcpyHostToDevice);
    auto rt = tbl(L, w);
    cudaMalloc(&dr, rt.size() * sizeof(Data64));
    cudaMemcpy(dr, rt.data(), rt.size() * sizeof(Data64), cudaMemcpyHostToDevice);

    ntt_configuration<Data64> cfg{};
    cfg.n_power = lg;
    cfg.ntt_type = FORWARD;
    cfg.ntt_layout = PerPolynomial;
    cfg.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg.zero_padding = false;
    cfg.mod_inverse = Ninverse<Data64>(1);
    cfg.stream = 0;
    GPU_NTT_Inplace<Data64>(d, dr, Modulus<Data64>(Q), cfg, 1);
    cudaDeviceSynchronize();

    std::vector<uint64_t> got(L);
    cudaMemcpy(got.data(), d, L * sizeof(Data64), cudaMemcpyDeviceToHost);

    int nat = 0, rev = 0;
    for (int i = 0; i < L; ++i) {
        if (got[i] == want[i]) ++nat;
        if (got[i] == want[brev(i, lg)]) ++rev;
    }
    std::printf("L=%d  natural-order matches %d/%d   bitrev-order matches %d/%d\n",
                L, nat, L, rev, L);
    std::printf("  want[0..3] = %llu %llu %llu %llu\n",
                (unsigned long long)want[0], (unsigned long long)want[1],
                (unsigned long long)want[2], (unsigned long long)want[3]);
    std::printf("  got [0..3] = %llu %llu %llu %llu\n",
                (unsigned long long)got[0], (unsigned long long)got[1],
                (unsigned long long)got[2], (unsigned long long)got[3]);
    std::printf("%s\n", nat == L ? "NATURAL" : (rev == L ? "BITREVERSED" : "NEITHER"));
    return 0;
}
