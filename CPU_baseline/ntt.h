// Negacyclic NTT over FFp<P> for polynomial multiplication mod (X^N + 1).
//
// Requires P to be NTT-friendly: (P-1) divisible by 2N.
// Uses a primitive 2N-th root of unity ψ so that ψ^N ≡ -1 (mod P),
// folding the negacyclic reduction into the NTT itself — no 2x padding.
//
// Precomputes all twiddle factors at construction. Forward and inverse
// transforms are in-place with O(N log N) multiplications.

#ifndef COMMON_NTT_H__
#define COMMON_NTT_H__

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "common/ffp.h"

template<uint64_t P>
class NTT {
public:
    explicit NTT(int N) : N_(N) {
        assert(N > 0 && (N & (N - 1)) == 0);
        assert((P - 1) % (2 * static_cast<uint64_t>(N)) == 0);

        log_N_ = __builtin_ctz(static_cast<unsigned>(N));
        inv_N_ = FFp<P>(static_cast<uint64_t>(N)).inv();

        // Find primitive root g of Z_P*.
        uint64_t g = find_primitive_root();

        // ψ = g^{(P-1)/(2N)}: primitive 2N-th root of unity.
        // ψ^N = -1, so ψ handles negacyclic reduction.
        FFp<P> psi = pow(FFp<P>(g), (P - 1) / (2 * N));
        FFp<P> psi_inv = psi.inv();

        // Precompute twist factors ψ^i and ψ^{-i}.
        psi_pow_.resize(N);
        psi_inv_pow_.resize(N);
        psi_pow_[0] = psi_inv_pow_[0] = FFp<P>::one();
        for (int i = 1; i < N; ++i) {
            psi_pow_[i]     = psi_pow_[i - 1] * psi;
            psi_inv_pow_[i] = psi_inv_pow_[i - 1] * psi_inv;
        }

        // ω = ψ² : primitive N-th root of unity for the standard NTT.
        FFp<P> omega     = psi * psi;
        FFp<P> omega_inv = psi_inv * psi_inv;

        // Precompute forward/inverse roots: root[k] = ω^k for k = 0..N/2-1.
        int half = N / 2;
        root_fwd_.resize(half);
        root_inv_.resize(half);
        root_fwd_[0] = root_inv_[0] = FFp<P>::one();
        for (int i = 1; i < half; ++i) {
            root_fwd_[i] = root_fwd_[i - 1] * omega;
            root_inv_[i] = root_inv_[i - 1] * omega_inv;
        }
    }

    // In-place negacyclic forward NTT.
    //   Input:  coefficients a[0..N-1].
    //   Output: evaluations at ψ^{2·bitrev(i)+1} for i = 0..N-1.
    void forward(FFp<P>* a) const {
        // Pre-twist: a[i] *= ψ^i.
        for (int i = 0; i < N_; ++i) a[i] *= psi_pow_[i];
        ntt_core(a, root_fwd_);
    }

    // In-place negacyclic inverse NTT.
    //   Input:  NTT-domain values.
    //   Output: coefficients of the polynomial mod (X^N + 1).
    void inverse(FFp<P>* a) const {
        ntt_core(a, root_inv_);
        // Scale by N^{-1} and undo twist: a[i] *= N^{-1} · ψ^{-i}.
        for (int i = 0; i < N_; ++i) a[i] *= inv_N_ * psi_inv_pow_[i];
    }

    // c = a * b mod (X^N + 1).  a, b: length N.  c: length N (may alias a or b).
    void poly_mul(FFp<P>* c, const FFp<P>* a, const FFp<P>* b) const {
        std::vector<FFp<P>> fa(a, a + N_), fb(b, b + N_);
        forward(fa.data());
        forward(fb.data());
        for (int i = 0; i < N_; ++i) fa[i] *= fb[i];
        inverse(fa.data());
        std::copy(fa.begin(), fa.end(), c);
    }

    int N() const { return N_; }

private:
    int N_, log_N_;
    FFp<P> inv_N_;
    std::vector<FFp<P>> root_fwd_, root_inv_;
    std::vector<FFp<P>> psi_pow_, psi_inv_pow_;

    // Iterative radix-2 Cooley-Tukey NTT (standard, not negacyclic).
    // The negacyclic twist is handled by the caller via pre/post-twist.
    void ntt_core(FFp<P>* a, const std::vector<FFp<P>>& roots) const {
        // Bit-reverse permutation.
        for (int i = 1, j = 0; i < N_; ++i) {
            int bit = N_ >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }

        // Butterfly stages.
        for (int len = 2; len <= N_; len <<= 1) {
            int half = len >> 1;
            int step = N_ / len;     // root stride
            for (int i = 0; i < N_; i += len) {
                for (int j = 0; j < half; ++j) {
                    FFp<P> u = a[i + j];
                    FFp<P> v = a[i + j + half] * roots[j * step];
                    a[i + j]        = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }

    // Find the smallest primitive root of Z_P*.
    static uint64_t find_primitive_root() {
        uint64_t phi = P - 1;
        std::vector<uint64_t> factors;
        uint64_t x = phi;
        for (uint64_t d = 2; d * d <= x; ++d) {
            if (x % d != 0) continue;
            factors.push_back(d);
            while (x % d == 0) x /= d;
        }
        if (x > 1) factors.push_back(x);

        for (uint64_t g = 2; g < P; ++g) {
            bool ok = true;
            for (uint64_t q : factors) {
                if (pow(FFp<P>(g), phi / q) == FFp<P>::one()) {
                    ok = false;
                    break;
                }
            }
            if (ok) return g;
        }
        return 0;
    }

    static FFp<P> pow(FFp<P> base, uint64_t exp) {
        FFp<P> result = FFp<P>::one();
        while (exp > 0) {
            if (exp & 1) result *= base;
            base *= base;
            exp >>= 1;
        }
        return result;
    }
};

#endif  // COMMON_NTT_H__
