// Host-side helpers to derive negacyclic-NTT roots of unity for a 62-bit
// NTT-friendly prime. Shared by the CUDA poly-mul backends so the root
// derivation lives in one place.
//
// For the ring R_p = Z_p[X]/(X^N + 1) we need a primitive 2N-th root of
// unity psi (psi^N = -1) and the N-th root omega = psi^2. These exist iff
// (p - 1) % (2N) == 0.

#ifndef COMMON_NTT_ROOTS_H__
#define COMMON_NTT_ROOTS_H__

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pcg_cuda {

inline uint64_t mod_mul_u64(uint64_t a, uint64_t b, uint64_t p) {
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % p);
}

inline uint64_t mod_pow_u64(uint64_t base, uint64_t exp, uint64_t p) {
    uint64_t out = 1;
    base %= p;
    while (exp) {
        if (exp & 1) out = mod_mul_u64(out, base, p);
        base = mod_mul_u64(base, base, p);
        exp >>= 1;
    }
    return out;
}

inline uint64_t mod_inv_u64(uint64_t x, uint64_t p) {
    return mod_pow_u64(x, p - 2, p);  // Fermat: x^(p-2) ≡ x^{-1} (mod p), p prime
}

// Smallest primitive root g of Z_p^* (p prime), via factoring p-1.
inline uint64_t find_primitive_root(uint64_t p) {
    const uint64_t phi = p - 1;
    uint64_t x = phi;
    std::vector<uint64_t> factors;
    for (uint64_t d = 2; d * d <= x; ++d) {
        if (x % d != 0) continue;
        factors.push_back(d);
        while (x % d == 0) x /= d;
    }
    if (x > 1) factors.push_back(x);

    for (uint64_t g = 2; g < p; ++g) {
        bool ok = true;
        for (uint64_t q : factors) {
            if (mod_pow_u64(g, phi / q, p) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
    throw std::runtime_error("ntt_roots: failed to find primitive root");
}

// psi: a primitive 2N-th root of unity (psi^N = -1, negacyclic).
inline uint64_t negacyclic_psi(uint64_t p, int N) {
    if ((p - 1) % (2ull * static_cast<uint64_t>(N)) != 0)
        throw std::runtime_error("ntt_roots: prime is not NTT-friendly for this N");
    const uint64_t g = find_primitive_root(p);
    return mod_pow_u64(g, (p - 1) / (2ull * static_cast<uint64_t>(N)), p);
}

}  // namespace pcg_cuda

#endif  // COMMON_NTT_ROOTS_H__
