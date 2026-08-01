// Kogge-Stone parallel-prefix adder for BoolCircuit.
//
// AND depth: 1 (initial generate) + ceil(log2(n)) (prefix steps) = O(log n).
// Gate count: n AND + n XOR (initial) + 2n*log(n) AND + n*log(n) XOR (prefix)
//             ≈ (2 log n + 1) * n AND gates total.
//
// For n=32: 6 communication rounds instead of 32 with ripple-carry.

#ifndef BOOL_CIRCUIT_ADDER_H__
#define BOOL_CIRCUIT_ADDER_H__

#include <vector>
#include "bool_circuit/circuit.h"

namespace bool_circuit {

// Build an n-bit Kogge-Stone adder.
//   a[0..n-1]: ALICE's input wires (LSB-first)
//   b[0..n-1]: BOB's input wires   (LSB-first)
// Returns (n+1) output wires (LSB-first): sum[0..n-1] and sum[n] = carry-out.
inline std::vector<int>
build_kogge_stone_adder(BoolCircuit& c, int n,
                        const std::vector<int>& a,
                        const std::vector<int>& b) {
    // Initial generate / propagate.
    std::vector<int> G(n), P(n), p_init(n);
    for (int i = 0; i < n; ++i) {
        G[i]      = c.add_and(a[i], b[i]);   // g_i = a_i & b_i
        P[i]      = c.add_xor(a[i], b[i]);   // p_i = a_i ^ b_i
        p_init[i] = P[i];
    }

    // Parallel prefix (Kogge-Stone): log2(n) steps, each with up to n
    // independent prefix operations.  Each prefix op uses 2 AND gates at
    // the same depth level, so +1 AND depth per step.
    //
    //   (G_i, P_i) = (G_i, P_i) ∘ (G_{i-d}, P_{i-d})
    //
    //   G_new = G_i ^ (P_i & G_{i-d})     [1 AND + 1 XOR]
    //   P_new = P_i & P_{i-d}              [1 AND]
    //
    // The XOR (instead of OR) is correct because G_i=1 implies P_i=0,
    // so G_i and P_i & G_{i-d} are never both 1.
    for (int d = 1; d < n; d <<= 1) {
        std::vector<int> G_new = G, P_new = P;
        for (int i = d; i < n; ++i) {
            int pg   = c.add_and(P[i], G[i - d]);
            G_new[i] = c.add_xor(G[i], pg);
            P_new[i] = c.add_and(P[i], P[i - d]);
        }
        G = G_new;
        P = P_new;
    }

    // Sum bits.  After prefix, G[i] = carry into position i+1.
    std::vector<int> sum(n + 1);
    sum[0] = p_init[0];                                  // no carry-in
    for (int i = 1; i < n; ++i)
        sum[i] = c.add_xor(p_init[i], G[i - 1]);        // p_i ^ c_i
    sum[n] = G[n - 1];                                   // carry-out

    return sum;
}

}  // namespace bool_circuit

#endif  // BOOL_CIRCUIT_ADDER_H__
