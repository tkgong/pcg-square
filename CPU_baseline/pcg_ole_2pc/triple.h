// Multiplication triple generation from Ring-OLE.
//
// From 2 ring-OLE instances over R_p = Z_p[X]/(X^N+1):
//   OLE 1: z_{0,1} + z_{1,1} = x_{0,1} · x_{1,1}
//   OLE 2: z_{0,2} + z_{1,2} = x_{0,2} · x_{1,2}
//
// Construct ring triple (a, b, c) with c = a · b over R_p:
//   a_0 = x_{0,1},  a_1 = x_{1,2}    →  a = a_0 + a_1 (unknown to both)
//   b_0 = x_{0,2},  b_1 = x_{1,1}    →  b = b_0 + b_1 (unknown to both)
//   c_0 = a_0·b_0 + z_{0,1} + z_{0,2}
//   c_1 = a_1·b_1 + z_{1,1} + z_{1,2}
//
// Correctness: c_0 + c_1 = a_0·b_0 + a_1·b_1 + (x_{0,1}·x_{1,1}) + (x_{0,2}·x_{1,2})
//            = a_0·b_0 + a_1·b_1 + a_0·b_1 + a_1·b_0 = (a_0+a_1)(b_0+b_1) = a·b.
//
// To extract N independent Z_p triples: NTT-evaluate the ring polynomials.
// Each NTT slot i gives (a_slot[i], b_slot[i], c_slot[i]) with
// c_slot[i] = a_slot[i] · b_slot[i] over Z_p.

#ifndef PCG_OLE_2PC_TRIPLE_H__
#define PCG_OLE_2PC_TRIPLE_H__

#include <cassert>
#include <iostream>
#include <vector>

#include "common/ffp.h"
#include "common/ntt.h"
#include "pcg_ole_2pc/pcg_ole.h"

namespace pcg_ole {

// One party's share of a Z_p multiplication triple.
template<uint64_t P>
struct TripleShare {
    FFp<P> a;   // share of a
    FFp<P> b;   // share of b
    FFp<P> c;   // share of c = a · b
};

// Generate N multiplication triples over Z_p from 2 ring-OLE instances.
//
// Runs PCG_OLE::gen_and_expand twice, then combines the outputs into
// ring triples and NTT-decomposes into N independent Z_p triples.
template<uint64_t P>
std::vector<TripleShare<P>>
generate_triples(int party, emp::NetIO* io, const PCGParams& params,
                 bool_circuit::TripleGen& tg,
                 const std::vector<Poly<P>>& public_a) {
    const int N = params.N;
    const std::string tag = "[P" + std::to_string(party) + "]";

    // ── Run 2 independent OLEs ──────────────────────────────────────────
    PCG_OLE<P> pcg(party, io, params);

    std::cout << tag << " Triple gen: running OLE 1...\n"; std::cout.flush();
    OLEOutput<P> ole1 = pcg.gen_and_expand(tg, public_a);

    std::cout << tag << " Triple gen: running OLE 2...\n"; std::cout.flush();
    OLEOutput<P> ole2 = pcg.gen_and_expand(tg, public_a);

    // ── Build ring triple shares ────────────────────────────────────────
    //   P0 (party=0): a_0 = x_{0,1},  b_0 = x_{0,2}
    //   P1 (party=1): a_1 = x_{1,2},  b_1 = x_{1,1}
    Poly<P> a_share(N), b_share(N), c_share(N);

    if (party == 0) {
        // a_0 = ole1.x,  b_0 = ole2.x
        a_share = ole1.x;
        b_share = ole2.x;
        // c_0 = a_0 · b_0 + z_{0,1} + z_{0,2}
        PCG_PROFILE_SCOPE("triple.local_ring_product");
        c_share = poly_add<P>(poly_add<P>(
                      poly_mul_backend<P>(a_share, b_share, N, params.poly_mul_backend),
                      ole1.z), ole2.z);
    } else {
        // a_1 = ole2.x,  b_1 = ole1.x
        a_share = ole2.x;
        b_share = ole1.x;
        // c_1 = a_1 · b_1 + z_{1,1} + z_{1,2}
        PCG_PROFILE_SCOPE("triple.local_ring_product");
        c_share = poly_add<P>(poly_add<P>(
                      poly_mul_backend<P>(a_share, b_share, N, params.poly_mul_backend),
                      ole1.z), ole2.z);
    }

    // ── NTT to extract N independent Z_p triples ────────────────────────
    // Each NTT slot is an independent evaluation point where
    // the ring multiplication becomes a scalar Z_p multiplication.
    assert((P - 1) % (2 * static_cast<uint64_t>(N)) == 0);
    NTT<P> ntt(N);

    std::vector<FFp<P>> a_ntt(a_share.begin(), a_share.end());
    std::vector<FFp<P>> b_ntt(b_share.begin(), b_share.end());
    std::vector<FFp<P>> c_ntt(c_share.begin(), c_share.end());

    ntt.forward(a_ntt.data());
    ntt.forward(b_ntt.data());
    ntt.forward(c_ntt.data());

    std::vector<TripleShare<P>> triples(N);
    for (int i = 0; i < N; ++i) {
        triples[i].a = a_ntt[i];
        triples[i].b = b_ntt[i];
        triples[i].c = c_ntt[i];
    }

    std::cout << tag << " Generated " << N << " Z_p multiplication triples.\n";
    return triples;
}

}  // namespace pcg_ole

#endif  // PCG_OLE_2PC_TRIPLE_H__
