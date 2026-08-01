#include "bool_circuit/circuit.h"
#include "common/comm_trace.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace bool_circuit {

// =========================================================================
// TripleGen — Beaver AND triples from a single-direction FerretCOT
//
// For n triples we run 2n standard OTs (ALICE = sender, BOB = receiver):
//   OTs 0..n-1:   msgs = (r_i, r_i ^ a0_i), choice = b1_i   → shares of a0·b1
//   OTs n..2n-1:  msgs = (s_i, s_i ^ b0_i), choice = a1_i   → shares of b0·a1
//
// c0_i = a0_i & b0_i ^ r_i ^ s_i              (ALICE)
// c1_i = a1_i & b1_i ^ lsb(out_i) ^ lsb(out_{n+i})  (BOB)
// =========================================================================

TripleGen::TripleGen(int party, emp::NetIO* io)
    : party_(party), io_(io) {
    ios_[0] = io;
    ferret_ = new emp::FerretCOT<emp::NetIO>(party, 1, ios_);
}

TripleGen::~TripleGen() { delete ferret_; }

void TripleGen::generate(uint8_t* a, uint8_t* b, uint8_t* c, int64_t n) {
    if (n <= 0) return;

    emp::PRG prg;
    bool* a_bool = new bool[n];
    bool* b_bool = new bool[n];
    prg.random_bool(a_bool, n);
    prg.random_bool(b_bool, n);
    for (int64_t i = 0; i < n; ++i) { a[i] = a_bool[i]; b[i] = b_bool[i]; }

    if (party_ == emp::ALICE) {
        bool* r = new bool[n]; prg.random_bool(r, n);
        bool* s = new bool[n]; prg.random_bool(s, n);

        emp::block* m0 = new emp::block[2 * n];
        emp::block* m1 = new emp::block[2 * n];
        for (int64_t i = 0; i < n; ++i) {
            m0[i]     = emp::makeBlock(0, static_cast<uint64_t>(r[i]));
            m1[i]     = emp::makeBlock(0, static_cast<uint64_t>(r[i] ^ a[i]));
            m0[n + i] = emp::makeBlock(0, static_cast<uint64_t>(s[i]));
            m1[n + i] = emp::makeBlock(0, static_cast<uint64_t>(s[i] ^ b[i]));
        }

        ferret_->send(m0, m1, 2 * n);
        io_->flush();

        for (int64_t i = 0; i < n; ++i)
            c[i] = (a[i] & b[i]) ^ r[i] ^ s[i];

        delete[] m0; delete[] m1; delete[] r; delete[] s;

    } else {
        bool* choices = new bool[2 * n];
        for (int64_t i = 0; i < n; ++i) {
            choices[i]     = b[i];   // choice for cross term a_alice * b_bob
            choices[n + i] = a[i];   // choice for cross term b_alice * a_bob
        }

        emp::block* out = new emp::block[2 * n];
        ferret_->recv(out, choices, 2 * n);

        for (int64_t i = 0; i < n; ++i)
            c[i] = (a[i] & b[i])
                  ^ static_cast<uint8_t>(emp::getLSB(out[i]))
                  ^ static_cast<uint8_t>(emp::getLSB(out[n + i]));

        delete[] out; delete[] choices;
    }

    delete[] a_bool; delete[] b_bool;
}

// =========================================================================
// Gilboa OT-based Z_p multiplication
//
// ALICE (sender) holds x[i].  BOB (receiver) holds y[i].
// For each multiplication, decompose y into nbits bits.
// For each bit j: ALICE sends (r_j, r_j + x·2^j mod p), BOB selects with y_j.
// ALICE's share: p − Σ r_j.  BOB's share: Σ selected values.
// Sum: x·y mod p.
// =========================================================================

void TripleGen::zp_multiply(const uint64_t* my_vals, int count, uint64_t p,
                            uint64_t* my_shares) {
    if (count <= 0) return;

    int nbits = 0;
    { uint64_t tmp = p - 1; while (tmp) { ++nbits; tmp >>= 1; } }

    int64_t total_ots = static_cast<int64_t>(count) * nbits;

    if (party_ == emp::ALICE) {
        emp::PRG prg;
        emp::block* m0 = new emp::block[total_ots];
        emp::block* m1 = new emp::block[total_ots];
        uint64_t* r_sums = new uint64_t[count]();

        for (int i = 0; i < count; ++i) {
            uint64_t x = my_vals[i] % p;
            for (int j = 0; j < nbits; ++j) {
                uint64_t r;
                prg.random_data(&r, sizeof(r));
                r = r % p;
                int64_t idx = static_cast<int64_t>(i) * nbits + j;
                uint64_t x_shifted = x;
                for (int s = 0; s < j; ++s) x_shifted = (x_shifted * 2) % p;
                m0[idx] = emp::makeBlock(0, r);
                m1[idx] = emp::makeBlock(0, (r + x_shifted) % p);
                r_sums[i] = (r_sums[i] + r) % p;
            }
        }

        ferret_->send(m0, m1, total_ots);
        io_->flush();

        for (int i = 0; i < count; ++i)
            my_shares[i] = (p - r_sums[i]) % p;

        delete[] m0; delete[] m1; delete[] r_sums;

    } else {
        bool* choices = new bool[total_ots];
        for (int i = 0; i < count; ++i) {
            uint64_t y = my_vals[i] % p;
            for (int j = 0; j < nbits; ++j)
                choices[static_cast<int64_t>(i) * nbits + j] = (y >> j) & 1;
        }

        emp::block* out = new emp::block[total_ots];
        ferret_->recv(out, choices, total_ots);

        for (int i = 0; i < count; ++i) {
            uint64_t s = 0;
            for (int j = 0; j < nbits; ++j) {
                uint64_t v;
                std::memcpy(&v, &out[static_cast<int64_t>(i) * nbits + j],
                            sizeof(uint64_t));
                s = (s + v % p) % p;
            }
            my_shares[i] = s;
        }

        delete[] out; delete[] choices;
    }
}

// =========================================================================
// zp_triple — batched Z_p Beaver triples via two zp_multiply cross-terms.
//
//   r_σ, s_σ  sampled locally at random in [0,p).
//   rs_σ = r_σ·s_σ + share(r0·s1) + share(r1·s0)   (mod p)
// so (r0+r1)(s0+s1) = r0s0 + r1s1 + r0s1 + r1s0 = rs0 + rs1.
//
// Cross-term r0·s1: ALICE feeds r0, BOB feeds s1  (zp_multiply x=r0, y=s1).
// Cross-term r1·s0: ALICE feeds s0, BOB feeds r1  (zp_multiply x=s0, y=r1).
// =========================================================================
void TripleGen::zp_triple(int count, uint64_t p,
                          uint64_t* r, uint64_t* s, uint64_t* rs) {
    if (count <= 0) return;

    emp::PRG prg;
    for (int i = 0; i < count; ++i) {
        uint64_t a, b;
        prg.random_data(&a, sizeof(a));
        prg.random_data(&b, sizeof(b));
        r[i] = a % p;
        s[i] = b % p;
    }

    // Cross term 1: r0·s1  (ALICE holds r, BOB holds s).
    std::vector<uint64_t> cross1(count);
    // Cross term 2: r1·s0  (ALICE holds s, BOB holds r).
    std::vector<uint64_t> cross2(count);

    if (party_ == emp::ALICE) {
        zp_multiply(r, count, p, cross1.data());   // x = r0
        zp_multiply(s, count, p, cross2.data());   // x = s0
    } else {
        zp_multiply(s, count, p, cross1.data());   // y = s1
        zp_multiply(r, count, p, cross2.data());   // y = r1
    }

    for (int i = 0; i < count; ++i) {
        unsigned __int128 loc =
            (unsigned __int128)r[i] * (unsigned __int128)s[i];
        uint64_t local = (uint64_t)(loc % p);
        uint64_t acc = local;
        acc = (acc + cross1[i]) % p;
        acc = (acc + cross2[i]) % p;
        rs[i] = acc;
    }
}

// =========================================================================
// BoolCircuit — construction
// =========================================================================

int BoolCircuit::add_input(int owner) {
    int w = num_wires_++;
    wire_level_.push_back(0);
    gates_.push_back(Gate(INPUT, -1, -1, w, owner, 0));
    return w;
}

int BoolCircuit::add_xor(int w0, int w1) {
    int w = num_wires_++;
    int lv = std::max(wire_level_[w0], wire_level_[w1]);
    wire_level_.push_back(lv);
    gates_.push_back(Gate(OP_XOR, w0, w1, w, -1, lv));
    return w;
}

int BoolCircuit::add_not(int w0) {
    int w = num_wires_++;
    int lv = wire_level_[w0];
    wire_level_.push_back(lv);
    gates_.push_back(Gate(OP_NOT, w0, -1, w, -1, lv));
    return w;
}

int BoolCircuit::add_and(int w0, int w1) {
    int w = num_wires_++;
    int lv = std::max(wire_level_[w0], wire_level_[w1]) + 1;
    wire_level_.push_back(lv);
    gates_.push_back(Gate(OP_AND, w0, w1, w, -1, lv));
    return w;
}

void BoolCircuit::set_output(int wire) { output_wires_.push_back(wire); }

int BoolCircuit::num_and_gates() const {
    int cnt = 0;
    for (auto& g : gates_) if (g.type == OP_AND) ++cnt;
    return cnt;
}

// =========================================================================
// BoolCircuit — secure evaluation  (GMW with Beaver triples)
//
// Wire values are XOR-shared:  ⟨x⟩_0 ⊕ ⟨x⟩_1 = x.
//   INPUT:  owner secret-shares its bit with a random mask.
//   XOR:    ⟨z⟩_b = ⟨x⟩_b ⊕ ⟨y⟩_b                              (free)
//   NOT:    ⟨z⟩_ALICE = 1 ⊕ ⟨x⟩_ALICE,  ⟨z⟩_BOB = ⟨x⟩_BOB      (free)
//   AND:    open d = x ⊕ a, e = y ⊕ b, then
//           ⟨z⟩_ALICE = d·e ⊕ d·⟨b⟩ ⊕ e·⟨a⟩ ⊕ ⟨c⟩
//           ⟨z⟩_BOB   =       d·⟨b⟩ ⊕ e·⟨a⟩ ⊕ ⟨c⟩
//
// AND gates at the same depth level are batched into one communication round.
// =========================================================================

// Returns all wire shares (size = num_wires_).
std::vector<uint8_t>
BoolCircuit::evaluate_core(int party, emp::NetIO* io,
                           TripleGen& tg,
                           const std::vector<uint8_t>& my_inputs) const {
    const int party_idx = party - 1;   // ALICE(1)→0, BOB(2)→1

    // ── Generate all AND triples up front ──────────────────────────────
    int n_and = num_and_gates();
    std::vector<uint8_t> ta(n_and), tb(n_and), tc(n_and);
    if (n_and > 0) tg.generate(ta.data(), tb.data(), tc.data(), n_and);

    // ── Wire share storage ──────────────────────────────────────────────
    std::vector<uint8_t> wire(num_wires_, 0);

    // ── Input sharing ──────────────────────────────────────────────────
    //   Owner sends random mask r to peer; sets its share to x ⊕ r.
    //   Peer's share is r.
    //   Batched per-owner: all owner-0 masks first, then owner-1.
    emp::PRG prg;
    int my_inp_idx = 0;

    for (int owner = 0; owner < 2; ++owner) {
        std::vector<uint8_t> masks;
        std::vector<int>     wires;

        for (auto& g : gates_) {
            if (g.type != INPUT || g.owner != owner) continue;
            if (party_idx == owner) {
                uint8_t mask;
                bool tmp; prg.random_bool(&tmp, 1); mask = tmp;
                wire[g.out] = my_inputs[my_inp_idx++] ^ mask;
                masks.push_back(mask);
            } else {
                wires.push_back(g.out);
            }
        }

        if (party_idx == owner) {
            io->send_data(masks.data(), masks.size());
            io->flush();
        } else {
            std::vector<uint8_t> recv_masks(wires.size());
            io->recv_data(recv_masks.data(), recv_masks.size());
            for (size_t i = 0; i < wires.size(); ++i)
                wire[wires[i]] = recv_masks[i];
        }
    }

    // ── Group gates by level ────────────────────────────────────────────
    int max_level = 0;
    for (auto& g : gates_) max_level = std::max(max_level, g.level);

    std::vector<std::vector<size_t>> levels(max_level + 1);
    for (size_t i = 0; i < gates_.size(); ++i)
        levels[gates_[i].level].push_back(i);

    // ── Evaluate level by level ─────────────────────────────────────────
    int and_idx = 0;

    for (int lv = 0; lv <= max_level; ++lv) {
        // Collect AND gates at this level for batched d/e exchange.
        std::vector<size_t> and_gates;
        for (size_t gi : levels[lv])
            if (gates_[gi].type == OP_AND) and_gates.push_back(gi);

        // (a) Batch AND gates: compute d,e shares, exchange, compute outputs.
        if (!and_gates.empty()) {
            size_t n = and_gates.size();
            std::vector<uint8_t> d_shares(n), e_shares(n);
            for (size_t i = 0; i < n; ++i) {
                d_shares[i] = wire[gates_[and_gates[i]].in0] ^ ta[and_idx + i];
                e_shares[i] = wire[gates_[and_gates[i]].in1] ^ tb[and_idx + i];
            }

            // Two send_data calls coalesce into one TCP write (single flush),
            // so this is ONE round trip carrying 2n bytes per direction.
            std::vector<uint8_t> peer_d(n), peer_e(n);
            PCG_COMM_TRACE(io, "circuit_and", lv, static_cast<int>(n),
                           2 * n, 2 * n, {
                if (party == emp::ALICE) {
                    io->send_data(d_shares.data(), n);
                    io->send_data(e_shares.data(), n);
                    io->flush();
                    PCG_COMM_TRACE_MID();
                    io->recv_data(peer_d.data(), n);
                    io->recv_data(peer_e.data(), n);
                } else {
                    io->recv_data(peer_d.data(), n);
                    io->recv_data(peer_e.data(), n);
                    PCG_COMM_TRACE_MID();
                    io->send_data(d_shares.data(), n);
                    io->send_data(e_shares.data(), n);
                    io->flush();
                }
            });

            for (size_t i = 0; i < n; ++i) {
                uint8_t d = d_shares[i] ^ peer_d[i];
                uint8_t e = e_shares[i] ^ peer_e[i];
                uint8_t ai = ta[and_idx + i];
                uint8_t bi = tb[and_idx + i];
                uint8_t ci = tc[and_idx + i];

                if (party == emp::ALICE)
                    wire[gates_[and_gates[i]].out] = (d & e) ^ (d & bi) ^ (e & ai) ^ ci;
                else
                    wire[gates_[and_gates[i]].out] = (d & bi) ^ (e & ai) ^ ci;
            }
            and_idx += static_cast<int>(n);
        }

        // (b) Free gates (XOR, NOT) — local, processed AFTER AND gates at
        //     this level so they can read AND outputs.
        for (size_t gi : levels[lv]) {
            const Gate& g = gates_[gi];
            switch (g.type) {
            case OP_XOR:
                wire[g.out] = wire[g.in0] ^ wire[g.in1];
                break;
            case OP_NOT:
                wire[g.out] = (party == emp::ALICE) ? (1 ^ wire[g.in0]) : wire[g.in0];
                break;
            default: break;
            }
        }
    }

    return wire;
}

std::vector<uint8_t>
BoolCircuit::evaluate(int party, emp::NetIO* io,
                      TripleGen& tg,
                      const std::vector<uint8_t>& my_inputs) const {
    auto wire = evaluate_core(party, io, tg, my_inputs);

    size_t n_out = output_wires_.size();
    std::vector<uint8_t> my_out(n_out);
    for (size_t i = 0; i < n_out; ++i) my_out[i] = wire[output_wires_[i]];

    std::vector<uint8_t> peer_out(n_out);
    if (party == emp::ALICE) {
        io->send_data(my_out.data(), n_out);
        io->flush();
        io->recv_data(peer_out.data(), n_out);
    } else {
        io->recv_data(peer_out.data(), n_out);
        io->send_data(my_out.data(), n_out);
        io->flush();
    }

    std::vector<uint8_t> result(n_out);
    for (size_t i = 0; i < n_out; ++i) result[i] = my_out[i] ^ peer_out[i];
    return result;
}

std::vector<uint8_t>
BoolCircuit::evaluate_shared(int party, emp::NetIO* io,
                             TripleGen& tg,
                             const std::vector<uint8_t>& my_inputs) const {
    auto wire = evaluate_core(party, io, tg, my_inputs);

    size_t n_out = output_wires_.size();
    std::vector<uint8_t> shares(n_out);
    for (size_t i = 0; i < n_out; ++i) shares[i] = wire[output_wires_[i]];
    return shares;
}

}  // namespace bool_circuit
