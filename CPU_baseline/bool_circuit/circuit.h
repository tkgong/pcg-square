// GMW-style secure two-party Boolean circuit evaluation.
//
// Beaver AND triples are generated from FerretCOT (single direction: ALICE
// sends, BOB receives). XOR and NOT gates are free (local computation).
// AND gates are batched by circuit depth to minimize communication rounds.

#ifndef BOOL_CIRCUIT_CIRCUIT_H__
#define BOOL_CIRCUIT_CIRCUIT_H__

#include <cstdint>
#include <vector>

#include <emp-tool/emp-tool.h>
#include <emp-ot/ferret/ferret_cot.h>

namespace bool_circuit {

// ---- Beaver triple generation from FerretCOT ----------------------------
//
// Generates random AND triples (a, b, c) where (a0^a1)&(b0^b1) = c0^c1.
// Uses a single FerretCOT instance (ALICE=sender, BOB=receiver) and 2n
// standard OTs per n triples.
class TripleGen {
public:
    // party: emp::ALICE (=1) or emp::BOB (=2).
    TripleGen(int party, emp::NetIO* io);
    ~TripleGen();

    TripleGen(const TripleGen&) = delete;
    TripleGen& operator=(const TripleGen&) = delete;

    // Fill a[0..n), b[0..n), c[0..n) with this party's shares.
    void generate(uint8_t* a, uint8_t* b, uint8_t* c, int64_t n);

    // Gilboa OT-based Z_p multiplication.
    // ALICE holds x[i], BOB holds y[i], produces additive shares of x[i]·y[i] mod p.
    // my_vals: this party's inputs.  my_shares: output additive shares.
    void zp_multiply(const uint64_t* my_vals, int count, uint64_t p,
                     uint64_t* my_shares);

    // Batched Z_p Beaver triples. Fills this party's additive shares r[i],
    // s[i], rs[i] with rs = r*s mod p reconstructed across parties:
    //   (r0+r1)(s0+s1) = rs0 + rs1  (mod p).
    // Built from two zp_multiply cross-terms + the local product; r_σ, s_σ
    // are sampled locally at random. Input-independent — batch with payload OTs.
    // NOTE (crypto review): standard Beaver-from-two-OT-multiplies construction.
    void zp_triple(int count, uint64_t p,
                   uint64_t* r, uint64_t* s, uint64_t* rs);

private:
    int party_;
    emp::NetIO* io_;
    emp::NetIO* ios_[1];   // kept alive for FerretCOT (stores pointer to it)
    emp::FerretCOT<emp::NetIO>* ferret_;
};

// ---- Boolean circuit definition + evaluation ----------------------------

class BoolCircuit {
public:
    enum GateType { INPUT, OP_XOR, OP_NOT, OP_AND };

    // Circuit construction — returns the output wire index.
    // owner: 0 = first party (ALICE), 1 = second party (BOB).
    int add_input(int owner);
    int add_xor(int w0, int w1);
    int add_not(int w);
    int add_and(int w0, int w1);

    void set_output(int wire);

    int num_wires()     const { return num_wires_; }
    int num_and_gates() const;

    // Secure evaluation.
    //   party:     emp::ALICE or emp::BOB.
    //   io:        network channel (same as TripleGen's).
    //   tg:        pre-built TripleGen for AND triples.
    //   my_inputs: this party's input bits, in add_input order.
    // Returns reconstructed plaintext output bits.
    std::vector<uint8_t> evaluate(int party, emp::NetIO* io,
                                  TripleGen& tg,
                                  const std::vector<uint8_t>& my_inputs) const;

    // Like evaluate(), but returns this party's XOR share of the output
    // wires WITHOUT reconstruction.  No output-phase communication.
    std::vector<uint8_t> evaluate_shared(int party, emp::NetIO* io,
                                         TripleGen& tg,
                                         const std::vector<uint8_t>& my_inputs) const;

private:
    struct Gate {
        GateType type;
        int in0;
        int in1;
        int out;
        int owner;
        int level;

        Gate(GateType t, int i0, int i1, int o, int ow, int lv)
            : type(t), in0(i0), in1(i1), out(o), owner(ow), level(lv) {}
    };

    std::vector<uint8_t> evaluate_core(int party, emp::NetIO* io,
                                        TripleGen& tg,
                                        const std::vector<uint8_t>& my_inputs) const;

    std::vector<Gate> gates_;
    std::vector<int>  wire_level_;
    int num_wires_ = 0;
    std::vector<int> output_wires_;
};

}  // namespace bool_circuit

#endif  // BOOL_CIRCUIT_CIRCUIT_H__
