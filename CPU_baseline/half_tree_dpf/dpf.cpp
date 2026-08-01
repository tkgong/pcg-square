#include "half_tree_dpf/dpf.h"
#include "bool_circuit/circuit.h"
#include "common/comm_trace.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <emp-tool/utils/aes.h>
#include <emp-tool/utils/prg.h>

#ifdef PCG_ENABLE_CUDA
#include "common/aes_gpu.h"
#include "common/dpf_gpu.h"
#include "common/poly_mul_cuda.h"  // is_cuda_available()
#endif

namespace half_tree_dpf {

const Block ZERO_BLOCK = emp::zero_block;

// Fixed public seed so both parties derive the same W (simulates F_Rand).
static const Block F_RAND_SEED =
    emp::makeBlock(0xdeadbeefcafe0001ULL, 0x0123456789abcdefULL);

bool block_eq(const Block& a, const Block& b) noexcept {
    return std::memcmp(&a, &b, sizeof(Block)) == 0;
}

std::string hex(const Block& b) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    const auto* p = reinterpret_cast<const uint8_t*>(&b);
    for (int i = 0; i < 16; ++i) ss << std::setw(2) << static_cast<unsigned>(p[i]);
    return ss.str();
}

// =========================================================================
// Construction — create the tree PRG (AES or ChaCha8)
// =========================================================================

HalfTreeDPF::HalfTreeDPF(int party, emp::NetIO* io, int n, int w,
                         const Block& master_key, PRGType prg_type)
    : party_(party), n_(n), w_(w), io_(io),
      master_key_(master_key), prg_type_(prg_type) {
    const int num_keys = (1 << w) - 1;

    switch (prg_type) {
    case PRGType::AES:
        prg_.reset(new AESTreePRG(num_keys, master_key));
        break;
    case PRGType::CHACHA8:
        prg_.reset(new ChaCha8TreePRG(num_keys));
        break;
    }
}

// =========================================================================
// F_DeltaShiftShare — real two-party computation using Boolean circuits
//
// For each level i and child k ∈ [0, m-2]:
//   indicator_{i,k} = ∏_{c=0}^{w-1} (α_{bit(i,c)} ⊕ k̄_c)   [w-1 AND gates]
//   DltSft_{i,k}    = indicator_{i,k} · Δ                     [128 AND gates]
//
// α and Δ remain secret (XOR-shared throughout). Only lsb(Δ)=1 is public.
// Total AND depth: w (indicator tree) + 1 (Δ multiply) = w+1.
// =========================================================================

DeltaShiftShareResult
HalfTreeDPF::F_DeltaShiftShare(uint64_t alpha_share, Block delta_share,
                                bool_circuit::TripleGen& tg) {
    const int m      = 1 << w_;
    const int levels = n_ / w_;
    const int n_k    = m - 1;

    // Force lsb(Δ) = 1 without communication: P0 sets lsb=1, P1 sets lsb=0.
    if (party_ == 0)
        reinterpret_cast<uint8_t*>(&delta_share)[0] |= 1u;
    else
        reinterpret_cast<uint8_t*>(&delta_share)[0] &= static_cast<uint8_t>(~1u);

    // ── Build the Boolean circuit ─────────────────────────────────────────
    bool_circuit::BoolCircuit circuit;

    // Inputs: each party provides n bits of ⟨α⟩ + 128 bits of ⟨Δ⟩.
    std::vector<int> alice_alpha(n_), bob_alpha(n_);
    for (int j = 0; j < n_; ++j) alice_alpha[j] = circuit.add_input(0);
    for (int j = 0; j < n_; ++j) bob_alpha[j]   = circuit.add_input(1);

    std::vector<int> alice_delta(128), bob_delta(128);
    for (int j = 0; j < 128; ++j) alice_delta[j] = circuit.add_input(0);
    for (int j = 0; j < 128; ++j) bob_delta[j]   = circuit.add_input(1);

    // Reconstruct shared bits (free XOR gates).
    std::vector<int> alpha_xor(n_), delta_xor(128);
    for (int j = 0; j < n_; ++j)
        alpha_xor[j] = circuit.add_xor(alice_alpha[j], bob_alpha[j]);
    for (int j = 0; j < 128; ++j)
        delta_xor[j] = circuit.add_xor(alice_delta[j], bob_delta[j]);

    // For each (level i, child k): indicator then indicator·Δ.
    for (int i = 0; i < levels; ++i) {
        for (int k = 0; k < n_k; ++k) {
            // Factor per bit: α_bit ⊕ k̄_c = (match → 1, mismatch → 0).
            std::vector<int> factors(w_);
            for (int c = 0; c < w_; ++c) {
                int pos = n_ - w_ * (i + 1) + c;   // bit position in α (LSB=0)
                int k_c = (k >> c) & 1;
                factors[c] = k_c ? alpha_xor[pos]
                                 : circuit.add_not(alpha_xor[pos]);
            }

            // AND tree → indicator (w-1 AND gates, depth ceil(log2(w))).
            while (factors.size() > 1) {
                std::vector<int> next;
                for (size_t j = 0; j + 1 < factors.size(); j += 2)
                    next.push_back(circuit.add_and(factors[j], factors[j + 1]));
                if (factors.size() & 1)
                    next.push_back(factors.back());
                factors = std::move(next);
            }
            int indicator = factors[0];

            // indicator · Δ  (128 parallel AND gates, depth +1).
            for (int j = 0; j < 128; ++j)
                circuit.set_output(circuit.add_and(indicator, delta_xor[j]));
        }
    }

    // ── Prepare this party's input bits ──────────────────────────────────
    std::vector<uint8_t> my_inputs;
    my_inputs.reserve(n_ + 128);

    for (int j = 0; j < n_; ++j)
        my_inputs.push_back(static_cast<uint8_t>((alpha_share >> j) & 1));

    const uint8_t* d_bytes = reinterpret_cast<const uint8_t*>(&delta_share);
    for (int j = 0; j < 128; ++j)
        my_inputs.push_back(static_cast<uint8_t>((d_bytes[j / 8] >> (j % 8)) & 1));

    // ── Evaluate (party_ is 0/1, emp expects ALICE=1/BOB=2) ─────────────
    auto shares = circuit.evaluate_shared(party_ + 1, io_, tg, my_inputs);

    // ── Pack output bits into Blocks ────────────────────────────────────
    std::vector<std::vector<Block>> DltSft_share(levels, std::vector<Block>(n_k));
    int bit_idx = 0;
    for (int i = 0; i < levels; ++i) {
        for (int k = 0; k < n_k; ++k) {
            Block blk = ZERO_BLOCK;
            uint8_t* bp = reinterpret_cast<uint8_t*>(&blk);
            for (int j = 0; j < 128; ++j, ++bit_idx)
                bp[j / 8] |= static_cast<uint8_t>((shares[bit_idx] & 1) << (j % 8));
            DltSft_share[i][k] = blk;
        }
    }

    return {delta_share, std::move(DltSft_share)};
}

// =========================================================================
// batch_F_DeltaShiftShare — B parallel DltSft computations in ONE circuit
//
// Builds a single BoolCircuit containing B independent copies of the
// DltSft sub-circuit (indicator · Δ). All B instances share the same
// AND depth and are evaluated in one call to evaluate_shared, so the
// communication rounds equal those of a single instance.
// =========================================================================

std::vector<DeltaShiftShareResult>
HalfTreeDPF::batch_F_DeltaShiftShare(
    const std::vector<uint64_t>& alpha_shares,
    std::vector<Block>& delta_shares,
    bool_circuit::TripleGen& tg) {

    const int B      = static_cast<int>(alpha_shares.size());
    const int m      = 1 << w_;
    const int levels = n_ / w_;
    const int n_k    = m - 1;

    // Force lsb(Δ) for all instances.
    for (int b = 0; b < B; ++b) {
        if (party_ == 0)
            reinterpret_cast<uint8_t*>(&delta_shares[b])[0] |= 1u;
        else
            reinterpret_cast<uint8_t*>(&delta_shares[b])[0] &=
                static_cast<uint8_t>(~1u);
    }

    // ── Build ONE circuit with B parallel DltSft sub-circuits ────────────
    bool_circuit::BoolCircuit circuit;

    // Per-instance input wires and reconstructed XOR wires.
    struct InstWires {
        std::vector<int> alpha_xor;   // n_ wires
        std::vector<int> delta_xor;   // 128 wires
    };
    std::vector<InstWires> inst(B);

    for (int b = 0; b < B; ++b) {
        // Inputs: party 0 provides n_ + 128 bits, party 1 likewise.
        std::vector<int> aa(n_), ba(n_), ad(128), bd(128);
        for (int j = 0; j < n_; ++j)   aa[j] = circuit.add_input(0);
        for (int j = 0; j < n_; ++j)   ba[j] = circuit.add_input(1);
        for (int j = 0; j < 128; ++j)  ad[j] = circuit.add_input(0);
        for (int j = 0; j < 128; ++j)  bd[j] = circuit.add_input(1);

        inst[b].alpha_xor.resize(n_);
        inst[b].delta_xor.resize(128);
        for (int j = 0; j < n_; ++j)
            inst[b].alpha_xor[j] = circuit.add_xor(aa[j], ba[j]);
        for (int j = 0; j < 128; ++j)
            inst[b].delta_xor[j] = circuit.add_xor(ad[j], bd[j]);
    }

    // DltSft sub-circuits for all B instances (same topology, independent wires).
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < levels; ++i) {
            for (int k = 0; k < n_k; ++k) {
                std::vector<int> factors(w_);
                for (int c = 0; c < w_; ++c) {
                    int pos = n_ - w_ * (i + 1) + c;
                    int k_c = (k >> c) & 1;
                    factors[c] = k_c ? inst[b].alpha_xor[pos]
                                     : circuit.add_not(inst[b].alpha_xor[pos]);
                }
                while (factors.size() > 1) {
                    std::vector<int> next;
                    for (size_t j = 0; j + 1 < factors.size(); j += 2)
                        next.push_back(circuit.add_and(factors[j], factors[j + 1]));
                    if (factors.size() & 1) next.push_back(factors.back());
                    factors = std::move(next);
                }
                int indicator = factors[0];
                for (int j = 0; j < 128; ++j)
                    circuit.set_output(circuit.add_and(indicator,
                                                       inst[b].delta_xor[j]));
            }
        }
    }

    // ── Prepare inputs for all B instances (concatenated) ────────────────
    std::vector<uint8_t> my_inputs;
    my_inputs.reserve(B * (n_ + 128));

    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < n_; ++j)
            my_inputs.push_back(
                static_cast<uint8_t>((alpha_shares[b] >> j) & 1));
        const uint8_t* d = reinterpret_cast<const uint8_t*>(&delta_shares[b]);
        for (int j = 0; j < 128; ++j)
            my_inputs.push_back(
                static_cast<uint8_t>((d[j / 8] >> (j % 8)) & 1));
    }

    // ── Single circuit evaluation for all B instances ────────────────────
    auto shares = circuit.evaluate_shared(party_ + 1, io_, tg, my_inputs);

    // ── Unpack: B × levels × n_k × 128 output bits ─────────────────────
    const int bits_per_inst = levels * n_k * 128;
    std::vector<DeltaShiftShareResult> results(B);

    for (int b = 0; b < B; ++b) {
        std::vector<std::vector<Block>> dltsft(levels, std::vector<Block>(n_k));
        int base = b * bits_per_inst;
        for (int i = 0; i < levels; ++i) {
            for (int k = 0; k < n_k; ++k) {
                Block blk = ZERO_BLOCK;
                uint8_t* bp = reinterpret_cast<uint8_t*>(&blk);
                for (int j = 0; j < 128; ++j) {
                    int idx = base + (i * n_k + k) * 128 + j;
                    bp[j / 8] |= static_cast<uint8_t>(
                        (shares[idx] & 1) << (j % 8));
                }
                dltsft[i][k] = blk;
            }
        }
        results[b] = {delta_shares[b], std::move(dltsft)};
    }

    return results;
}

// =========================================================================
// setup_params — derives shared W via deterministic PRG
// =========================================================================

PartyInput HalfTreeDPF::setup_params(const Block& delta_share,
                                    std::vector<std::vector<Block>> DltSft_share) const {
    emp::PRG rng(&F_RAND_SEED);
    Block W;
    rng.random_block(&W, 1);

    return PartyInput{n_, w_, W, delta_share, std::move(DltSft_share)};
}

// =========================================================================
// DPF.Eval (local)
// =========================================================================

Block HalfTreeDPF::eval(const DPFKey& key, uint64_t j) const {
    const int m      = 1 << key.w;
    const int levels = key.n / key.w;

    Block seed = key.root;
    std::vector<Block> h(m - 1);
    for (int i = 0; i < levels; ++i) {
        bool     t     = get_lsb(seed);
        uint64_t digit = (j >> (key.n - key.w * (i + 1)))
                         & static_cast<uint64_t>(m - 1);

        // One parent → (m-1) pipelined hashes.
        prg_->hash(h.data(), &seed, /*n_parents=*/1);

        Block h_xor = ZERO_BLOCK;
        for (int k = 0; k < m - 1; ++k) h_xor = block_xor(h_xor, h[k]);

        if (static_cast<int>(digit) < m - 1) {
            seed = block_xor(h[digit], cond_xor(t, key.cw[i][digit]));
        } else {
            seed = block_xor(block_xor(h_xor, seed),
                             cond_xor(t, key.cw[i][m - 1]));
        }
    }
    return seed;
}

// =========================================================================
// DPF.Gen for one party
// =========================================================================

DPFKey HalfTreeDPF::gen(const PartyInput& inp) {
    const int m      = 1 << inp.w;
    const int levels = inp.n / inp.w;
    const std::string tag = "[P" + std::to_string(party_) + "]";

    Block root = block_xor(inp.delta_share, inp.W);

    std::vector<Block> tree = {root};
    std::vector<std::vector<Block>> CW(levels, std::vector<Block>(m));

    for (int i = 0; i < levels; ++i) {
        const size_t sz = tree.size();

        // Pipelined batch: sz parents → sz*(m-1) hashes.
        // Layout: hashed[j*(m-1) + k] = H_k(tree[j]).
        std::vector<Block> hashed(sz * static_cast<size_t>(m - 1));
        prg_->hash(hashed.data(), tree.data(), sz);

        std::vector<Block> hashed_xor(sz, ZERO_BLOCK);
        for (size_t j = 0; j < sz; ++j)
            for (int k = 0; k < m - 1; ++k)
                hashed_xor[j] = block_xor(hashed_xor[j], hashed[j * (m - 1) + k]);

        // Compute all (m-1) CW shares locally, then exchange in one batch
        // to reduce from (m-1) round trips per level down to 1.
        std::vector<Block> my_cws(m - 1);
        for (int k = 0; k < m - 1; ++k) {
            my_cws[k] = ZERO_BLOCK;
            for (size_t j = 0; j < sz; ++j)
                my_cws[k] = block_xor(my_cws[k], hashed[j * (m - 1) + k]);
            my_cws[k] = block_xor(my_cws[k], inp.DltSft_share[i][k]);
        }

        std::vector<Block> their_cws(m - 1);
        if (party_ == 0) {
            io_->send_block(my_cws.data(), m - 1);
            io_->flush();
            io_->recv_block(their_cws.data(), m - 1);
        } else {
            io_->recv_block(their_cws.data(), m - 1);
            io_->send_block(my_cws.data(), m - 1);
            io_->flush();
        }

        for (int k = 0; k < m - 1; ++k)
            CW[i][k] = block_xor(my_cws[k], their_cws[k]);

        CW[i][m - 1] = ZERO_BLOCK;
        for (int k = 0; k < m - 1; ++k)
            CW[i][m - 1] = block_xor(CW[i][m - 1], CW[i][k]);

        for (int k = 0; k < m; ++k)
            std::cout << tag << "   Level " << std::setw(2) << i
                      << "  CW[" << i << "," << k << "] = 0x" << hex(CW[i][k]) << "\n";

        std::vector<Block> next;
        next.reserve(sz * static_cast<size_t>(m));
        for (size_t j = 0; j < sz; ++j) {
            bool t = get_lsb(tree[j]);
            for (int k = 0; k < m - 1; ++k)
                next.push_back(block_xor(hashed[j * (m - 1) + k],
                                         cond_xor(t, CW[i][k])));
            next.push_back(block_xor(block_xor(hashed_xor[j], tree[j]),
                                     cond_xor(t, CW[i][m - 1])));
        }
        tree = std::move(next);
    }

    return DPFKey{party_, inp.n, inp.w, root, std::move(CW)};
}

// =========================================================================
// gen_full_eval — fused Gen + full Eval
//
// Same tree expansion as gen(), but returns the N leaf nodes directly
// instead of packing correction words into a DPFKey. This avoids the
// cost of re-traversing root→leaf for every j when the full output
// vector is needed.
// =========================================================================

std::vector<Block> HalfTreeDPF::gen_full_eval(const PartyInput& inp) {
    const int m      = 1 << inp.w;
    const int levels = inp.n / inp.w;

    Block root = block_xor(inp.delta_share, inp.W);
    std::vector<Block> tree = {root};

    for (int i = 0; i < levels; ++i) {
        const size_t sz = tree.size();

        std::vector<Block> hashed(sz * static_cast<size_t>(m - 1));
        prg_->hash(hashed.data(), tree.data(), sz);

        std::vector<Block> hashed_xor(sz, ZERO_BLOCK);
        for (size_t j = 0; j < sz; ++j)
            for (int k = 0; k < m - 1; ++k)
                hashed_xor[j] = block_xor(hashed_xor[j], hashed[j * (m - 1) + k]);

        std::vector<Block> my_cws(m - 1);
        for (int k = 0; k < m - 1; ++k) {
            my_cws[k] = ZERO_BLOCK;
            for (size_t j = 0; j < sz; ++j)
                my_cws[k] = block_xor(my_cws[k], hashed[j * (m - 1) + k]);
            my_cws[k] = block_xor(my_cws[k], inp.DltSft_share[i][k]);
        }

        std::vector<Block> their_cws(m - 1);
        if (party_ == 0) {
            io_->send_block(my_cws.data(), m - 1);
            io_->flush();
            io_->recv_block(their_cws.data(), m - 1);
        } else {
            io_->recv_block(their_cws.data(), m - 1);
            io_->send_block(my_cws.data(), m - 1);
            io_->flush();
        }

        std::vector<Block> CW(m);
        for (int k = 0; k < m - 1; ++k)
            CW[k] = block_xor(my_cws[k], their_cws[k]);
        CW[m - 1] = ZERO_BLOCK;
        for (int k = 0; k < m - 1; ++k)
            CW[m - 1] = block_xor(CW[m - 1], CW[k]);

        std::vector<Block> next;
        next.reserve(sz * static_cast<size_t>(m));
        for (size_t j = 0; j < sz; ++j) {
            bool t = get_lsb(tree[j]);
            for (int k = 0; k < m - 1; ++k)
                next.push_back(block_xor(hashed[j * (m - 1) + k],
                                         cond_xor(t, CW[k])));
            next.push_back(block_xor(block_xor(hashed_xor[j], tree[j]),
                                     cond_xor(t, CW[m - 1])));
        }
        tree = std::move(next);
    }

    return tree;  // N = 2^n leaf shares
}

// =========================================================================
// batch_gen_full_eval — B concurrent DPF full evaluations
//
// At each tree level:
//   1. Hash all parents across all B instances (pipelined via MKeyPRP).
//   2. Compute B*(m-1) CW shares locally.
//   3. Exchange ALL B*(m-1) CW shares in one send/recv (1 round trip).
//   4. Expand all B trees to the next level.
//
// Communication: levels round trips total, each carrying B*(m-1) blocks.
// =========================================================================

std::vector<std::vector<Block>>
HalfTreeDPF::batch_gen_full_eval(const std::vector<PartyInput>& inps) {
    const int B      = static_cast<int>(inps.size());
    const int m      = 1 << w_;
    const int levels = n_ / w_;

#ifdef PCG_ENABLE_CUDA
    // GPU full-domain evaluation: AES maps to the device AES CCR core,
    // ChaCha8 to the device ChaCha8 keystream PRG (both bit-exact vs the
    // CPU TreePRG backends).
    if (eval_backend_ == EvalBackend::GPU && B > 0 &&
        pcg_cuda::is_cuda_available())
        return batch_gen_full_eval_gpu(inps);
#endif

    std::vector<std::vector<Block>> trees(B);
    for (int b = 0; b < B; ++b)
        trees[b] = {block_xor(inps[b].delta_share, inps[b].W)};

    for (int i = 0; i < levels; ++i) {
        const size_t sz = trees[0].size();

        // ── Hash all parents across all B instances ─────────────────────
        std::vector<std::vector<Block>> h_all(B);
        std::vector<std::vector<Block>> hx_all(B);
        for (int b = 0; b < B; ++b) {
            h_all[b].resize(sz * static_cast<size_t>(m - 1));
            prg_->hash(h_all[b].data(), trees[b].data(), sz);

            hx_all[b].assign(sz, ZERO_BLOCK);
            for (size_t j = 0; j < sz; ++j)
                for (int k = 0; k < m - 1; ++k)
                    hx_all[b][j] = block_xor(hx_all[b][j],
                                             h_all[b][j * (m - 1) + k]);
        }

        // ── Compute CW shares for all instances ────────────────────────
        const size_t total_cws = static_cast<size_t>(B) * (m - 1);
        std::vector<Block> my_cws(total_cws);
        for (int b = 0; b < B; ++b) {
            for (int k = 0; k < m - 1; ++k) {
                Block cw = ZERO_BLOCK;
                for (size_t j = 0; j < sz; ++j)
                    cw = block_xor(cw, h_all[b][j * (m - 1) + k]);
                cw = block_xor(cw, inps[b].DltSft_share[i][k]);
                my_cws[b * (m - 1) + k] = cw;
            }
        }

        // ── Single round trip for ALL B*(m-1) CW shares ────────────────
        // M_l = B*(m-1)*sizeof(Block), constant across levels (the frontier
        // is collapsed by the XOR reduction above before the send).
        std::vector<Block> their_cws(total_cws);
        const size_t cw_bytes = total_cws * sizeof(Block);
        PCG_COMM_TRACE(io_, "dpf_gen", i, B, cw_bytes, cw_bytes, {
            if (party_ == 0) {
                io_->send_block(my_cws.data(), total_cws);
                io_->flush();
                PCG_COMM_TRACE_MID();
                io_->recv_block(their_cws.data(), total_cws);
            } else {
                io_->recv_block(their_cws.data(), total_cws);
                PCG_COMM_TRACE_MID();
                io_->send_block(my_cws.data(), total_cws);
                io_->flush();
            }
        });

        // ── Reconstruct CWs and expand each tree ───────────────────────
        for (int b = 0; b < B; ++b) {
            Block CW[16];   // m ≤ 16 (w ≤ 4)
            CW[m - 1] = ZERO_BLOCK;
            for (int k = 0; k < m - 1; ++k) {
                CW[k] = block_xor(my_cws[b * (m - 1) + k],
                                  their_cws[b * (m - 1) + k]);
                CW[m - 1] = block_xor(CW[m - 1], CW[k]);
            }

            std::vector<Block> next;
            next.reserve(sz * static_cast<size_t>(m));
            for (size_t j = 0; j < sz; ++j) {
                bool t = get_lsb(trees[b][j]);
                for (int k = 0; k < m - 1; ++k)
                    next.push_back(block_xor(h_all[b][j * (m - 1) + k],
                                             cond_xor(t, CW[k])));
                next.push_back(block_xor(block_xor(hx_all[b][j], trees[b][j]),
                                         cond_xor(t, CW[m - 1])));
            }
            trees[b] = std::move(next);
        }
    }

    return trees;
}

// =========================================================================
// batch_gen_full_eval_gpu — device full-domain evaluation (AES or ChaCha8)
//
// Bridges the half-tree DPF to common/dpf_gpu.cu: derives the AES sub-key
// schedules from master_key_ (identical to AESTreePRG), flattens roots and
// DltSft shares into POD blocks, and supplies a host callback for the tiny
// per-level CW-share network round. Leaves stay device-resident until the
// single final copy-out.
// =========================================================================

#ifdef PCG_ENABLE_CUDA
std::vector<std::vector<Block>>
HalfTreeDPF::batch_gen_full_eval_gpu(const std::vector<PartyInput>& inps) {
    const int B      = static_cast<int>(inps.size());
    const int m      = 1 << w_;
    const int nk     = m - 1;
    const int levels = n_ / w_;
    const size_t D   = static_cast<size_t>(1) << n_;

    // AES: derive the nk tree-PRG sub-keys from master_key_ (matching
    // AESTreePRG), then expand each into AES round keys for the device.
    // ChaCha8 is keyless (the parent block is the key) — no schedule.
    const pcg_cuda::DpfGpuPrg gpu_prg = (prg_type_ == PRGType::CHACHA8)
        ? pcg_cuda::DpfGpuPrg::CHACHA8 : pcg_cuda::DpfGpuPrg::AES;
    std::vector<uint8_t> round_keys;
    if (gpu_prg == pcg_cuda::DpfGpuPrg::AES) {
        round_keys.resize(static_cast<size_t>(nk) * 176);
        emp::AES_KEY master;
        emp::AES_set_encrypt_key(master_key_, &master);
        for (int k = 0; k < nk; ++k) {
            emp::block tweak = emp::makeBlock(0, static_cast<uint64_t>(k));
            emp::block c = tweak;
            emp::AES_ecb_encrypt_blks(&c, 1, &master);
            emp::block sk = c ^ tweak;
            uint8_t skb[16];
            std::memcpy(skb, &sk, 16);
            pcg_cuda::aes128_expand_key_host(
                skb, &round_keys[static_cast<size_t>(k) * 176]);
        }
    }

    // Flatten roots and DltSft shares into DpfBlk arrays.
    std::vector<pcg_cuda::DpfBlk> roots(B);
    std::vector<pcg_cuda::DpfBlk> dltsft(static_cast<size_t>(B) * levels * nk);
    for (int b = 0; b < B; ++b) {
        Block root = block_xor(inps[b].delta_share, inps[b].W);
        std::memcpy(&roots[b], &root, sizeof(pcg_cuda::DpfBlk));
        for (int i = 0; i < levels; ++i)
            for (int k = 0; k < nk; ++k)
                std::memcpy(
                    &dltsft[(static_cast<size_t>(b) * levels + i) * nk + k],
                    &inps[b].DltSft_share[i][k], sizeof(pcg_cuda::DpfBlk));
    }

    // Per-level CW-share exchange on the host (same ordering as the CPU path).
    const int party = party_;
    emp::NetIO* io = io_;
    auto exchange = [party, io, B](const pcg_cuda::DpfBlk* my,
                                   pcg_cuda::DpfBlk* peer, size_t count) {
        const Block* mine = reinterpret_cast<const Block*>(my);
        Block* theirs = reinterpret_cast<Block*>(peer);
        const int n = static_cast<int>(count);
        const size_t bytes = count * sizeof(Block);
        PCG_COMM_TRACE(io, "dpf_gen_gpu", -1, B, bytes, bytes, {
            if (party == 0) {
                io->send_block(mine, n);
                io->flush();
                PCG_COMM_TRACE_MID();
                io->recv_block(theirs, n);
            } else {
                io->recv_block(theirs, n);
                PCG_COMM_TRACE_MID();
                io->send_block(mine, n);
                io->flush();
            }
        });
    };

    std::vector<pcg_cuda::DpfBlk> leaves(static_cast<size_t>(B) * D);
    pcg_cuda::dpf_gpu_batch_full_eval(
        B, n_, w_, roots.data(), dltsft.data(),
        round_keys.empty() ? nullptr : round_keys.data(), party_, exchange,
        leaves.data(), gpu_prg);

    std::vector<std::vector<Block>> trees(B);
    for (int b = 0; b < B; ++b) {
        trees[b].resize(D);
        std::memcpy(trees[b].data(), &leaves[static_cast<size_t>(b) * D],
                    D * sizeof(Block));
    }
    return trees;
}

// Same prep as batch_gen_full_eval_gpu, but the leaves stay device-resident
// (no B*D copy-out; see dpf.h). Kept separate so the host-returning oracle
// path above stays byte-for-byte what it was.
const pcg_cuda::DpfBlk*
HalfTreeDPF::batch_gen_full_eval_device(const std::vector<PartyInput>& inps) {
    const int B      = static_cast<int>(inps.size());
    const int m      = 1 << w_;
    const int nk     = m - 1;
    const int levels = n_ / w_;

    const pcg_cuda::DpfGpuPrg gpu_prg = (prg_type_ == PRGType::CHACHA8)
        ? pcg_cuda::DpfGpuPrg::CHACHA8 : pcg_cuda::DpfGpuPrg::AES;
    std::vector<uint8_t> round_keys;
    if (gpu_prg == pcg_cuda::DpfGpuPrg::AES) {
        round_keys.resize(static_cast<size_t>(nk) * 176);
        emp::AES_KEY master;
        emp::AES_set_encrypt_key(master_key_, &master);
        for (int k = 0; k < nk; ++k) {
            emp::block tweak = emp::makeBlock(0, static_cast<uint64_t>(k));
            emp::block c = tweak;
            emp::AES_ecb_encrypt_blks(&c, 1, &master);
            emp::block sk = c ^ tweak;
            uint8_t skb[16];
            std::memcpy(skb, &sk, 16);
            pcg_cuda::aes128_expand_key_host(
                skb, &round_keys[static_cast<size_t>(k) * 176]);
        }
    }

    std::vector<pcg_cuda::DpfBlk> roots(B);
    std::vector<pcg_cuda::DpfBlk> dltsft(static_cast<size_t>(B) * levels * nk);
    for (int b = 0; b < B; ++b) {
        Block root = block_xor(inps[b].delta_share, inps[b].W);
        std::memcpy(&roots[b], &root, sizeof(pcg_cuda::DpfBlk));
        for (int i = 0; i < levels; ++i)
            for (int k = 0; k < nk; ++k)
                std::memcpy(
                    &dltsft[(static_cast<size_t>(b) * levels + i) * nk + k],
                    &inps[b].DltSft_share[i][k], sizeof(pcg_cuda::DpfBlk));
    }

    const int party = party_;
    emp::NetIO* io = io_;
    auto exchange = [party, io, B](const pcg_cuda::DpfBlk* my,
                                   pcg_cuda::DpfBlk* peer, size_t count) {
        const Block* mine = reinterpret_cast<const Block*>(my);
        Block* theirs = reinterpret_cast<Block*>(peer);
        const int n = static_cast<int>(count);
        const size_t bytes = count * sizeof(Block);
        PCG_COMM_TRACE(io, "dpf_gen_gpu", -1, B, bytes, bytes, {
            if (party == 0) {
                io->send_block(mine, n);
                io->flush();
                PCG_COMM_TRACE_MID();
                io->recv_block(theirs, n);
            } else {
                io->recv_block(theirs, n);
                PCG_COMM_TRACE_MID();
                io->send_block(mine, n);
                io->flush();
            }
        });
    };

    return pcg_cuda::dpf_gpu_batch_full_eval_device(
        B, n_, w_, roots.data(), dltsft.data(),
        round_keys.empty() ? nullptr : round_keys.data(), party_, exchange,
        gpu_prg);
}
#else
std::vector<std::vector<Block>>
HalfTreeDPF::batch_gen_full_eval_gpu(const std::vector<PartyInput>&) {
    throw std::runtime_error(
        "GPU DPF requested but built without PCG_ENABLE_CUDA");
}

const pcg_cuda::DpfBlk*
HalfTreeDPF::batch_gen_full_eval_device(const std::vector<PartyInput>&) {
    throw std::runtime_error(
        "GPU DPF requested but built without PCG_ENABLE_CUDA");
}
#endif

}  // namespace half_tree_dpf
