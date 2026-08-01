// Template implementation for PCG_OLE<P>.  Included from pcg_ole.h.

#ifndef PCG_OLE_2PC_PCG_OLE_IMPL_H__
#define PCG_OLE_2PC_PCG_OLE_IMPL_H__

#include <cassert>
#include <cstring>
#include <iostream>

#include <emp-tool/emp-tool.h>
#include <openssl/rand.h>

#include "common/prof.h"
#include "common/comm_trace.h"

#ifdef PCG_ENABLE_CUDA
#include "common/leaf_convert_cuda.h"
#include "common/poly_mul_cuda.h"   // is_cuda_available()
#endif

namespace pcg_ole {

using Block = half_tree_dpf::Block;

static inline uint64_t block_to_uint64(const Block& blk) {
    uint64_t lo;
    std::memcpy(&lo, &blk, sizeof(uint64_t));
    return lo;
}

static inline Block uint64_to_block(uint64_t val) {
    Block blk = half_tree_dpf::ZERO_BLOCK;
    std::memcpy(&blk, &val, sizeof(uint64_t));
    return blk;
}

template<uint64_t P>
PCG_OLE<P>::PCG_OLE(int party, emp::NetIO* io, const PCGParams& params)
    : party_(party), io_(io), params_(params) {
    assert(party == 0 || party == 1);
    assert(params.N > 0 && (params.N & (params.N - 1)) == 0);
    assert(params.t > 0 && params.N % params.t == 0);
}

template<uint64_t P>
OLEOutput<P>
PCG_OLE<P>::gen_and_expand(bool_circuit::TripleGen& tg,
                           const std::vector<Poly<P>>& public_a) {
    using F = FFp<P>;
    const int N = params_.N;
    const int c = params_.c;
    const int t = params_.t;
    const int w = params_.w;
    const int bin_sz = N / t;
    const int D = 2 * bin_sz;
    const int dpf_n = __builtin_ctz(D);
    assert((1 << dpf_n) == D);

    [[maybe_unused]] const std::string tag =
        "[P" + std::to_string(party_) + "]";

    // ── 1. Sample regular noise ─────────────────────────────────────────
    PCG_PROF_TIC(_tic_noise);
    emp::PRG prg;
    std::vector<std::vector<uint64_t>> my_positions(c, std::vector<uint64_t>(t));
    std::vector<std::vector<F>>        my_payloads(c, std::vector<F>(t));

    for (int i = 0; i < c; ++i) {
        for (int k = 0; k < t; ++k) {
            uint64_t r;
            prg.random_data(&r, sizeof(r));
            my_positions[i][k] = static_cast<uint64_t>(k) * bin_sz + (r % bin_sz);

            prg.random_data(&r, sizeof(r));
            my_payloads[i][k] = F(r % (P - 1) + 1);   // non-zero
        }
    }

    // ── 2. Build sparse polynomials e^i_σ ───────────────────────────────
    std::vector<Poly<P>> my_errors(c);
    for (int i = 0; i < c; ++i) {
        my_errors[i] = poly_zero<P>(N);
        for (int k = 0; k < t; ++k)
            my_errors[i][my_positions[i][k]] = my_payloads[i][k];
    }
    PCG_PROF_TOC(PREPROCESSING, _tic_noise);

    // ── 3. DPF setup via real 2-PC ──────────────────────────────────────
    const uint8_t S_bytes[16] = {0x2b,0x7e,0x15,0x16, 0x28,0xae,0xd2,0xa6,
                                 0xab,0xf7,0x15,0x88, 0x09,0xcf,0x4f,0x3c};
    Block aes_key;
    std::memcpy(&aes_key, S_bytes, 16);

    half_tree_dpf::HalfTreeDPF dpf(party_, io_, dpf_n, w, aes_key,
                                    params_.prg_type);

    int off_bits = 0;
    { int tmp = bin_sz - 1; while (tmp) { ++off_bits; tmp >>= 1; } }
    if (off_bits == 0) off_bits = 1;

    std::vector<Poly<P>> g_all(c * c);

    // Device path: g comes back from the GPU already negacyclically folded
    // (N residues), stored here; step 4 skips the host fold for those blocks.
    std::vector<std::vector<uint64_t>> g_dev(c * c);
#ifdef PCG_ENABLE_CUDA
    const bool device_leaves =
        params_.dpf_eval_backend == half_tree_dpf::EvalBackend::GPU &&
        pcg_cuda::is_cuda_available();
#else
    const bool device_leaves = false;
#endif

    for (int ii = 0; ii < c; ++ii) {
        for (int jj = 0; jj < c; ++jj) {
            const int B = t * t;

            // ── 3a. Position addition (Kogge-Stone adder) ───────────────
            PCG_PROF_TIC(_tic_posadd);
            bool_circuit::BoolCircuit pos_circuit;
            std::vector<std::vector<int>> aw(B), bw(B), sw(B);

            for (int idx = 0; idx < B; ++idx) {
                aw[idx].resize(off_bits);
                bw[idx].resize(off_bits);
                for (int bit = 0; bit < off_bits; ++bit)
                    aw[idx][bit] = pos_circuit.add_input(0);
                for (int bit = 0; bit < off_bits; ++bit)
                    bw[idx][bit] = pos_circuit.add_input(1);
                sw[idx] = bool_circuit::build_kogge_stone_adder(
                    pos_circuit, off_bits, aw[idx], bw[idx]);
                for (int bit = 0; bit <= off_bits; ++bit)
                    pos_circuit.set_output(sw[idx][bit]);
            }

            std::vector<uint8_t> pos_inputs;
            for (int idx = 0; idx < B; ++idx) {
                int kk = idx / t, ll = idx % t;
                uint64_t my_off = (party_ == 0)
                    ? my_positions[ii][kk] % bin_sz
                    : my_positions[jj][ll] % bin_sz;
                for (int bit = 0; bit < off_bits; ++bit)
                    pos_inputs.push_back(static_cast<uint8_t>((my_off >> bit) & 1));
            }

            auto pos_shares = pos_circuit.evaluate_shared(
                party_ + 1, io_, tg, pos_inputs);

            std::vector<uint64_t> alpha_shares(B, 0);
            int pbi = 0;
            for (int idx = 0; idx < B; ++idx) {
                for (int bit = 0; bit <= off_bits; ++bit) {
                    alpha_shares[idx] |=
                        static_cast<uint64_t>(pos_shares[pbi] & 1) << bit;
                    ++pbi;
                }
            }
            PCG_PROF_TOC(PREPROCESSING, _tic_posadd);
            PCG_PROF_TOC(PRE_KS, _tic_posadd);

            // ── 3b. Payload multiplication (Gilboa OT) ──────────────────
            PCG_PROF_TIC(_tic_payot);
            std::vector<uint64_t> my_pay_raw(B);
            for (int idx = 0; idx < B; ++idx) {
                int kk = idx / t, ll = idx % t;
                my_pay_raw[idx] = (party_ == 0)
                    ? my_payloads[ii][kk].val()
                    : my_payloads[jj][ll].val();
            }
            std::vector<uint64_t> beta_raw(B);
            tg.zp_multiply(my_pay_raw.data(), B, P, beta_raw.data());

            // F_p Beaver triples for the output-layer ⟨A·B⟩ multiplication
            // (input-independent, batched with the payload OTs).
            std::vector<uint64_t> bt_r(B), bt_s(B), bt_rs(B);
            tg.zp_triple(B, P, bt_r.data(), bt_s.data(), bt_rs.data());
            PCG_PROF_TOC(PREPROCESSING, _tic_payot);
            PCG_PROF_TOC(PRE_GILBOA, _tic_payot);

            // ── 3c. DPF Gen (batch F_DeltaShiftShare + batch_gen_full_eval) ─
            // All B DltSft computations run in ONE circuit evaluation.
            PCG_PROF_TIC(_tic_dltsft);
            std::vector<Block> delta_shares_vec(B);
            for (int idx = 0; idx < B; ++idx) {
                if (RAND_bytes(reinterpret_cast<uint8_t*>(&delta_shares_vec[idx]),
                               16) != 1)
                    throw std::runtime_error("RAND_bytes failed");
            }

            auto all_dltsft = dpf.batch_F_DeltaShiftShare(
                alpha_shares, delta_shares_vec, tg);

            std::vector<half_tree_dpf::PartyInput> dpf_inputs(B);
            for (int idx = 0; idx < B; ++idx)
                dpf_inputs[idx] = dpf.setup_params(
                    all_dltsft[idx].delta_share,
                    std::move(all_dltsft[idx].DltSft_share));
            PCG_PROF_TOC(PREPROCESSING, _tic_dltsft);
            PCG_PROF_TOC(PRE_DLTSFT, _tic_dltsft);

#ifdef PCG_VERBOSE
            std::cout << tag << " DPF batch (i=" << ii << ",j=" << jj
                      << "): " << B << " instances, D=" << D << "\n";
            std::cout.flush();
#endif

#ifdef PCG_ENABLE_CUDA
            // ── Device fast path: leaves never cross PCIe ───────────────
            // Same output-layer algebra as the host path below; the GPU
            // computes the sign-free raw sums (sumC, sumT) and, after the
            // two Beaver opens, the folded g directly (bit-exact: identical
            // ChaCha8 out-hash, mod-P adds are order-independent).
            if (device_leaves) {
                const pcg_cuda::DpfBlk* d_leaves =
                    dpf.batch_gen_full_eval_device(dpf_inputs);

                PCG_PROF_TIC(_tic_out_local1);
                std::vector<uint64_t> sumC(B), sumT(B);
                pcg_cuda::dpf_out_sums(d_leaves, B, static_cast<size_t>(D),
                                       P, sumC.data(), sumT.data());

                std::vector<uint64_t> A_raw(B), B_raw(B);
                for (int idx = 0; idx < B; ++idx) {
                    F A_sh = F(sumT[idx]);
                    if (party_ == 1) A_sh = -A_sh;
                    F B_sh = F::raw(sumC[idx]);
                    if (party_ == 0) B_sh = -B_sh;
                    B_sh += F::raw(beta_raw[idx]);
                    A_raw[idx] = A_sh.val();
                    B_raw[idx] = B_sh.val();
                }

                // Open d = A - r and e = B - s (one round trip).
                std::vector<uint64_t> de_my(2 * B), de_peer(2 * B);
                for (int idx = 0; idx < B; ++idx) {
                    de_my[idx] =
                        (F::raw(A_raw[idx]) - F::raw(bt_r[idx])).val();
                    de_my[B + idx] =
                        (F::raw(B_raw[idx]) - F::raw(bt_s[idx])).val();
                }
                PCG_PROF_TOC(DPF_EXPAND_LOCAL, _tic_out_local1);

                PCG_PROF_TIC(_tic_out_comm1);
                {
                    PCG_PROF_ZONE(NET_WAIT);
                    const size_t de_bytes = 2 * B * sizeof(uint64_t);
                    PCG_COMM_TRACE(io_, "beaver_de", ii * c + jj, B,
                                   de_bytes, de_bytes, {
                        if (party_ == 0) {
                            io_->send_data(de_my.data(), de_bytes);
                            io_->flush();
                            PCG_COMM_TRACE_MID();
                            io_->recv_data(de_peer.data(), de_bytes);
                        } else {
                            io_->recv_data(de_peer.data(), de_bytes);
                            PCG_COMM_TRACE_MID();
                            io_->send_data(de_my.data(), de_bytes);
                            io_->flush();
                        }
                    });
                }
                PCG_PROF_TOC(DPF_EXPAND_COMM, _tic_out_comm1);

                // ⟨CW⟩ = [σ=0]·de + d·⟨s⟩ + e·⟨r⟩ + ⟨rs⟩; open CW.
                PCG_PROF_TIC(_tic_out_local2);
                std::vector<uint64_t> CW_my_raw(B), CW_peer_raw(B);
                for (int idx = 0; idx < B; ++idx) {
                    F dv = F::raw(de_my[idx]) + F::raw(de_peer[idx]);
                    F ev = F::raw(de_my[B + idx]) + F::raw(de_peer[B + idx]);
                    F cw = dv * F::raw(bt_s[idx]) + ev * F::raw(bt_r[idx])
                         + F::raw(bt_rs[idx]);
                    if (party_ == 0) cw += dv * ev;
                    CW_my_raw[idx] = cw.val();
                }
                PCG_PROF_TOC(DPF_EXPAND_LOCAL, _tic_out_local2);

                PCG_PROF_TIC(_tic_out_comm2);
                {
                    PCG_PROF_ZONE(NET_WAIT);
                    const size_t cw_bytes = B * sizeof(uint64_t);
                    PCG_COMM_TRACE(io_, "beaver_cw", ii * c + jj, B,
                                   cw_bytes, cw_bytes, {
                        if (party_ == 0) {
                            io_->send_data(CW_my_raw.data(), cw_bytes);
                            io_->flush();
                            PCG_COMM_TRACE_MID();
                            io_->recv_data(CW_peer_raw.data(), cw_bytes);
                        } else {
                            io_->recv_data(CW_peer_raw.data(), cw_bytes);
                            PCG_COMM_TRACE_MID();
                            io_->send_data(CW_my_raw.data(), cw_bytes);
                            io_->flush();
                        }
                    });
                }
                PCG_PROF_TOC(DPF_EXPAND_COMM, _tic_out_comm2);

                PCG_PROF_TIC(_tic_out_local3);
                std::vector<uint64_t> CW_full(B);
                for (int idx = 0; idx < B; ++idx)
                    CW_full[idx] = (F::raw(CW_my_raw[idx])
                                  + F::raw(CW_peer_raw[idx])).val();

                std::vector<uint64_t> g_red(N);
                pcg_cuda::dpf_out_scatter_g(d_leaves, B,
                                            static_cast<size_t>(D), t,
                                            party_, P, CW_full.data(),
                                            g_red.data(), N);
                g_dev[ii * c + jj] = std::move(g_red);
                PCG_PROF_TOC(DPF_EXPAND_LOCAL, _tic_out_local3);
                continue;
            }
#endif

            // NOTE: batch_gen_full_eval performs `levels = dpf_n/w` blocking
            // round trips of B*(m-1) blocks each (dpf.cpp, "dpf_gen" rows of
            // the comm trace). Those are NOT counted in the DPF_EXPAND_COMM /
            // NET_WAIT zones below -- only the two Beaver opens are. Use the
            // comm-trace CSV (PCG_ENABLE_PROFILING) for the full picture.
            PCG_PROF_TIC(_tic_dpf_gen);
            auto all_leaves = dpf.batch_gen_full_eval(dpf_inputs);
            PCG_PROF_TOC(DPF_GEN, _tic_dpf_gen);

            // ── 3d. Output layer: hash leaves + Beaver-corrected CW ─────
            //
            // C_σ[d] = Convert(H'(leaf)),  τ_σ[d] = lsb(leaf).
            // ⟨A⟩ = (-1)^σ Σ τ,  ⟨B⟩ = (-1)^{1-σ} Σ C + ⟨β⟩.
            // CW = A·B via Beaver triple (open d = A-r, e = B-s, then CW).
            // y_σ[d] = (-1)^σ (C_σ[d] + τ_σ[d]·CW).
            PCG_PROF_TIC(_tic_out_local1);
            Poly<P> g_ij(2 * N, F::zero());

            // Output-layer hash: ChaCha8 keystream keyed by the leaf,
            // domain-separated from the tree PRG by a fixed nonzero nonce
            // (tree levels run with nonce = 0). See ChaCha8OutHash.
            half_tree_dpf::ChaCha8OutHash out_hash;

            std::vector<std::vector<F>>       all_C(B, std::vector<F>(D));
            std::vector<std::vector<uint8_t>> all_tau(B, std::vector<uint8_t>(D));
            std::vector<uint64_t> A_raw(B), B_raw(B);

            #pragma omp parallel for schedule(static) if(B >= 2)
            for (int idx = 0; idx < B; ++idx) {
                std::vector<Block> hashed(D);
                out_hash.hash(hashed.data(), all_leaves[idx].data(), D);

                F sumC = F::zero();
                uint64_t sumT = 0;
                for (int d = 0; d < D; ++d) {
                    F cval(block_to_uint64(hashed[d]));
                    uint8_t tau =
                        half_tree_dpf::get_lsb(all_leaves[idx][d]) ? 1 : 0;
                    all_C[idx][d]   = cval;
                    all_tau[idx][d] = tau;
                    sumC += cval;
                    sumT += tau;
                }
                F A_sh = F(sumT);
                if (party_ == 1) A_sh = -A_sh;
                F B_sh = sumC;
                if (party_ == 0) B_sh = -B_sh;
                B_sh += F::raw(beta_raw[idx]);
                A_raw[idx] = A_sh.val();
                B_raw[idx] = B_sh.val();
            }

            // Open d = A - r and e = B - s (one round trip, 2B elements).
            std::vector<uint64_t> de_my(2 * B), de_peer(2 * B);
            for (int idx = 0; idx < B; ++idx) {
                de_my[idx]     = (F::raw(A_raw[idx]) - F::raw(bt_r[idx])).val();
                de_my[B + idx] = (F::raw(B_raw[idx]) - F::raw(bt_s[idx])).val();
            }
            PCG_PROF_TOC(DPF_EXPAND_LOCAL, _tic_out_local1);

            PCG_PROF_TIC(_tic_out_comm1);
            {
                PCG_PROF_ZONE(NET_WAIT);
                const size_t de_bytes = 2 * B * sizeof(uint64_t);
                PCG_COMM_TRACE(io_, "beaver_de", ii * c + jj, B,
                               de_bytes, de_bytes, {
                    if (party_ == 0) {
                        io_->send_data(de_my.data(), de_bytes);
                        io_->flush();
                        PCG_COMM_TRACE_MID();
                        io_->recv_data(de_peer.data(), de_bytes);
                    } else {
                        io_->recv_data(de_peer.data(), de_bytes);
                        PCG_COMM_TRACE_MID();
                        io_->send_data(de_my.data(), de_bytes);
                        io_->flush();
                    }
                });
            }

            PCG_PROF_TOC(DPF_EXPAND_COMM, _tic_out_comm1);

            // ⟨CW⟩ = [σ=0]·de + d·⟨s⟩ + e·⟨r⟩ + ⟨rs⟩; open CW.
            PCG_PROF_TIC(_tic_out_local2);
            std::vector<uint64_t> CW_my_raw(B), CW_peer_raw(B);
            for (int idx = 0; idx < B; ++idx) {
                F dv = F::raw(de_my[idx])     + F::raw(de_peer[idx]);
                F ev = F::raw(de_my[B + idx]) + F::raw(de_peer[B + idx]);
                F cw = dv * F::raw(bt_s[idx]) + ev * F::raw(bt_r[idx])
                     + F::raw(bt_rs[idx]);
                if (party_ == 0) cw += dv * ev;
                CW_my_raw[idx] = cw.val();
            }
            PCG_PROF_TOC(DPF_EXPAND_LOCAL, _tic_out_local2);

            PCG_PROF_TIC(_tic_out_comm2);
            {
                PCG_PROF_ZONE(NET_WAIT);
                const size_t cw_bytes = B * sizeof(uint64_t);
                PCG_COMM_TRACE(io_, "beaver_cw", ii * c + jj, B,
                               cw_bytes, cw_bytes, {
                    if (party_ == 0) {
                        io_->send_data(CW_my_raw.data(), cw_bytes);
                        io_->flush();
                        PCG_COMM_TRACE_MID();
                        io_->recv_data(CW_peer_raw.data(), cw_bytes);
                    } else {
                        io_->recv_data(CW_peer_raw.data(), cw_bytes);
                        PCG_COMM_TRACE_MID();
                        io_->send_data(CW_my_raw.data(), cw_bytes);
                        io_->flush();
                    }
                });
            }

            PCG_PROF_TOC(DPF_EXPAND_COMM, _tic_out_comm2);

            PCG_PROF_TIC(_tic_out_local3);
            for (int idx = 0; idx < B; ++idx) {
                F CW = F::raw(CW_my_raw[idx]) + F::raw(CW_peer_raw[idx]);
                int kk = idx / t, ll = idx % t;
                int base = (kk + ll) * bin_sz;

                for (int d = 0; d < D; ++d) {
                    F y = all_C[idx][d];
                    if (all_tau[idx][d]) y += CW;
                    if (party_ == 1) y = -y;
                    int pos = base + d;
                    if (pos < 2 * N)
                        g_ij[pos] += y;
                }
            }

            g_all[ii * c + jj] = std::move(g_ij);
            PCG_PROF_TOC(DPF_EXPAND_LOCAL, _tic_out_local3);
        }
    }

    // ── 4. Compute OLE output ───────────────────────────────────────────
    PCG_PROF_TIC(_tic_step4);
    std::vector<Poly<P>> a(c);
    a[0] = poly_zero<P>(N);
    a[0][0] = F::one();
    for (int i = 1; i < c; ++i) a[i] = public_a[i - 1];

    Poly<P> x = poly_zero<P>(N);
    for (int i = 0; i < c; ++i)
        x = poly_add<P>(x, poly_mul<P>(a[i], my_errors[i], N));

    Poly<P> z = poly_zero<P>(N);
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < c; ++j) {
            Poly<P> g_reduced = poly_zero<P>(N);
            if (device_leaves) {
                // GPU path returned g already negacyclically folded.
                const auto& gr = g_dev[i * c + j];
                for (int d = 0; d < N; ++d)
                    g_reduced[d] = F::raw(gr[d]);
            } else {
                for (int d = 0; d < 2 * N; ++d) {
                    if (g_all[i * c + j][d].is_zero()) continue;
                    if (d < N) g_reduced[d]     += g_all[i * c + j][d];
                    else       g_reduced[d - N] -= g_all[i * c + j][d];
                }
            }
            Poly<P> ai_aj = poly_mul<P>(a[i], a[j], N);
            z = poly_add<P>(z, poly_mul<P>(ai_aj, g_reduced, N));
        }
    }
    PCG_PROF_TOC(DPF_EXPAND_LOCAL, _tic_step4);

#ifdef PCG_VERBOSE
    std::cout << tag << " OLE output computed.\n";
#endif
    return {std::move(x), std::move(z)};
}

}  // namespace pcg_ole

#endif  // PCG_OLE_2PC_PCG_OLE_IMPL_H__
