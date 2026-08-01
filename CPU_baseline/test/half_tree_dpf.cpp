// Two-process driver for the Half-Tree m-ary DPF.
//
//   P0 (server):  ./test_half_tree_dpf 0 <port> <n> <w> <alpha_0>
//   P1 (client):  ./test_half_tree_dpf 1 <host> <port> <n> <w> <alpha_1>
//
// α = α_0 ⊕ α_1.  Start P0 first; P1 connects once P0 is listening.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openssl/rand.h>

#include <emp-tool/io/net_io_channel.h>

#include "bool_circuit/circuit.h"
#include "half_tree_dpf/dpf.h"

#ifdef PCG_ENABLE_CUDA
#include "common/poly_mul_cuda.h"  // pcg_cuda::is_cuda_available()
#endif

using half_tree_dpf::Block;
using half_tree_dpf::DPFKey;
using half_tree_dpf::HalfTreeDPF;
using half_tree_dpf::PartyInput;
using half_tree_dpf::DeltaShiftShareResult;
using half_tree_dpf::ZERO_BLOCK;
using half_tree_dpf::block_eq;
using half_tree_dpf::block_xor;
using half_tree_dpf::get_lsb;
using half_tree_dpf::hex;

// Exchange α and Δ shares for verification only (insecure — test use only).
static void exchange_for_verify(int party, emp::NetIO* io,
                                uint64_t alpha_share, const Block& delta_share,
                                uint64_t& alpha_out, Block& Delta_out) {
    uint64_t peer_alpha;
    Block    peer_delta;
    if (party == 0) {
        io->send_data(&alpha_share, sizeof(uint64_t));
        io->send_block(&delta_share, 1);
        io->flush();
        io->recv_data(&peer_alpha, sizeof(uint64_t));
        io->recv_block(&peer_delta, 1);
    } else {
        io->recv_data(&peer_alpha, sizeof(uint64_t));
        io->recv_block(&peer_delta, 1);
        io->send_data(&alpha_share, sizeof(uint64_t));
        io->send_block(&delta_share, 1);
        io->flush();
    }
    alpha_out = alpha_share ^ peer_alpha;
    Delta_out = block_xor(delta_share, peer_delta);
}

static void verify(int party, emp::NetIO* io, HalfTreeDPF& dpf,
                   const DPFKey& key, uint64_t alpha, const Block& Delta) {
    const uint64_t N = 1ULL << key.n;
    const std::string tag = "[P" + std::to_string(party) + "]";

    std::vector<Block> mine(N);
    for (uint64_t j = 0; j < N; ++j) mine[j] = dpf.eval(key, j);

    std::vector<Block> theirs(N);
    if (party == 0) {
        io->send_block(mine.data(), N); io->flush();
        io->recv_block(theirs.data(), N);
    } else {
        io->recv_block(theirs.data(), N);
        io->send_block(mine.data(), N); io->flush();
    }

    int ok = 0, bad = 0;
    for (uint64_t j = 0; j < N; ++j) {
        Block combined = block_xor(mine[j], theirs[j]);
        bool  correct  = (j == alpha) ? block_eq(combined, Delta)
                                      : block_eq(combined, ZERO_BLOCK);
        correct ? ++ok : ++bad;
    }

    Block f_alpha = block_xor(mine[alpha], theirs[alpha]);
    std::cout << tag << " Verification: " << ok << "/" << N << " correct";
    if (bad == 0)
        std::cout << "   PASS\n"
                  << tag << "   f(α=" << alpha << ") = Δ = 0x" << hex(f_alpha) << "\n";
    else
        std::cout << "   FAIL  (" << bad << " error(s))\n";
}

static void usage(const char* prog) {
    std::cerr
        << "\nUsage:\n"
        << "  P0 (server):  " << prog << " 0 <port> <n> <w> <alpha_0>\n"
        << "  P1 (client):  " << prog << " 1 <host> <port> <n> <w> <alpha_1>\n\n"
        << "  n       : domain exponent, domain size = 2^n  (1 ≤ n ≤ 25)\n"
        << "  w       : branching width, m = 2^w  (w ≥ 1, divides n)\n"
        << "  alpha_b : party b's share of α  (0 ≤ alpha_b < 2^n)\n"
        << "  α = alpha_0 ⊕ alpha_1.  Start P0 first.\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const int party = std::atoi(argv[1]);
    if (party < 0 || party > 1) { usage(argv[0]); return 1; }

    std::string host;
    int      port, n, w;
    uint64_t alpha_share;

    if (party == 0) {
        if (argc < 6) { usage(argv[0]); return 1; }
        port        = std::atoi(argv[2]);
        n           = std::atoi(argv[3]);
        w           = std::atoi(argv[4]);
        alpha_share = static_cast<uint64_t>(std::atoll(argv[5]));
    } else {
        if (argc < 7) { usage(argv[0]); return 1; }
        host        = argv[2];
        port        = std::atoi(argv[3]);
        n           = std::atoi(argv[4]);
        w           = std::atoi(argv[5]);
        alpha_share = static_cast<uint64_t>(std::atoll(argv[6]));
    }

    if (n < 1 || n > 25) { std::cerr << "Error: need 1 ≤ n ≤ 25\n"; return 1; }
    if (w < 1 || n % w != 0) {
        std::cerr << "Error: w must be ≥ 1 and divide n  (n=" << n << ", w=" << w << ")\n";
        return 1;
    }
    if (alpha_share >= (1ULL << n)) {
        std::cerr << "Error: alpha_share must be < 2^n\n"; return 1;
    }

    const int m      = 1 << w;
    const int levels = n / w;
    const std::string tag = "[P" + std::to_string(party) + "]";

    const uint8_t S_bytes[16] = {0x2b,0x7e,0x15,0x16, 0x28,0xae,0xd2,0xa6,
                                 0xab,0xf7,0x15,0x88, 0x09,0xcf,0x4f,0x3c};
    Block aes_key;
    std::memcpy(&aes_key, S_bytes, 16);

    std::cout << tag << " n=" << n << "  w=" << w << "  m=" << m
              << "  levels=" << levels << "  N=" << (1ULL << n)
              << "  α_" << party << "=" << alpha_share << "\n";
    std::cout.flush();

    if (party == 0) std::cout << tag << " Listening on port " << port << " ...\n";
    else            std::cout << tag << " Connecting to " << host << ":" << port << " ...\n";
    std::cout.flush();

    emp::NetIO net_io(party == 0 ? nullptr : host.c_str(), port, /*quiet=*/true);
    emp::NetIO* io = &net_io;
    std::cout << tag << " Connected.\n"; std::cout.flush();

    // ── Set up TripleGen for the Boolean MPC in F_DeltaShiftShare ────────
    // party_ is 0/1; emp expects ALICE=1, BOB=2.
    bool_circuit::TripleGen tg(party + 1, io);
    std::cout << tag << " TripleGen ready.\n"; std::cout.flush();

    // PCG_DPF_PRG=chacha8 switches the tree PRG (default AES); the CPU-vs-GPU
    // leaf-equality assertion below then gates the ChaCha8 device path.
    half_tree_dpf::PRGType prg_type = half_tree_dpf::PRGType::CHACHA8;
    if (const char* e = std::getenv("PCG_DPF_PRG")) {
        if (std::string(e) == "chacha8") prg_type = half_tree_dpf::PRGType::CHACHA8;
        else if (std::string(e) != "aes") { std::cerr << "Error: PCG_DPF_PRG must be aes|chacha8\n"; return 1; }
    }
    HalfTreeDPF dpf(party, io, n, w, aes_key, prg_type);

    Block delta_share;
    if (RAND_bytes(reinterpret_cast<uint8_t*>(&delta_share), 16) != 1)
        throw std::runtime_error("RAND_bytes failed for Δ share");

    // ── Secure DltSft computation (no α or Δ revealed) ───────────────────
    std::cout << tag << " Computing DltSft via Boolean circuit...\n"; std::cout.flush();
    DeltaShiftShareResult res = dpf.F_DeltaShiftShare(alpha_share, delta_share, tg);
    std::cout << tag << " DltSft complete.  ⟨Δ⟩_" << party
              << " = 0x" << hex(res.delta_share) << "\n"; std::cout.flush();

    // ── Exchange α, Δ for verification only ──────────────────────────────
    uint64_t alpha;
    Block    Delta;
    exchange_for_verify(party, io, alpha_share, res.delta_share, alpha, Delta);
    std::cout << tag << " α=" << alpha << "  Δ=0x" << hex(Delta)
              << "  (lsb=" << static_cast<int>(get_lsb(Delta)) << ")\n"; std::cout.flush();

    // ── Batch DPF: generate B instances concurrently ──────────────────────
    // Default B=4 (historic standalone sweep). PCG_DPF_BENCH_B overrides so the
    // crossover can be measured as a function of total work B·2^n, matching the
    // protocol's real batch (t^2 instances, e.g. 256 at t=16).
    int B = 4;
    if (const char* e = std::getenv("PCG_DPF_BENCH_B")) {
        B = std::atoi(e);
        if (B < 1 || B > 4096) { std::cerr << "Error: PCG_DPF_BENCH_B must be in [1,4096]\n"; return 1; }
    }
    const uint64_t N = 1ULL << n;
    std::cout << tag << " Generating " << B << " DPF instances (batch)...\n";
    std::cout.flush();

    // Each instance needs its own α share, Δ share, and DltSft shares.
    // Reuse the first instance from above; create B-1 more.
    std::vector<uint64_t>  alphas_share(B);
    std::vector<Block>     deltas_share(B);
    std::vector<half_tree_dpf::DeltaShiftShareResult> all_res(B);

    alphas_share[0] = alpha_share;
    all_res[0]      = std::move(res);

    for (int b = 1; b < B; ++b) {
        // Generate fresh random shares for each extra instance.
        emp::PRG prg;
        uint64_t tmp;
        prg.random_data(&tmp, sizeof(tmp));
        alphas_share[b] = tmp & ((1ULL << n) - 1);

        Block ds;
        if (RAND_bytes(reinterpret_cast<uint8_t*>(&ds), 16) != 1)
            throw std::runtime_error("RAND_bytes failed");
        deltas_share[b] = ds;

        all_res[b] = dpf.F_DeltaShiftShare(alphas_share[b], ds, tg);
    }

    // Build PartyInputs and collect per-instance α, Δ for verification.
    std::vector<half_tree_dpf::PartyInput> inps(B);
    std::vector<uint64_t> alphas(B);
    std::vector<Block>    Deltas(B);

    for (int b = 0; b < B; ++b) {
        inps[b] = dpf.setup_params(all_res[b].delta_share,
                                   std::move(all_res[b].DltSft_share));
        // Exchange α, Δ for verification only.
        exchange_for_verify(party, io, alphas_share[b],
                            all_res[b].delta_share, alphas[b], Deltas[b]);
    }

    std::cout << tag << " Running batch_gen_full_eval (B=" << B << ", "
              << levels << " levels, " << (m-1) << " CWs/level)...\n";
    std::cout.flush();

    auto t_cpu0 = std::chrono::steady_clock::now();
    auto all_leaves = dpf.batch_gen_full_eval(inps);
    auto t_cpu1 = std::chrono::steady_clock::now();

    std::cout << tag << " batch_gen_full_eval complete (cpu_ms="
              << std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count()
              << ")\n";

#ifdef PCG_ENABLE_CUDA
    // Run the GPU full-domain evaluation and assert leaf-for-leaf equality with
    // the CPU expansion. Both parties run CPU then GPU in lockstep, so the
    // per-level CW exchanges line up. Verification below then runs on the GPU
    // output, proving the device path end-to-end.
    if (pcg_cuda::is_cuda_available()) {
        dpf.set_eval_backend(half_tree_dpf::EvalBackend::GPU);
        auto t_gpu0 = std::chrono::steady_clock::now();
        auto gpu_leaves = dpf.batch_gen_full_eval(inps);   // cold: CUDA init + allocs
        auto t_gpu1 = std::chrono::steady_clock::now();
        gpu_leaves = dpf.batch_gen_full_eval(inps);        // warm: cached buffers
        auto t_gpu2 = std::chrono::steady_clock::now();
        dpf.set_eval_backend(half_tree_dpf::EvalBackend::CPU);
        std::cout << tag << " GPU batch_gen_full_eval complete (gpu_cold_ms="
                  << std::chrono::duration<double, std::milli>(t_gpu1 - t_gpu0).count()
                  << " gpu_warm_ms="
                  << std::chrono::duration<double, std::milli>(t_gpu2 - t_gpu1).count()
                  << ")\n";

        size_t mism = 0;
        for (int b = 0; b < B; ++b)
            for (uint64_t j = 0; j < N; ++j)
                if (!block_eq(all_leaves[b][j], gpu_leaves[b][j])) ++mism;

        std::cout << tag << " CPU-vs-GPU leaf equality: "
                  << (mism == 0 ? "PASS" : "FAIL")
                  << " (" << mism << " mismatches)\n";
        std::cout.flush();
        all_leaves = std::move(gpu_leaves);  // verify the GPU output below
    }
#endif

    // ── Verify all B instances ──────────────────────────────────────────
    if (n <= 20) {
        bool all_ok = true;
        for (int b = 0; b < B; ++b) {
            std::vector<Block> theirs(N);
            if (party == 0) {
                io->send_block(all_leaves[b].data(), N); io->flush();
                io->recv_block(theirs.data(), N);
            } else {
                io->recv_block(theirs.data(), N);
                io->send_block(all_leaves[b].data(), N); io->flush();
            }

            int ok = 0, bad = 0;
            for (uint64_t j = 0; j < N; ++j) {
                Block combined = block_xor(all_leaves[b][j], theirs[j]);
                bool correct = (j == alphas[b]) ? block_eq(combined, Deltas[b])
                                                : block_eq(combined, ZERO_BLOCK);
                correct ? ++ok : ++bad;
            }
            std::cout << tag << " Instance " << b << " (α=" << alphas[b]
                      << "): " << ok << "/" << N
                      << (bad == 0 ? "  PASS" : "  FAIL") << "\n";
            if (bad) all_ok = false;
        }
        std::cout << tag << " Batch verification: "
                  << (all_ok ? "ALL PASS" : "SOME FAIL") << "\n";
    }

    std::cout << tag << " Done.\n";
    return 0;
}
