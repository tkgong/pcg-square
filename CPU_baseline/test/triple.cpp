// Test multiplication triple generation from Ring-OLE PCG.
// Verifies: c_0 + c_1 = (a_0 + a_1) · (b_0 + b_1) for all w ∈ {1, 2, 3}.
//
//   ALICE:  ./test_triple 1 <port>
//   BOB:    ./test_triple 2 <host> <port>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <emp-tool/emp-tool.h>

#include "common/ffp.h"
#include "half_tree_dpf/dpf.h"
#include "pcg_ole_2pc/triple.h"

static constexpr uint64_t PRIME = 4611686018326724609ULL;
using F = FFp<PRIME>;
using pcg_ole::TripleShare;
using pcg_ole::PCGParams;
using pcg_ole::Poly;

static bool run_triple_test(int party01, int party_emp, emp::NetIO* io,
                            bool_circuit::TripleGen& tg, int w,
                            pcg_ole::PolyMulBackend backend,
                            half_tree_dpf::PRGType prg_type = half_tree_dpf::PRGType::CHACHA8) {
    // N=128, t=4 → D=64, dpf_n=6, divisible by 1, 2, 3.
    PCGParams params;
    params.N = 128;
    params.c = 2;
    params.t = 4;
    params.w = w;
    params.prg_type = prg_type;
    params.poly_mul_backend = backend;

    const int N = params.N;
    std::string tag = "[P" + std::to_string(party01) + "]";

    std::cout << tag << " Triple test: w=" << w << " N=" << N << std::endl;

    emp::PRG pub_prg(&emp::zero_block);
    std::vector<Poly<PRIME>> public_a(params.c - 1);
    for (int i = 0; i < params.c - 1; ++i) {
        public_a[i].resize(N);
        for (int j = 0; j < N; ++j) {
            uint64_t r;
            pub_prg.random_data(&r, sizeof(r));
            public_a[i][j] = F(r);
        }
    }

    auto triples = pcg_ole::generate_triples<PRIME>(
        party01, io, params, tg, public_a);

    // Exchange shares to verify.
    std::vector<uint64_t> my_a(N), my_b(N), my_c(N);
    std::vector<uint64_t> peer_a(N), peer_b(N), peer_c(N);
    for (int i = 0; i < N; ++i) {
        my_a[i] = triples[i].a.val();
        my_b[i] = triples[i].b.val();
        my_c[i] = triples[i].c.val();
    }
    if (party_emp == emp::ALICE) {
        io->send_data(my_a.data(), N * sizeof(uint64_t));
        io->send_data(my_b.data(), N * sizeof(uint64_t));
        io->send_data(my_c.data(), N * sizeof(uint64_t));
        io->flush();
        io->recv_data(peer_a.data(), N * sizeof(uint64_t));
        io->recv_data(peer_b.data(), N * sizeof(uint64_t));
        io->recv_data(peer_c.data(), N * sizeof(uint64_t));
    } else {
        io->recv_data(peer_a.data(), N * sizeof(uint64_t));
        io->recv_data(peer_b.data(), N * sizeof(uint64_t));
        io->recv_data(peer_c.data(), N * sizeof(uint64_t));
        io->send_data(my_a.data(), N * sizeof(uint64_t));
        io->send_data(my_b.data(), N * sizeof(uint64_t));
        io->send_data(my_c.data(), N * sizeof(uint64_t));
        io->flush();
    }

    int ok = 0;
    for (int i = 0; i < N; ++i) {
        F a = F(my_a[i]) + F(peer_a[i]);
        F b = F(my_b[i]) + F(peer_b[i]);
        F c = F(my_c[i]) + F(peer_c[i]);
        if (c == a * b) ++ok;
    }

    std::cout << tag << " w=" << w << ": " << ok << "/" << N
              << (ok == N ? "  PASS" : "  FAIL") << std::endl;
    return ok == N;
}

static void usage(const char* prog) {
    std::cerr << "\nUsage:\n"
              << "  ALICE:  " << prog << " 1 <port> [--poly-mul cpu|cuda-v1|cuda-gpuntt-merge|cuda-gpuntt-4step|auto]\n"
              << "  BOB:    " << prog << " 2 <host> <port> [--poly-mul cpu|cuda-v1|cuda-gpuntt-merge|cuda-gpuntt-4step|auto]\n"
              << "  (cuda, cuda-gpuntt aliases for cuda-gpuntt-merge)\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) { usage(argv[0]); return 1; }

    int party = std::atoi(argv[1]);
    if (party != emp::ALICE && party != emp::BOB) { usage(argv[0]); return 1; }

    const char* host = nullptr;
    int port;
    if (party == emp::ALICE) {
        port = std::atoi(argv[2]);
    } else {
        if (argc < 4) { usage(argv[0]); return 1; }
        host = argv[2];
        port = std::atoi(argv[3]);
    }

    int arg_offset = (party == emp::ALICE) ? 3 : 4;
    pcg_ole::PolyMulBackend backend = pcg_ole::PolyMulBackend::CPU;
    for (int i = arg_offset; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--poly-mul" && i + 1 < argc)
            backend = pcg_ole::parse_poly_mul_backend(argv[++i]);
        else {
            usage(argv[0]);
            return 1;
        }
    }

    if (backend != pcg_ole::PolyMulBackend::CPU) {
        std::string reason;
        if (!pcg_ole::cuda_poly_mul_supported<PRIME>(128, &reason)) {
            std::cout << "CUDA poly-mul unavailable; skipping: " << reason << std::endl;
            return 77;
        }
    }

    int party01 = party - 1;
    std::string tag = (party == emp::ALICE) ? "[ALICE]" : "[BOB]";
    std::cout << tag << " poly_mul=" << pcg_ole::to_string(backend) << std::endl;

    emp::NetIO io(host, port, /*quiet=*/true);
    std::cout << tag << " Connected." << std::endl;

    bool_circuit::TripleGen tg(party, &io);
    std::cout << tag << " TripleGen ready." << std::endl;

    bool all_ok = true;
    for (auto prg : {half_tree_dpf::PRGType::AES, half_tree_dpf::PRGType::CHACHA8}) {
        const char* name = (prg == half_tree_dpf::PRGType::AES) ? "AES" : "ChaCha8";
        for (int w : {1, 2, 3}) {
            std::cout << tag << " --- PRG=" << name << " w=" << w << " ---" << std::endl;
            all_ok &= run_triple_test(party01, party, &io, tg, w, backend, prg);
        }
    }

    std::cout << tag << " All triple tests: " << (all_ok ? "PASS" : "FAIL") << std::endl;
    return all_ok ? 0 : 1;
}
