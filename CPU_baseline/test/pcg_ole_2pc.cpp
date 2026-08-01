// Test for PCG-based OLE over R_p = Z_p[X]/(X^N+1) with regular noise.
// Verifies: z_0 + z_1 = x_0 · x_1 for the selected parameter set.
//
//   ALICE:  ./test_pcg_ole_2pc 1 <port>
//   BOB:    ./test_pcg_ole_2pc 2 <host> <port>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <emp-tool/emp-tool.h>

#include "common/ffp.h"
#include "pcg_ole_2pc/pcg_ole.h"

static constexpr uint64_t PRIME = 4611686018326724609ULL;
using F = FFp<PRIME>;
using pcg_ole::Poly;
using pcg_ole::OLEOutput;
using pcg_ole::PCG_OLE;
using pcg_ole::PCGParams;

static void usage(const char* prog) {
    std::cerr << "\nUsage:\n"
              << "  ALICE:  " << prog << " 1 <port> [N t w] [options]\n"
              << "  BOB:    " << prog << " 2 <host> <port> [N t w] [options]\n"
              << "Options:\n"
              << "  --N <degree>              default 128\n"
              << "  --t <weight>              default 4\n"
              << "  --w <branching bits>      default 1\n"
              << "  --poly-mul cpu|cuda-v1|cuda-gpuntt-merge|cuda-gpuntt-4step|auto"
                 "  default cpu\n"
              << "             (cuda, cuda-gpuntt aliases for cuda-gpuntt-merge)\n"
              << "  --dpf cpu|gpu             default cpu (gpu = device full-domain"
                 " eval; needs CUDA)\n"
              << "  --prg aes|chacha8         default aes\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) { usage(argv[0]); return 1; }

    int party = std::atoi(argv[1]);
    if (party != emp::ALICE && party != emp::BOB) { usage(argv[0]); return 1; }

    const char* host = nullptr;
    int port;
    int arg_offset;
    if (party == emp::ALICE) {
        port = std::atoi(argv[2]);
        arg_offset = 3;
    } else {
        if (argc < 4) { usage(argv[0]); return 1; }
        host = argv[2];
        port = std::atoi(argv[3]);
        arg_offset = 4;
    }

    int N_param = 128;
    int t_param = 4;
    int w_param = 1;
    pcg_ole::PolyMulBackend backend = pcg_ole::PolyMulBackend::CPU;
    half_tree_dpf::EvalBackend dpf_backend = half_tree_dpf::EvalBackend::CPU;
    half_tree_dpf::PRGType prg_type = half_tree_dpf::PRGType::CHACHA8;

    int pos = 0;
    for (int i = arg_offset; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) N_param = std::atoi(argv[++i]);
        else if (arg == "--t" && i + 1 < argc) t_param = std::atoi(argv[++i]);
        else if (arg == "--w" && i + 1 < argc) w_param = std::atoi(argv[++i]);
        else if (arg == "--poly-mul" && i + 1 < argc)
            backend = pcg_ole::parse_poly_mul_backend(argv[++i]);
        else if (arg == "--dpf" && i + 1 < argc)
            dpf_backend = pcg_ole::parse_dpf_backend(argv[++i]);
        else if (arg == "--prg" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "aes") prg_type = half_tree_dpf::PRGType::AES;
            else if (p == "chacha8") prg_type = half_tree_dpf::PRGType::CHACHA8;
            else { usage(argv[0]); return 1; }
        }
        else if (!arg.empty() && arg[0] != '-') {
            if (pos == 0) N_param = std::atoi(arg.c_str());
            else if (pos == 1) t_param = std::atoi(arg.c_str());
            else if (pos == 2) w_param = std::atoi(arg.c_str());
            else { usage(argv[0]); return 1; }
            ++pos;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (N_param <= 0 || (N_param & (N_param - 1)) != 0 || t_param <= 0 ||
        N_param % t_param != 0) {
        std::cerr << "invalid params: N must be a power of two divisible by t\n";
        return 1;
    }
    {
        const int D = 2 * (N_param / t_param);
        const int dpf_n = __builtin_ctz(D);
        if ((1 << dpf_n) != D || w_param <= 0 || dpf_n % w_param != 0) {
            std::cerr << "invalid params: w=" << w_param
                      << " must divide dpf_n=" << dpf_n
                      << " (D=2N/t=" << D << ")\n";
            return 1;
        }
    }
    if (backend != pcg_ole::PolyMulBackend::CPU) {
        std::string reason;
        if (!pcg_ole::cuda_poly_mul_supported<PRIME>(N_param, &reason)) {
            std::cout << "CUDA poly-mul unavailable; skipping: " << reason << std::endl;
            return 77;
        }
    }
    if (dpf_backend == half_tree_dpf::EvalBackend::GPU) {
#ifdef PCG_ENABLE_CUDA
        if (!pcg_cuda::is_cuda_available()) {
            std::cout << "GPU DPF requested but no CUDA device; skipping" << std::endl;
            return 77;
        }
#else
        std::cout << "GPU DPF requested but built without CUDA; skipping" << std::endl;
        return 77;
#endif
    }

    int party01 = party - 1;
    std::string tag = (party == emp::ALICE) ? "[ALICE]" : "[BOB]";

    std::cout << tag << " N=" << N_param << " t=" << t_param
              << " w=" << w_param
              << " poly_mul=" << pcg_ole::to_string(backend)
              << " dpf=" << pcg_ole::to_string(dpf_backend) << std::endl;

    emp::NetIO io(host, port, /*quiet=*/true);
    std::cout << tag << " Connected." << std::endl;

    bool_circuit::TripleGen tg(party, &io);
    std::cout << tag << " TripleGen ready." << std::endl;

    // Override params inside run_ole_test via a lambda wrapper.
    auto run = [&](int w, half_tree_dpf::PRGType prg) -> bool {
        PCGParams params;
        params.N = N_param;
        params.c = 2;
        params.t = t_param;
        params.w = w;
        params.prg_type = prg;
        params.poly_mul_backend = backend;
        params.dpf_eval_backend = dpf_backend;

        const int N = params.N;
        const int D = 2 * (N / params.t);

        std::cout << tag << " OLE test: N=" << N << " t=" << t_param
                  << " D=" << D << " dpf_n=" << __builtin_ctz(D)
                  << " w=" << w << std::endl;

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

        auto t0 = std::chrono::high_resolution_clock::now();
        PCG_OLE<PRIME> pcg(party01, &io, params);
        OLEOutput<PRIME> out = pcg.gen_and_expand(tg, public_a);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout << tag << " gen_and_expand: " << ms << " ms\n";

        // Exchange and verify the full ring relation.
        std::vector<uint64_t> my_x(N), my_z(N), peer_x(N), peer_z(N);
        for (int i = 0; i < N; ++i) {
            my_x[i] = out.x[i].val();
            my_z[i] = out.z[i].val();
        }
        if (party01 == 0) {
            io.send_data(my_x.data(), N * sizeof(uint64_t));
            io.send_data(my_z.data(), N * sizeof(uint64_t));
            io.flush();
            io.recv_data(peer_x.data(), N * sizeof(uint64_t));
            io.recv_data(peer_z.data(), N * sizeof(uint64_t));
        } else {
            io.recv_data(peer_x.data(), N * sizeof(uint64_t));
            io.recv_data(peer_z.data(), N * sizeof(uint64_t));
            io.send_data(my_x.data(), N * sizeof(uint64_t));
            io.send_data(my_z.data(), N * sizeof(uint64_t));
            io.flush();
        }

        Poly<PRIME> x0(N), x1(N), z_full(N);
        for (int i = 0; i < N; ++i) {
            (party01 == 0 ? x0[i] : x1[i]) = F(my_x[i]);
            (party01 == 0 ? x1[i] : x0[i]) = F(peer_x[i]);
            z_full[i] = F(my_z[i]) + F(peer_z[i]);
        }
        Poly<PRIME> product = pcg_ole::poly_mul<PRIME>(x0, x1, N);

        int mismatches = 0;
        for (int i = 0; i < N; ++i)
            if (z_full[i] != product[i]) ++mismatches;

        std::cout << tag << " verify: "
                  << (N - mismatches) << "/" << N
                  << (mismatches == 0 ? "  PASS" : "  FAIL") << "\n";
        return mismatches == 0;
    };

    bool ok = run(w_param, prg_type);
    std::cout << tag << " Result: " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
