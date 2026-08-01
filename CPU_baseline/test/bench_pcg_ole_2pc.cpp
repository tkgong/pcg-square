#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <emp-tool/emp-tool.h>

#include "common/comm_trace.h"
#include "common/ffp.h"
#include "common/prof.h"
#include "common/timing.h"
#include "pcg_ole_2pc/pcg_ole.h"

namespace {

constexpr uint64_t PRIME = 4611686018326724609ULL;
using F = FFp<PRIME>;
using pcg_ole::Poly;

struct Options {
    int N = 128;
    int c = 2;
    int t = 4;
    int w = 1;
    int warmup = 0;
    int iters = 1;
    half_tree_dpf::PRGType prg = half_tree_dpf::PRGType::CHACHA8;
    pcg_ole::PolyMulBackend backend = pcg_ole::PolyMulBackend::CPU;
    half_tree_dpf::EvalBackend dpf = half_tree_dpf::EvalBackend::CPU;
    std::string profile_csv;
};

struct Stats {
    double mean_ms = 0.0;
    double std_ms = 0.0;
};

void usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  ALICE: " << prog << " 1 <port> [options]\n"
              << "  BOB:   " << prog << " 2 <host> <port> [options]\n"
              << "Options:\n"
              << "  --N <degree> --c <compression> --t <weight> --w <branch bits>\n"
              << "  --prg aes|chacha8\n"
              << "  --warmup <count>\n"
              << "  --iters <count>\n"
              << "  --poly-mul cpu|cuda-v1|cuda-gpuntt-merge|cuda-gpuntt-4step|auto"
                 "  (cuda, cuda-gpuntt aliases for cuda-gpuntt-merge)\n"
              << "  --dpf cpu|gpu  (gpu = device full-domain eval; needs CUDA)\n"
              << "  --profile-csv <path>\n";
}

Options parse_options(int argc, char** argv, int offset) {
    Options opts;
    for (int i = offset; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) opts.N = std::atoi(argv[++i]);
        else if (arg == "--c" && i + 1 < argc) opts.c = std::atoi(argv[++i]);
        else if (arg == "--t" && i + 1 < argc) opts.t = std::atoi(argv[++i]);
        else if (arg == "--w" && i + 1 < argc) opts.w = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) opts.warmup = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) opts.iters = std::atoi(argv[++i]);
        else if (arg == "--poly-mul" && i + 1 < argc)
            opts.backend = pcg_ole::parse_poly_mul_backend(argv[++i]);
        else if (arg == "--dpf" && i + 1 < argc)
            opts.dpf = pcg_ole::parse_dpf_backend(argv[++i]);
        else if (arg == "--profile-csv" && i + 1 < argc)
            opts.profile_csv = argv[++i];
        else if (arg == "--prg" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "aes") opts.prg = half_tree_dpf::PRGType::AES;
            else if (p == "chacha8") opts.prg = half_tree_dpf::PRGType::CHACHA8;
            else throw std::runtime_error("unknown PRG: " + p);
        } else {
            throw std::runtime_error("unknown or incomplete option: " + arg);
        }
    }
    return opts;
}

Stats summarize(const std::vector<double>& samples) {
    Stats s;
    if (samples.empty()) return s;
    s.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double var = 0.0;
    for (double x : samples) {
        double d = x - s.mean_ms;
        var += d * d;
    }
    s.std_ms = std::sqrt(var / samples.size());
    return s;
}

std::string prg_name(half_tree_dpf::PRGType prg) {
    return prg == half_tree_dpf::PRGType::AES ? "aes" : "chacha8";
}

std::string profile_path_for_party(const std::string& path, int party01) {
    if (path.empty()) return "";
    std::string suffix = ".p" + std::to_string(party01);
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return path + suffix + ".csv";
    return path.substr(0, dot) + suffix + path.substr(dot);
}

std::vector<Poly<PRIME>> make_public_a(int c, int N) {
    emp::PRG pub_prg(&emp::zero_block);
    std::vector<Poly<PRIME>> public_a(c - 1);
    for (int i = 0; i < c - 1; ++i) {
        public_a[i].resize(N);
        for (int j = 0; j < N; ++j) {
            uint64_t r;
            pub_prg.random_data(&r, sizeof(r));
            public_a[i][j] = F(r);
        }
    }
    return public_a;
}

bool verify_ole(int party01, emp::NetIO& io, const pcg_ole::OLEOutput<PRIME>& out) {
    const int N = static_cast<int>(out.x.size());
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
    auto product = pcg_ole::poly_mul<PRIME>(x0, x1, N);
    return product == z_full;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            usage(argv[0]);
            return 1;
        }
        int party = std::atoi(argv[1]);
        if (party != emp::ALICE && party != emp::BOB) {
            usage(argv[0]);
            return 1;
        }

        const char* host = nullptr;
        int port = 0;
        int offset = 0;
        if (party == emp::ALICE) {
            port = std::atoi(argv[2]);
            offset = 3;
        } else {
            if (argc < 4) {
                usage(argv[0]);
                return 1;
            }
            host = argv[2];
            port = std::atoi(argv[3]);
            offset = 4;
        }

        Options opts = parse_options(argc, argv, offset);
        if (opts.backend != pcg_ole::PolyMulBackend::CPU) {
            std::string reason;
            if (!pcg_ole::cuda_poly_mul_supported<PRIME>(opts.N, &reason)) {
                std::cout << "CUDA poly-mul unavailable; skipping: " << reason << "\n";
                return 77;
            }
        }
        if (opts.dpf == half_tree_dpf::EvalBackend::GPU) {
#ifdef PCG_ENABLE_CUDA
            if (!pcg_cuda::is_cuda_available()) {
                std::cout << "GPU DPF requested but no CUDA device; skipping\n";
                return 77;
            }
#else
            std::cout << "GPU DPF requested but built without CUDA; skipping\n";
            return 77;
#endif
        }

        if (opts.N <= 0 || (opts.N & (opts.N - 1)) != 0 || opts.t <= 0 ||
            opts.N % opts.t != 0 || opts.c <= 0 || opts.warmup < 0 || opts.iters <= 0) {
            throw std::runtime_error("invalid PCG parameters");
        }
        int D = 2 * (opts.N / opts.t);
        int dpf_n = __builtin_ctz(D);
        if ((1 << dpf_n) != D || opts.w <= 0 || dpf_n % opts.w != 0)
            throw std::runtime_error("invalid DPF domain or w does not divide dpf_n");

        int party01 = party - 1;
        std::string tag = party == emp::ALICE ? "[ALICE]" : "[BOB]";
        pcg_profile::configure(profile_path_for_party(opts.profile_csv, party01), tag);
        // Per-exchange network trace (same --profile-csv stem, ".comm" infix);
        // only emitted when built with -DPCG_ENABLE_PROFILING.
        if (!opts.profile_csv.empty()) {
            std::string cp = profile_path_for_party(opts.profile_csv, party01);
            size_t dot = cp.find_last_of('.');
            std::string comm_path = (dot == std::string::npos)
                ? cp + ".comm.csv" : cp.substr(0, dot) + ".comm" + cp.substr(dot);
            pcg_comm::configure(comm_path, tag);
        }

        std::cout << tag << " bench N=" << opts.N << " c=" << opts.c
                  << " t=" << opts.t << " w=" << opts.w
                  << " prg=" << prg_name(opts.prg)
                  << " poly_mul=" << pcg_ole::to_string(opts.backend)
                  << " dpf=" << pcg_ole::to_string(opts.dpf)
                  << " warmup=" << opts.warmup
                  << " iters=" << opts.iters << "\n";

        emp::NetIO io(host, port, true);
        bool_circuit::TripleGen tg(party, &io);

        std::vector<double> samples;
        bool all_ok = true;
        auto run_once = [&](const char* sample, int iter, bool timed) {
            std::ostringstream ctx;
            ctx << "sample=" << sample
                << ";iter=" << iter
                << ";N=" << opts.N
                << ";c=" << opts.c
                << ";t=" << opts.t
                << ";w=" << opts.w
                << ";prg=" << prg_name(opts.prg)
                << ";poly_mul=" << pcg_ole::to_string(opts.backend)
                << ";dpf=" << pcg_ole::to_string(opts.dpf);
            pcg_profile::set_context(ctx.str());
            pcg_comm::set_context(ctx.str());
            const uint64_t bytes_before = static_cast<uint64_t>(io.counter);

            pcg_ole::PCGParams params{opts.N, opts.c, opts.t, opts.w, opts.prg,
                                      opts.backend, opts.dpf};
            auto public_a = make_public_a(opts.c, opts.N);
            pcg_ole::PCG_OLE<PRIME> pcg(party01, &io, params);

            auto t0 = std::chrono::steady_clock::now();
            auto out = pcg.gen_and_expand(tg, public_a);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Sent-byte total for the protocol only (verify_ole excluded).
            const uint64_t proto_bytes =
                static_cast<uint64_t>(io.counter) - bytes_before;

            bool ok = verify_ole(party01, io, out);
            all_ok &= ok;
            std::cout << tag << ' ' << sample << "=" << iter;
            if (timed) {
                samples.push_back(ms);
                std::cout << " ms=" << ms;
            }
            std::cout << " sent_bytes=" << proto_bytes
                      << " verify=" << (ok ? "PASS" : "FAIL") << "\n";
        };

        for (int iter = 0; iter < opts.warmup; ++iter) {
            run_once("warmup", iter, false);
        }
        for (int iter = 0; iter < opts.iters; ++iter) {
            run_once("timed", iter, true);
        }

        PCG_PROF_REPORT();

        Stats stats = summarize(samples);
        std::cout << "bench_result," << tag << ','
                  << opts.N << ',' << opts.c << ',' << opts.t << ',' << opts.w << ','
                  << prg_name(opts.prg) << ',' << pcg_ole::to_string(opts.backend) << ','
                  << pcg_ole::to_string(opts.dpf) << ','
                  << opts.warmup << ',' << opts.iters << ','
                  << stats.mean_ms << ',' << stats.std_ms << ','
                  << (all_ok ? 1 : 0) << "\n";
        return all_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
