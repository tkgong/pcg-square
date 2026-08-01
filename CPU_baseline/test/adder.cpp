// Test for Kogge-Stone parallel-prefix adder (logarithmic AND depth).
//
// ALICE inputs an n-bit integer A, BOB inputs an n-bit integer B.
// The circuit computes A + B as a secret-shared (n+1)-bit integer.
// To verify, shares are reconstructed and compared with plaintext A + B.
//
// Usage:
//   ALICE:  ./test_adder 1 <port> [n]
//   BOB:    ./test_adder 2 <host> <port> [n]
//   n defaults to 32.

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <emp-tool/emp-tool.h>

#include "bool_circuit/circuit.h"
#include "bool_circuit/adder.h"

using bool_circuit::BoolCircuit;
using bool_circuit::TripleGen;
using bool_circuit::build_kogge_stone_adder;

static uint64_t bits_to_int(const std::vector<uint8_t>& bits) {
    uint64_t v = 0;
    for (size_t i = 0; i < bits.size(); ++i)
        v |= static_cast<uint64_t>(bits[i] & 1) << i;
    return v;
}

static std::vector<uint8_t> int_to_bits(uint64_t v, int n) {
    std::vector<uint8_t> bits(n);
    for (int i = 0; i < n; ++i) bits[i] = (v >> i) & 1;
    return bits;
}

// Exhaustive test for small n (n <= 10).
static bool test_exhaustive(int party, emp::NetIO* io, TripleGen& tg, int n) {
    BoolCircuit c;
    std::vector<int> a(n), b(n);
    for (int i = 0; i < n; ++i) a[i] = c.add_input(0);
    for (int i = 0; i < n; ++i) b[i] = c.add_input(1);
    auto sum_wires = build_kogge_stone_adder(c, n, a, b);
    for (int i = 0; i <= n; ++i) c.set_output(sum_wires[i]);

    std::cout << "[P" << party << "] n=" << n
              << "  AND gates=" << c.num_and_gates()
              << "  wires=" << c.num_wires() << std::endl;

    uint64_t N = 1ULL << n;
    int fails = 0;
    for (uint64_t av = 0; av < N && fails < 10; ++av) {
        for (uint64_t bv = 0; bv < N && fails < 10; ++bv) {
            std::vector<uint8_t> inp;
            if (party == emp::ALICE) inp = int_to_bits(av, n);
            else                    inp = int_to_bits(bv, n);

            auto out = c.evaluate(party, io, tg, inp);
            uint64_t got = bits_to_int(out);
            uint64_t expected = av + bv;

            if (got != expected) {
                std::cerr << "  FAIL: " << av << "+" << bv
                          << " = " << got << " (expected " << expected << ")\n";
                ++fails;
            }
        }
    }
    return fails == 0;
}

// Spot-check for larger n: random inputs, verify with plaintext addition.
// Also demonstrates evaluate_shared(): output stays secret-shared, then
// reconstructed only for verification.
static bool test_random(int party, emp::NetIO* io, TripleGen& tg, int n,
                        int num_tests = 100) {
    BoolCircuit c;
    std::vector<int> a(n), b(n);
    for (int i = 0; i < n; ++i) a[i] = c.add_input(0);
    for (int i = 0; i < n; ++i) b[i] = c.add_input(1);
    auto sum_wires = build_kogge_stone_adder(c, n, a, b);
    for (int i = 0; i <= n; ++i) c.set_output(sum_wires[i]);

    std::cout << "[P" << party << "] n=" << n
              << "  AND gates=" << c.num_and_gates()
              << "  wires=" << c.num_wires() << std::endl;

    emp::PRG prg;
    uint64_t mask = (n >= 64) ? ~0ULL : ((1ULL << n) - 1);
    int fails = 0;

    for (int t = 0; t < num_tests && fails < 10; ++t) {
        uint64_t my_val;
        prg.random_data(&my_val, sizeof(my_val));
        my_val &= mask;

        // Exchange plaintext values for verification only.
        uint64_t peer_val;
        if (party == emp::ALICE) {
            io->send_data(&my_val, sizeof(uint64_t));
            io->flush();
            io->recv_data(&peer_val, sizeof(uint64_t));
        } else {
            io->recv_data(&peer_val, sizeof(uint64_t));
            io->send_data(&my_val, sizeof(uint64_t));
            io->flush();
        }

        uint64_t alice_val = (party == emp::ALICE) ? my_val : peer_val;
        uint64_t bob_val   = (party == emp::BOB)   ? my_val : peer_val;

        auto inp = int_to_bits(my_val, n);

        // Use evaluate_shared: each party gets their XOR share.
        auto my_share = c.evaluate_shared(party, io, tg, inp);

        // Reconstruct for verification (exchange shares).
        std::vector<uint8_t> peer_share(my_share.size());
        if (party == emp::ALICE) {
            io->send_data(my_share.data(), my_share.size());
            io->flush();
            io->recv_data(peer_share.data(), peer_share.size());
        } else {
            io->recv_data(peer_share.data(), peer_share.size());
            io->send_data(my_share.data(), my_share.size());
            io->flush();
        }

        std::vector<uint8_t> result(my_share.size());
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = my_share[i] ^ peer_share[i];

        uint64_t got = bits_to_int(result);
        uint64_t expected = alice_val + bob_val;

        if (got != expected) {
            std::cerr << "  FAIL: " << alice_val << " + " << bob_val
                      << " = " << got << " (expected " << expected << ")\n";
            ++fails;
        }
    }
    return fails == 0;
}

static void usage(const char* prog) {
    std::cerr << "\nUsage:\n"
              << "  ALICE:  " << prog << " 1 <port> [n]\n"
              << "  BOB:    " << prog << " 2 <host> <port> [n]\n"
              << "  n: bit width (default 32)\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) { usage(argv[0]); return 1; }

    int party = std::atoi(argv[1]);
    if (party != emp::ALICE && party != emp::BOB) { usage(argv[0]); return 1; }

    const char* host = nullptr;
    int port, n = 32;
    if (party == emp::ALICE) {
        port = std::atoi(argv[2]);
        if (argc >= 4) n = std::atoi(argv[3]);
    } else {
        if (argc < 4) { usage(argv[0]); return 1; }
        host = argv[2];
        port = std::atoi(argv[3]);
        if (argc >= 5) n = std::atoi(argv[4]);
    }

    std::string tag = "[P" + std::to_string(party) + "]";

    emp::NetIO io(host, port, /*quiet=*/true);
    std::cout << tag << " Connected." << std::endl;

    TripleGen tg(party, &io);
    std::cout << tag << " TripleGen ready." << std::endl;

    bool all_ok = true;

    // Exhaustive for small n.
    if (n <= 8) {
        std::cout << tag << " Exhaustive test (n=" << n << ")..." << std::endl;
        bool ok = test_exhaustive(party, &io, tg, n);
        std::cout << tag << " Exhaustive: " << (ok ? "PASS" : "FAIL") << std::endl;
        all_ok &= ok;
    }

    // Random spot-check.
    int num_tests = (n <= 8) ? 50 : 200;
    std::cout << tag << " Random test (n=" << n << ", " << num_tests << " trials)..."
              << std::endl;
    bool ok = test_random(party, &io, tg, n, num_tests);
    std::cout << tag << " Random: " << (ok ? "PASS" : "FAIL") << std::endl;
    all_ok &= ok;

    std::cout << tag << " All: " << (all_ok ? "PASS" : "FAIL") << std::endl;
    return all_ok ? 0 : 1;
}
