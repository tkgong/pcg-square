// Test for GMW Boolean circuit evaluation with FerretCOT-based Beaver triples.
//
// Usage:
//   P0 (ALICE):  ./test_bool_circuit 1 7788
//   P1 (BOB):    ./test_bool_circuit 2 127.0.0.1 7788

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <emp-tool/emp-tool.h>

#include "bool_circuit/circuit.h"

using bool_circuit::BoolCircuit;
using bool_circuit::TripleGen;

// ---- test helpers -------------------------------------------------------

static bool test_and_gate(int party, emp::NetIO* io, TripleGen& tg) {
    // z = x AND y.  ALICE has x, BOB has y.
    BoolCircuit c;
    int x = c.add_input(0);   // ALICE
    int y = c.add_input(1);   // BOB
    int z = c.add_and(x, y);
    c.set_output(z);

    bool ok = true;
    for (int xi = 0; xi < 2; ++xi) {
        for (int yi = 0; yi < 2; ++yi) {
            std::vector<uint8_t> inp;
            if (party == emp::ALICE) inp = {static_cast<uint8_t>(xi)};
            else                    inp = {static_cast<uint8_t>(yi)};

            auto out = c.evaluate(party, io, tg, inp);
            uint8_t expect = xi & yi;
            if (out[0] != expect) {
                std::cerr << "  AND FAIL: " << xi << "&" << yi
                          << " got " << (int)out[0] << " expected " << (int)expect << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

static bool test_xor_not(int party, emp::NetIO* io, TripleGen& tg) {
    // z = NOT(x XOR y).  ALICE has x, BOB has y.  (= XNOR)
    BoolCircuit c;
    int x = c.add_input(0);
    int y = c.add_input(1);
    int w = c.add_xor(x, y);
    int z = c.add_not(w);
    c.set_output(z);

    bool ok = true;
    for (int xi = 0; xi < 2; ++xi) {
        for (int yi = 0; yi < 2; ++yi) {
            std::vector<uint8_t> inp;
            if (party == emp::ALICE) inp = {static_cast<uint8_t>(xi)};
            else                    inp = {static_cast<uint8_t>(yi)};

            auto out = c.evaluate(party, io, tg, inp);
            uint8_t expect = 1 ^ (xi ^ yi);
            if (out[0] != expect) {
                std::cerr << "  XNOR FAIL: ~(" << xi << "^" << yi
                          << ") got " << (int)out[0] << " expected " << (int)expect << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

static bool test_full_adder(int party, emp::NetIO* io, TripleGen& tg) {
    // 1-bit full adder: sum = a ^ b ^ cin, cout = (a&b) | (cin & (a^b))
    // ALICE has a, cin.  BOB has b.
    // Rewrite cout without OR: cout = (a&b) ^ (cin & (a^b))
    //   (works because (a&b) and cin&(a^b) are never both 1)
    BoolCircuit c;
    int a   = c.add_input(0);   // ALICE
    int b   = c.add_input(1);   // BOB
    int cin = c.add_input(0);   // ALICE

    int a_xor_b  = c.add_xor(a, b);
    int sum      = c.add_xor(a_xor_b, cin);
    int a_and_b  = c.add_and(a, b);
    int cin_axb  = c.add_and(cin, a_xor_b);
    int cout_w   = c.add_xor(a_and_b, cin_axb);

    c.set_output(sum);
    c.set_output(cout_w);

    bool ok = true;
    for (int av = 0; av < 2; ++av)
    for (int bv = 0; bv < 2; ++bv)
    for (int cv = 0; cv < 2; ++cv) {
        std::vector<uint8_t> inp;
        if (party == emp::ALICE) inp = {static_cast<uint8_t>(av), static_cast<uint8_t>(cv)};
        else                    inp = {static_cast<uint8_t>(bv)};

        auto out = c.evaluate(party, io, tg, inp);
        int total = av + bv + cv;
        uint8_t exp_sum  = total & 1;
        uint8_t exp_cout = (total >> 1) & 1;

        if (out[0] != exp_sum || out[1] != exp_cout) {
            std::cerr << "  ADDER FAIL: a=" << av << " b=" << bv << " cin=" << cv
                      << " got sum=" << (int)out[0] << " cout=" << (int)out[1]
                      << " expected " << (int)exp_sum << " " << (int)exp_cout << "\n";
            ok = false;
        }
    }
    return ok;
}

// ---- 4-bit ripple-carry adder ------------------------------------------

static bool test_4bit_adder(int party, emp::NetIO* io, TripleGen& tg) {
    // ALICE has 4-bit number A, BOB has 4-bit number B.
    // Computes A + B (5-bit output: 4 sum bits + carry-out).
    BoolCircuit c;

    int a[4], b[4];
    for (int i = 0; i < 4; ++i) a[i] = c.add_input(0);
    for (int i = 0; i < 4; ++i) b[i] = c.add_input(1);

    int sum[4], carry = -1;
    for (int i = 0; i < 4; ++i) {
        int a_xor_b = c.add_xor(a[i], b[i]);
        if (i == 0) {
            sum[i] = a_xor_b;
            carry  = c.add_and(a[i], b[i]);
        } else {
            sum[i]     = c.add_xor(a_xor_b, carry);
            int a_and_b  = c.add_and(a[i], b[i]);
            int c_and_ab = c.add_and(carry, a_xor_b);
            carry = c.add_xor(a_and_b, c_and_ab);
        }
    }

    for (int i = 0; i < 4; ++i) c.set_output(sum[i]);
    c.set_output(carry);

    bool ok = true;
    for (int av = 0; av < 16; ++av)
    for (int bv = 0; bv < 16; ++bv) {
        std::vector<uint8_t> inp;
        if (party == emp::ALICE) {
            for (int i = 0; i < 4; ++i) inp.push_back((av >> i) & 1);
        } else {
            for (int i = 0; i < 4; ++i) inp.push_back((bv >> i) & 1);
        }

        auto out = c.evaluate(party, io, tg, inp);
        int expected = av + bv;
        int got = 0;
        for (int i = 0; i < 5; ++i) got |= (out[i] << i);

        if (got != expected) {
            std::cerr << "  4-BIT ADDER FAIL: " << av << "+" << bv
                      << " got " << got << " expected " << expected << "\n";
            ok = false;
        }
    }
    return ok;
}

// ---- main ---------------------------------------------------------------

static void usage(const char* prog) {
    std::cerr << "\nUsage:\n"
              << "  ALICE:  " << prog << " 1 <port>\n"
              << "  BOB:    " << prog << " 2 <host> <port>\n\n";
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

    std::string tag = (party == emp::ALICE) ? "[ALICE]" : "[BOB]";

    emp::NetIO io(host, port, /*quiet=*/true);
    std::cout << tag << " Connected." << std::endl;

    TripleGen tg(party, &io);
    std::cout << tag << " TripleGen ready." << std::endl;

    auto run = [&](const char* name, bool (*fn)(int, emp::NetIO*, TripleGen&)) {
        bool ok = fn(party, &io, tg);
        std::cout << tag << " " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        return ok;
    };

    bool all_ok = true;
    all_ok &= run("AND gate",       test_and_gate);
    all_ok &= run("XOR+NOT (XNOR)", test_xor_not);
    all_ok &= run("1-bit adder",    test_full_adder);
    all_ok &= run("4-bit adder",    test_4bit_adder);

    std::cout << tag << " All tests: " << (all_ok ? "PASS" : "FAIL") << std::endl;
    return all_ok ? 0 : 1;
}
