// Finite field element F_p for a compile-time prime P (up to 63 bits).
// Provides operator overloading so code reads naturally:
//   FFp<P> a(3), b(7), c = a * b + a.inv();

#ifndef COMMON_FFP_H__
#define COMMON_FFP_H__

#include <cstdint>
#include <iostream>

template<uint64_t P>
class FFp {
    uint64_t v_;

    static uint64_t reduce(uint64_t x) { return x % P; }
    static uint64_t mul128(uint64_t a, uint64_t b) {
        return static_cast<uint64_t>(
            static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b) % P);
    }

public:
    FFp() : v_(0) {}
    explicit FFp(uint64_t x) : v_(reduce(x)) {}
    explicit FFp(int x) : v_(static_cast<uint64_t>(
        ((static_cast<int64_t>(x) % static_cast<int64_t>(P))
         + static_cast<int64_t>(P)) % static_cast<int64_t>(P))) {}
    explicit FFp(int64_t x) : v_(static_cast<uint64_t>(
        ((x % static_cast<int64_t>(P))
         + static_cast<int64_t>(P)) % static_cast<int64_t>(P))) {}

    static constexpr uint64_t modulus() { return P; }
    uint64_t val() const { return v_; }

    // ---- arithmetic operators -------------------------------------------
    FFp operator+(FFp o) const { uint64_t s = v_ + o.v_; return FFp::raw(s >= P ? s - P : s); }
    FFp operator-(FFp o) const { return FFp::raw(v_ >= o.v_ ? v_ - o.v_ : v_ + P - o.v_); }
    FFp operator*(FFp o) const { return FFp::raw(mul128(v_, o.v_)); }
    FFp operator-()      const { return FFp::raw(v_ == 0 ? 0 : P - v_); }

    FFp& operator+=(FFp o) { *this = *this + o; return *this; }
    FFp& operator-=(FFp o) { *this = *this - o; return *this; }
    FFp& operator*=(FFp o) { *this = *this * o; return *this; }

    bool operator==(FFp o) const { return v_ == o.v_; }
    bool operator!=(FFp o) const { return v_ != o.v_; }

    // ---- modular inverse (extended GCD) ---------------------------------
    FFp inv() const {
        int64_t t0 = 0, t1 = 1;
        int64_t r0 = static_cast<int64_t>(P);
        int64_t r1 = static_cast<int64_t>(v_);
        while (r1 != 0) {
            int64_t q = r0 / r1;
            int64_t tmp;
            tmp = t0 - q * t1; t0 = t1; t1 = tmp;
            tmp = r0 - q * r1; r0 = r1; r1 = tmp;
        }
        return FFp(t0);
    }

    // ---- convenience ----------------------------------------------------
    bool is_zero() const { return v_ == 0; }

    static FFp zero() { return FFp::raw(0); }
    static FFp one()  { return FFp::raw(1); }

    // Construct without reduction (caller guarantees v < P).
    static FFp raw(uint64_t v) { FFp f; f.v_ = v; return f; }

    friend std::ostream& operator<<(std::ostream& os, FFp x) {
        return os << x.v_;
    }
};

#endif  // COMMON_FFP_H__
