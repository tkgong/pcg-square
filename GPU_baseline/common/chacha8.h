// ChaCha8 stream cipher / PRG.
// Derived from D.J. Bernstein's ChaCha reference (public domain) via
// libsodium's crypto_stream_chacha20/ref/chacha20_ref.c, with the round
// count reduced from 20 (10 double-rounds) to 8 (4 double-rounds).
//
// ChaCha8 is used as a fast PRG for DPF tree expansion and similar
// crypto primitives where full ChaCha20 security is not needed.
//
// Interface:
//   ChaCha8PRG prg(key_32bytes);
//   prg.fill(output_buffer, nbytes);           // counter-mode PRG
//   prg.fill_blocks(output_buffer, nblocks);   // 64-byte blocks

#ifndef COMMON_CHACHA8_H__
#define COMMON_CHACHA8_H__

#include <cstdint>
#include <cstring>

class ChaCha8PRG {
public:
    // Initialize with a 256-bit (32-byte) key.  Nonce is set to 0.
    explicit ChaCha8PRG(const uint8_t key[32]) {
        // "expand 32-byte k" constant.
        state_[0]  = 0x61707865u;
        state_[1]  = 0x3320646eu;
        state_[2]  = 0x79622d32u;
        state_[3]  = 0x6b206574u;
        state_[4]  = load32_le(key +  0);
        state_[5]  = load32_le(key +  4);
        state_[6]  = load32_le(key +  8);
        state_[7]  = load32_le(key + 12);
        state_[8]  = load32_le(key + 16);
        state_[9]  = load32_le(key + 20);
        state_[10] = load32_le(key + 24);
        state_[11] = load32_le(key + 28);
        state_[12] = 0;   // counter low
        state_[13] = 0;   // counter high
        state_[14] = 0;   // nonce low
        state_[15] = 0;   // nonce high
    }

    // Set 64-bit nonce (counter resets to 0).
    void set_nonce(uint64_t nonce) {
        state_[12] = 0;
        state_[13] = 0;
        state_[14] = static_cast<uint32_t>(nonce);
        state_[15] = static_cast<uint32_t>(nonce >> 32);
    }

    // Set 64-bit counter directly.
    void set_counter(uint64_t ctr) {
        state_[12] = static_cast<uint32_t>(ctr);
        state_[13] = static_cast<uint32_t>(ctr >> 32);
    }

    // Generate `nblocks` full 64-byte blocks into `out`.
    void fill_blocks(void* out, size_t nblocks) {
        auto* p = static_cast<uint8_t*>(out);
        for (size_t i = 0; i < nblocks; ++i) {
            chacha8_block(p);
            p += 64;
            // Increment 64-bit counter.
            if (++state_[12] == 0) ++state_[13];
        }
    }

    // Generate `nbytes` bytes (handles partial last block).
    void fill(void* out, size_t nbytes) {
        auto* p = static_cast<uint8_t*>(out);
        while (nbytes >= 64) {
            chacha8_block(p);
            p += 64;
            nbytes -= 64;
            if (++state_[12] == 0) ++state_[13];
        }
        if (nbytes > 0) {
            uint8_t tmp[64];
            chacha8_block(tmp);
            std::memcpy(p, tmp, nbytes);
            if (++state_[12] == 0) ++state_[13];
        }
    }

private:
    uint32_t state_[16];

    static uint32_t rotl32(uint32_t v, int n) {
        return (v << n) | (v >> (32 - n));
    }

    static uint32_t load32_le(const uint8_t* p) {
        return static_cast<uint32_t>(p[0])
             | (static_cast<uint32_t>(p[1]) << 8)
             | (static_cast<uint32_t>(p[2]) << 16)
             | (static_cast<uint32_t>(p[3]) << 24);
    }

    static void store32_le(uint8_t* p, uint32_t v) {
        p[0] = static_cast<uint8_t>(v);
        p[1] = static_cast<uint8_t>(v >> 8);
        p[2] = static_cast<uint8_t>(v >> 16);
        p[3] = static_cast<uint8_t>(v >> 24);
    }

#define CHACHA8_QR(a, b, c, d) \
    a += b; d ^= a; d = rotl32(d, 16); \
    c += d; b ^= c; b = rotl32(b, 12); \
    a += b; d ^= a; d = rotl32(d, 8);  \
    c += d; b ^= c; b = rotl32(b, 7);

    // Produce one 64-byte keystream block.
    void chacha8_block(uint8_t out[64]) const {
        uint32_t x0  = state_[0],  x1  = state_[1],  x2  = state_[2],  x3  = state_[3];
        uint32_t x4  = state_[4],  x5  = state_[5],  x6  = state_[6],  x7  = state_[7];
        uint32_t x8  = state_[8],  x9  = state_[9],  x10 = state_[10], x11 = state_[11];
        uint32_t x12 = state_[12], x13 = state_[13], x14 = state_[14], x15 = state_[15];

        // 4 double-rounds = 8 rounds.
        for (int i = 8; i > 0; i -= 2) {
            // Column round.
            CHACHA8_QR(x0, x4, x8,  x12)
            CHACHA8_QR(x1, x5, x9,  x13)
            CHACHA8_QR(x2, x6, x10, x14)
            CHACHA8_QR(x3, x7, x11, x15)
            // Diagonal round.
            CHACHA8_QR(x0, x5, x10, x15)
            CHACHA8_QR(x1, x6, x11, x12)
            CHACHA8_QR(x2, x7, x8,  x13)
            CHACHA8_QR(x3, x4, x9,  x14)
        }

        // Add initial state (ChaCha final addition).
        store32_le(out +  0, x0  + state_[0]);
        store32_le(out +  4, x1  + state_[1]);
        store32_le(out +  8, x2  + state_[2]);
        store32_le(out + 12, x3  + state_[3]);
        store32_le(out + 16, x4  + state_[4]);
        store32_le(out + 20, x5  + state_[5]);
        store32_le(out + 24, x6  + state_[6]);
        store32_le(out + 28, x7  + state_[7]);
        store32_le(out + 32, x8  + state_[8]);
        store32_le(out + 36, x9  + state_[9]);
        store32_le(out + 40, x10 + state_[10]);
        store32_le(out + 44, x11 + state_[11]);
        store32_le(out + 48, x12 + state_[12]);
        store32_le(out + 52, x13 + state_[13]);
        store32_le(out + 56, x14 + state_[14]);
        store32_le(out + 60, x15 + state_[15]);
    }

#undef CHACHA8_QR
};

#endif  // COMMON_CHACHA8_H__
