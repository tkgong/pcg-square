// Functional check of one configuration (TOP = p_pu__uni / p_pu__spu / p_pu__aim; -DHAS_ARX / -DHAS_BF16).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>
#include DESIGN_H
using namespace cxxrtl_design;
static TOP top;
static uint64_t cycles = 0;
static void tick() { top.p_clk.set<bool>(false); top.step(); top.p_clk.set<bool>(true); top.step(); cycles++; }
static void crf(int a, uint32_t w) { top.p_crf__we.set<bool>(true); top.p_crf__wa.set<uint32_t>(a); top.p_crf__wd.set<uint32_t>(w); tick(); top.p_crf__we.set<bool>(false); }
static void srf(int a, uint32_t w) { top.p_srf__we.set<bool>(true); top.p_srf__wa.set<uint32_t>(a); top.p_srf__wd.set<uint32_t>(w); tick(); top.p_srf__we.set<bool>(false); }
static void setdin(const uint32_t *w) { for (int i = 0; i < 8; i++) top.p_din.data[i] = w[i]; }
static void getdout(uint32_t *w) { for (int i = 0; i < 8; i++) w[i] = top.p_dout.data[i]; }
static void stepi() { top.p_step.set<bool>(true); tick(); top.p_step.set<bool>(false); }
static uint32_t ins(int opc, int dg, int da, int sa, int sb, int f16 = 0, int sidx = 0, int rot = 0, int fl = 0) {
  return (uint32_t) opc << 27 | dg << 26 | da << 23 | sa << 20 | sb << 17 | f16 << 16 | sidx << 11 | rot << 8 | fl; }
static inline uint32_t rotl(uint32_t x, int r) { return x << r | x >> (32 - r); }
static void qr(uint32_t *x, int a, int b, int c, int d) {
  x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 12);
  x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 8); x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 7); }
static float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
int main() {
  std::mt19937 rng(12345); int bad = 0;
  top.p_rst.set<bool>(true); tick(); tick(); top.p_rst.set<bool>(false);
#ifdef HAS_ARX
  uint32_t K[16]; const uint32_t sig[4] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
  for (int i = 0; i < 4; i++) K[i] = sig[i];
  for (int i = 4; i < 16; i++) K[i] = rng();
  for (int w = 0; w < 8; w++) { srf(2 * w, K[4 + w] & 0xffff); srf(2 * w + 1, K[4 + w] >> 16); }
  for (int j = 0; j < 4; j++) srf(16 + j, K[12 + j]);
  crf(0, ins(18, 0, 0, 0, 0)); crf(1, ins(18, 0, 0, 0, 0)); crf(2, ins(18, 0, 0, 0, 0)); crf(3, ins(9, 0, 0, 0, 0));
  top.p_pc__rst.set<bool>(true); tick(); top.p_pc__rst.set<bool>(false);
  const int NB = 3; uint32_t seed[NB][8][4], col[8], exp[NB][8][8];
  for (int b = 0; b < NB; b++) for (int l = 0; l < 8; l++) {
    for (int j = 0; j < 4; j++) seed[b][l][j] = rng();
    uint32_t x[16]; for (int i = 0; i < 12; i++) x[i] = K[i]; for (int j = 0; j < 4; j++) x[12 + j] = seed[b][l][j];
    for (int r = 0; r < 4; r++) { qr(x, 0, 4, 8, 12); qr(x, 1, 5, 9, 13); qr(x, 2, 6, 10, 14); qr(x, 3, 7, 11, 15);
                                  qr(x, 0, 5, 10, 15); qr(x, 1, 6, 11, 12); qr(x, 2, 7, 8, 13); qr(x, 3, 4, 9, 14); }
    for (int j = 0; j < 4; j++) { uint32_t L = (x[j] + K[j]) ^ ((seed[b][l][0] & 1) ? K[12 + j] : 0), R = L ^ seed[b][l][j];
      exp[b][2 * j + l / 4][2 * (l % 4)] = L; exp[b][2 * j + l / 4][2 * (l % 4) + 1] = R; }
  }
  auto push = [&](int b) { for (int j = 0; j < 4; j++) { for (int l = 0; l < 8; l++) col[l] = seed[b][l][j]; setdin(col);
                                                         top.p_sk__push.set<bool>(true); tick(); top.p_sk__push.set<bool>(false); tick(); tick(); tick(); } };
  auto drain = [&](int b) { for (int c = 0; c < 8; c++) { for (int l = 0; l < 8; l++) col[l] = seed[b][l][c / 2]; setdin(col);   // parent re-read
                              top.p_sq__pop.set<bool>(true); tick(); top.p_sq__pop.set<bool>(false); getdout(col);
                              for (int i = 0; i < 8; i++) if (col[i] != exp[b][c][i]) bad++; tick(); tick(); tick(); } };
  push(0);
  for (int b = 0; b < NB; b++) {
    stepi(); uint64_t c0 = cycles; int n = 0;                 // EXTEND for batch b
    if (b > 0) drain(b - 1); else for (int i = 0; i < 12; i++) tick();   // previous batch leaves during this one; seeds stay in the skid for 12 ticks
    if (b + 1 < NB) push(b + 1);                              // next batch's seeds arrive during this one
    n = (int) (cycles - c0);
    while (top.p_busy.get<bool>()) { tick(); n++; }
    printf("EXTEND batch %d: busy for %s%d ticks\n", b, n > 148 ? "<= " : "", n);
    if (b == 0) { /* exact length measured separately below */ }
  }
  drain(NB - 1);
  // exact length of one EXTEND
  for (int l = 0; l < 4; l++) { top.p_sk__push.set<bool>(true); tick(); top.p_sk__push.set<bool>(false); }
  top.p_pc__rst.set<bool>(true); tick(); top.p_pc__rst.set<bool>(false);
  stepi(); int n = 0; while (top.p_busy.get<bool>()) { tick(); n++; }
  printf("EXTEND length: %d ticks (model: 155)\n", n);
  printf("ChaCha8 EXTEND, %d batches back to back, %d child columns: %d word mismatches\n", NB, NB * 8, bad);
  { // mod-ALU hooks, single-stepped: REDP (mask + fold, conditional subtract), ADDP (add, conditional subtract)
    const uint64_t P = 0x3FFFFFFFFA000001ull; int mbad = 0, p = 0;
    crf(p++, ins(1, 0, 0, 0, 0)); crf(p++, ins(1, 1, 0, 0, 0));
    crf(p++, ins(16, 0, 1, 0, 0, 0, 0, 0, 0x40 | 0x20 | 0x10));          // G0[1] = lo62(G0[0]) + fold(hi2)
    crf(p++, ins(16, 0, 1, 1, 0, 0, 8, 0, 0x80 | 0x20 | 0x02));          // G0[1] -= P where >= P
    crf(p++, ins(16, 0, 2, 1, 0, 0, 0, 0, 0x20));                        // G0[2] = G0[1] + G1[0]
    crf(p++, ins(16, 0, 2, 2, 0, 0, 8, 0, 0x80 | 0x20 | 0x02));          // G0[2] -= P where >= P
    crf(p++, ins(2, 0, 0, 1, 0)); crf(p++, ins(2, 0, 0, 2, 0)); crf(p++, ins(9, 0, 0, 0, 0));
    for (int t = 0; t < 2000; t++) {
      top.p_pc__rst.set<bool>(true); tick(); top.p_pc__rst.set<bool>(false);
      uint64_t x[4], y[4]; uint32_t w[8], o[8];
      for (int k = 0; k < 4; k++) { x[k] = (uint64_t) rng() << 32 | rng(); y[k] = ((uint64_t) rng() << 32 | rng()) % P; if (t % 5 == 0) x[k] |= 0xC000000000000000ull; if (t % 11 == 0) x[k] = P + k - 1; }
      for (int k = 0; k < 4; k++) { w[2 * k] = (uint32_t) x[k]; w[2 * k + 1] = (uint32_t) (x[k] >> 32); } setdin(w); stepi();
      for (int k = 0; k < 4; k++) { w[2 * k] = (uint32_t) y[k]; w[2 * k + 1] = (uint32_t) (y[k] >> 32); } setdin(w); stepi();
      for (int i = 0; i < 4; i++) stepi();
      stepi(); getdout(o); for (int k = 0; k < 4; k++) if (((uint64_t) o[2 * k + 1] << 32 | o[2 * k]) != x[k] % P) mbad++;
      stepi(); getdout(o); for (int k = 0; k < 4; k++) if (((uint64_t) o[2 * k + 1] << 32 | o[2 * k]) != (x[k] % P + y[k]) % P) mbad++;
    }
    printf("mod-ALU REDP / ADDP, 2000 x 4 lanes each: %d mismatches\n", mbad); bad += mbad;
  }
#endif
#ifdef HAS_BF16
  top.p_rst.set<bool>(true); tick(); top.p_rst.set<bool>(false);
  int p = 0;
  crf(p++, ins(1, 0, 0, 0, 0)); crf(p++, ins(1, 1, 0, 0, 0));
  crf(p++, ins(3, 0, 1, 0, 0)); crf(p++, ins(4, 0, 2, 0, 0)); crf(p++, ins(7, 0, 0, 0, 0));
  crf(p++, ins(5, 0, 0, 0, 0)); crf(p++, ins(5, 0, 0, 0, 0)); crf(p++, ins(5, 0, 0, 0, 0));
  crf(p++, ins(0, 0, 0, 0, 0)); crf(p++, ins(0, 0, 0, 0, 0)); crf(p++, ins(0, 0, 0, 0, 0));
  crf(p++, ins(6, 0, 3, 0, 0)); crf(p++, ins(2, 0, 0, 1, 0)); crf(p++, ins(2, 0, 0, 2, 0)); crf(p++, ins(2, 0, 0, 3, 0));
  crf(p++, ins(9, 0, 0, 0, 0));
  int fbad = 0, nt = 0; double worst = 0;
  std::uniform_real_distribution<float> U(-8.f, 8.f); std::uniform_int_distribution<int> E(-6, 6);
  for (int t = 0; t < 2000; t++) {
    top.p_pc__rst.set<bool>(true); tick(); top.p_pc__rst.set<bool>(false);
    uint16_t a[16], b[16]; uint32_t w[8], o[8];
    for (int i = 0; i < 16; i++) { a[i] = f2bf(std::ldexp(U(rng), E(rng))); b[i] = f2bf(std::ldexp(U(rng), E(rng))); }
    if (t % 7 == 0) { a[3] = 0; b[5] = 0; }
    for (int i = 0; i < 8; i++) w[i] = a[2 * i] | (uint32_t) a[2 * i + 1] << 16; setdin(w); stepi();
    for (int i = 0; i < 8; i++) w[i] = b[2 * i] | (uint32_t) b[2 * i + 1] << 16; setdin(w); stepi();
    for (int i = 0; i < 10; i++) stepi();
    double dot = 0, mag = 0; for (int i = 0; i < 16; i++) { dot += (double) bf2f(a[i]) * bf2f(b[i]); mag += std::fabs((double) bf2f(a[i]) * bf2f(b[i])); }
    stepi(); getdout(o);
    for (int i = 0; i < 16; i++) { double r = bf2f(o[i / 2] >> (16 * (i & 1))), e = (double) bf2f(a[i]) * bf2f(b[i]);
      double er = std::fabs(r - e) / (std::fabs(e) + 1e-30); nt++; if (e != 0 && er > 1.0 / 128) fbad++; if (e != 0 && er > worst) worst = er; }
    stepi(); getdout(o);
    for (int i = 0; i < 16; i++) { double r = bf2f(o[i / 2] >> (16 * (i & 1))), e = (double) bf2f(a[i]) + bf2f(b[i]);
      double big = std::fmax(std::fabs(bf2f(a[i])), std::fabs(bf2f(b[i]))); nt++; if (std::fabs(r - e) > big / 64) fbad++; }
    stepi(); getdout(o);
    { double r = bf2f(o[0] & 0xffff), e = 3 * dot; nt++; if (std::fabs(r - e) > 3 * mag / 100) { fbad++; if (fbad < 5) printf("  mac %g vs %g (mag %g)\n", r, e, 3 * mag); } }
  }
  printf("BF16 element-wise multiply / add / 3-term MAC: %d checks, %d outside tolerance (worst multiply rel. error %.4f)\n", nt, fbad, worst);
  bad += fbad;
#endif
  return bad != 0;
}
