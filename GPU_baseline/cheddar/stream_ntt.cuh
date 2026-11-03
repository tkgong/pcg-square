// Streaming multi-pass NTT for PCG^2 (no Cheddar code): the N-point negacyclic NTT as P passes of R_p = 2^M_p-point
// transforms, every pass a pure register kernel on contiguous groups (one thread = one group of 2^M_p elements: load,
// M_p radix-2 stages in registers, the inter-pass twiddle, store; no shared memory, no __syncthreads).  Between passes
// the data is re-laid out so that the next pass's groups are contiguous: a transpose of the [R_{p+1}] x [N / R_{p+1}]
// view, which the DRU performs in memory (in the bench it is a GPU transpose outside the timed SM lane).  The SPUs
// write g directly in pass 0's layout (the R_0 digit least significant).
//
// Mixed radix: n = d_0 S_0 + d_1 S_1 + ... + d_{P-1}, S_p = N / (R_0 ... R_p).  Pass p does R_p-point DFTs over digit d_p
// for every value of the other digits; the output digit k_p replaces d_p (stored bit-reversed within the group).  The
// negacyclic twist psi^n is folded into pass 0 (negacyclic R_0-point transform, psi_R0 = psi^S_0, plus psi^m on the
// output); the four-step twiddle of pass p is omega_{Nsub}^{k_p m} with Nsub = N / (R_0 ... R_{p-1}) and m the value of
// the lower digits d_{p+1}..d_{P-1}, generated per group as a running product (two table lookups per group).
#pragma once
namespace sntt {
using u64 = ::u64;
typedef unsigned __int128 u128;
__constant__ u64 cP, cMU;
struct Sh { u64 w, q; };
__constant__ Sh cTw[6][64];     // per pass p: pass 0 = psi_{R0}^brv (negacyclic CT), others = omega_{Rp}^j (cyclic GS-DIF), j < Rp/2
__constant__ Sh cTwI[6][64];    // inverses: pass 0 = psi_{R0}^-brv (negacyclic GS), others = omega_{Rp}^-j (cyclic CT-DIT)
__device__ __forceinline__ u64 addm(u64 a, u64 b) { u64 s = a + b; return s >= cP ? s - cP : s; }
__device__ __forceinline__ u64 subm(u64 a, u64 b) { return a >= b ? a - b : a + cP - b; }
__device__ __forceinline__ u64 mulm(u64 a, u64 b) {   // Barrett, mu = floor(2^124 / P), P < 2^62
  const u64 hi = __umul64hi(a, b), lo = a * b, x1 = (hi << 4) | (lo >> 60), q = __umul64hi(x1, cMU);
  u64 r = lo - q * cP; while (r >= cP) r -= cP; return r; }
__device__ __forceinline__ u64 muls(u64 a, Sh w) { const u64 q = __umul64hi(a, w.q), r = a * w.w - q * cP; return r >= cP ? r - cP : r; }

template <int M> __device__ __forceinline__ void load_group(u64 *v, const u64 *__restrict__ x) {
  const uint4 *s = (const uint4 *) x;
#pragma unroll
  for (int i = 0; i < (1 << M) / 2; i++) { uint4 t = s[i]; v[2 * i] = (u64) t.x | ((u64) t.y << 32); v[2 * i + 1] = (u64) t.z | ((u64) t.w << 32); }
}
template <int M> __device__ __forceinline__ void store_group(u64 *__restrict__ x, const u64 *v) {
  uint4 *d = (uint4 *) x;
#pragma unroll
  for (int i = 0; i < (1 << M) / 2; i++) d[i] = make_uint4((unsigned) v[2 * i], (unsigned) (v[2 * i] >> 32), (unsigned) v[2 * i + 1], (unsigned) (v[2 * i + 1] >> 32));
}
// negacyclic R-point, Cooley-Tukey with merged psi (natural in, bit-reversed out), table tw[m + i] = psi_R^brv
template <int M> __device__ __forceinline__ void nega_fwd(u64 *v, const Sh *tw) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int s = 0; s < M; s++) {
#pragma unroll
    for (int b = 0; b < R / 2; b++) { const int t = R >> (s + 1), i = b >> (M - 1 - s), j = b & (t - 1), p = i * 2 * t + j;
      const u64 u = v[p], w = muls(v[p + t], tw[(1 << s) + i]); v[p] = addm(u, w); v[p + t] = subm(u, w); } }
}
// negacyclic inverse, Gentleman-Sande (bit-reversed in, natural out), table itw[h + i] = psi_R^-brv; no 1/R here
template <int M> __device__ __forceinline__ void nega_inv(u64 *v, const Sh *itw) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int s = 0; s < M; s++) {
#pragma unroll
    for (int b = 0; b < R / 2; b++) { const int t = 1 << s, h = R >> (s + 1), i = b >> s, j = b & (t - 1), p = i * 2 * t + j;
      const u64 u = v[p], w = v[p + t]; v[p] = addm(u, w); v[p + t] = muls(subm(u, w), itw[h + i]); } }
}
// cyclic R-point, Gentleman-Sande DIF (natural in, bit-reversed out), table tw[j] = omega_R^j
template <int M> __device__ __forceinline__ void cyc_fwd(u64 *v, const Sh *tw) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int s = 0; s < M; s++) {
#pragma unroll
    for (int b = 0; b < R / 2; b++) { const int t = R >> (s + 1), g = b >> (M - 1 - s), j = b & (t - 1), p = g * 2 * t + j;
      const u64 u = v[p], w = v[p + t]; v[p] = addm(u, w); v[p + t] = muls(subm(u, w), tw[j << s]); } }
}
// cyclic inverse, Cooley-Tukey DIT (bit-reversed in, natural out), table itw[j] = omega_R^-j; no 1/R here
template <int M> __device__ __forceinline__ void cyc_inv(u64 *v, const Sh *itw) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int s = 0; s < M; s++) {
#pragma unroll
    for (int b = 0; b < R / 2; b++) { const int t = 1 << s, g = b >> s, j = b & (t - 1), p = g * 2 * t + j;
      const u64 u = v[p], w = muls(v[p + t], itw[j << (M - 1 - s)]); v[p] = addm(u, w); v[p + t] = subm(u, w); } }
}
__host__ __device__ constexpr int brev(int k, int M) { int r = 0; for (int i = 0; i < M; i++) r |= ((k >> i) & 1) << (M - 1 - i); return r; }   // constexpr: folded in the unrolled loops, keeps v[] in registers

// forward pass p.  x: item-major (N per item), groups contiguous.  shift = M_0 + ... + M_{p-1} (m = g >> shift), stepN =
// N / Nsub = R_0 ... R_{p-1} (omega_{Nsub}^m = omegaN[m * stepN]); pass 0: shift 0, stepN 1, start psi[m].
// LAST: no output twiddle; with acc: multiply by bhat (same layout) and accumulate over the items in registers.
template <int M, bool FIRST, bool LAST> __global__ void __launch_bounds__(256) pass_fwd(u64 *__restrict__ x, int p, int shift, size_t stepN,
                                                                                         const u64 *__restrict__ omegaN, const u64 *__restrict__ psiN, size_t N,
                                                                                         const u64 *__restrict__ bhat, u64 *__restrict__ acc, int items) {
  constexpr int R = 1 << M;
  const size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x; if (g * R >= N) return;
  u64 v[R], a[R];
  if (LAST && acc) {
#pragma unroll
    for (int i = 0; i < R; i++) a[i] = 0;
  }
  const int nit = (LAST && acc) ? items : 1;
  for (int it = 0; it < nit; it++) {
    load_group<M>(v, x + (size_t) it * N + g * R);
    if (FIRST) nega_fwd<M>(v, cTw[0]); else cyc_fwd<M>(v, cTw[p]);
    if (!LAST) {   // output twiddle: out[brv(k)] *= start * base^k, base = omega_Nsub^m, start = psi^m (pass 0) or 1
      const size_t m = g >> shift; const u64 base = omegaN[m * stepN]; u64 w = FIRST ? psiN[m] : 1;
#pragma unroll
      for (int k = 0; k < R; k++) { const int pos = brev(k, M); v[pos] = mulm(v[pos], w); w = mulm(w, base); }
      store_group<M>(x + g * R, v);
    } else if (acc) {
      const u64 *b = bhat + (size_t) it * N + g * R;
#pragma unroll
      for (int i = 0; i < R; i++) a[i] = addm(a[i], mulm(v[i], b[i]));
    } else store_group<M>(x + g * R, v);
  }
  if (LAST && acc) store_group<M>(acc + g * R, a);
}
// inverse pass p (the mirror): undo the output twiddle (inverse base), inverse DFT; pass 0 (FIRST) also applies 1/N
template <int M, bool FIRST, bool LAST> __global__ void __launch_bounds__(256) pass_inv(u64 *__restrict__ x, int p, int shift, size_t stepN,
                                                                                         const u64 *__restrict__ omegaNi, const u64 *__restrict__ psiNi, size_t N, u64 ninv) {
  constexpr int R = 1 << M;
  const size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x; if (g * R >= N) return;
  u64 v[R];
  load_group<M>(v, x + g * R);
  if (!LAST) {
    const size_t m = g >> shift; const u64 base = omegaNi[m * stepN]; u64 w = FIRST ? psiNi[m] : 1;
#pragma unroll
    for (int k = 0; k < R; k++) { const int pos = brev(k, M); v[pos] = mulm(v[pos], w); w = mulm(w, base); }
  }
  if (FIRST) { nega_inv<M>(v, cTwI[0]);
#pragma unroll
    for (int i = 0; i < R; i++) v[i] = mulm(v[i], ninv); }
  else cyc_inv<M>(v, cTwI[p]);
  store_group<M>(x + g * R, v);
}

// host: the pass plan and tables for one L
struct Plan {
  int L, P; int Mp[6]; u64 psi, psii, ninv; u64 *omegaN = nullptr, *omegaNi = nullptr, *psiN = nullptr, *psiNi = nullptr;
  static u64 mulm_h(u64 a, u64 b, u64 P) { return (u64) ((u128) a * b % P); }
  static u64 powm_h(u64 a, u64 e, u64 P) { u64 r = 1; while (e) { if (e & 1) r = mulm_h(r, a, P); a = mulm_h(a, a, P); e >>= 1; } return r; }
  static Sh sh(u64 w, u64 P) { return Sh{w, (u64) (((u128) w << 64) / P)}; }
  static void plan(int L, int Mmax, int *Mp, int &P) {   // balanced passes of at most Mmax stages
    P = (L + Mmax - 1) / Mmax; for (int p = 0; p < P; p++) Mp[p] = L / P + (p < L % P ? 1 : 0); }
  Plan(int L_, int Mmax, u64 P_) : L(L_) {
    plan(L, Mmax, Mp, P); const size_t N = 1ull << L; const u64 PP = P_;
    u64 g = 3; while (powm_h(g, (PP - 1) / 2, PP) == 1) g++;
    psi = powm_h(g, (PP - 1) / (2 * N), PP); psii = powm_h(psi, PP - 2, PP); ninv = powm_h(N % PP, PP - 2, PP);
    const u64 mu = (u64) (((u128) 1 << 124) / PP);
    cudaMemcpyToSymbol(cP, &PP, 8); cudaMemcpyToSymbol(cMU, &mu, 8);
    const u64 om = mulm_h(psi, psi, PP), omi = mulm_h(psii, psii, PP);
    Sh tw[6][64], twi[6][64]; memset(tw, 0, sizeof tw); memset(twi, 0, sizeof twi);
    size_t S = N;
    for (int p = 0; p < P; p++) {
      const int R = 1 << Mp[p]; S /= R;   // S = S_p
      if (p == 0) {   // negacyclic R-point: psi_R = psi^S_0 (primitive 2R-th root), bit-reversed powers
        const u64 pr = powm_h(psi, S, PP), pri = powm_h(psii, S, PP);
        for (int k = 0; k < R; k++) { int r = 0; for (int q = 0; q < Mp[0]; q++) r |= ((k >> q) & 1) << (Mp[0] - 1 - q); tw[0][k] = sh(powm_h(pr, r, PP), PP); twi[0][k] = sh(powm_h(pri, r, PP), PP); }
      } else {        // cyclic R-point: omega_R = omega^(N/R)
        const u64 wr = powm_h(om, N / R, PP), wri = powm_h(omi, N / R, PP); u64 w = 1, wi = 1;
        for (int j = 0; j < R / 2; j++) { tw[p][j] = sh(w, PP); twi[p][j] = sh(wi, PP); w = mulm_h(w, wr, PP); wi = mulm_h(wi, wri, PP); }
      }
    }
    cudaMemcpyToSymbol(cTw, tw, sizeof tw); cudaMemcpyToSymbol(cTwI, twi, sizeof twi);
    std::vector<u64> a(N), ai(N), b(N), bi(N); a[0] = ai[0] = b[0] = bi[0] = 1;
    for (size_t j = 1; j < N; j++) { a[j] = mulm_h(a[j - 1], om, PP); ai[j] = mulm_h(ai[j - 1], omi, PP); b[j] = mulm_h(b[j - 1], psi, PP); bi[j] = mulm_h(bi[j - 1], psii, PP); }
    auto up = [&](u64 **d, const std::vector<u64> &h) { cudaMalloc(d, N * 8); cudaMemcpy(*d, h.data(), N * 8, cudaMemcpyHostToDevice); };
    up(&omegaN, a); up(&omegaNi, ai); up(&psiN, b); up(&psiNi, bi);
  }
  ~Plan() { cudaFree(omegaN); cudaFree(omegaNi); cudaFree(psiN); cudaFree(psiNi); }
  int shift(int p) const { int s = 0; for (int q = 0; q < p; q++) s += Mp[q]; return s; }
  size_t stepN(int p) const { return 1ull << shift(p); }
};
}   // namespace sntt
