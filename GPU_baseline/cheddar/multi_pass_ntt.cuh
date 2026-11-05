// Multi-pass streaming NTT for PCG^2 (no Cheddar code).  The N-point negacyclic NTT as k passes of R_p = 2^M_p-point
// transforms (M_p = 3 or 4: 8 or 16 elements per thread), every pass a register kernel on contiguous groups: one thread
// = one group, M_p radix-2 stages in registers (constant-memory twiddles), the inter-pass twist, store.  Between passes the
// data is re-laid out so that the next pass's groups are contiguous (a transpose of the [R_{p+1}] x [N / R_{p+1}] view),
// which the DRU performs in memory; in the bench it is a GPU transpose, outside the timed SM lane (mp) or inside it (mpf,
// the pipeline without a DRU).  The SPUs write g directly in pass 0's layout (the R_0 digit least significant).  Compared
// with the two-pass kernel (two_pass_ntt.cuh) the SM lane moves k/2 times the bytes and relies on k - 1 re-layouts per
// transform instead of one: the design exists for a GPU whose passes are compute-bound (B200), where the extra SM bytes
// are hidden and the re-layouts are the whole cost of running it without the DRU.
//
// Mixed radix: n = d_0 S_0 + d_1 S_1 + ... + d_{k-1}, S_p = N / (R_0 ... R_p).  Pass p does R_p-point DFTs over digit d_p
// for every value of the other digits; the output digit replaces d_p (bit-reversed within the group).  The negacyclic
// twist psi^n is folded into pass 0 (negacyclic R_0-point transform with psi^S_0, plus psi^m on the output, m = the lower
// digits); the four-step twist of pass p is omega_Nsub^(k m), Nsub = N / (R_0 ... R_{p-1}).  Every twist is a power of
// psi, psi^(m c_k) with c_k = 2k + 1 (pass 0) or 2 k stepN.  When m is uniform over the block (2^shift >= the block size)
// the R pairs come from one N-entry Shoup table (psi^(e-N) = -psi^e: the pair of -w is (P - w, ~q)); otherwise m = 32 mh +
// ml and psi^(m c_k) = T1[ml][k] T2[mh][k] with T1 (32 x R, per pass) staged in shared memory and T2 warp-uniform (two
// Shoup multiplies per element).  No running product.  Lazy reduction as in the two-pass kernel: values in
// [0, 2P) through the passes, one conditional subtraction per butterfly output, outputs reduced to [0, P) where consumed.
// Coalescing: a block of MP_BS threads stages its MP_BS groups through shared memory (row = group, padded), so the global
// loads and stores are contiguous across the block.
#pragma once
namespace mp {
using u64 = ::u64;
typedef long long i64;
typedef unsigned __int128 u128;
__constant__ u64 cP, cQI;   // modulus (< 2^62); P^-1 mod 2^64 (Montgomery, the b^ product)
struct __align__(16) Sh { u64 w, q; };   // Shoup pair: w, floor(w 2^64 / P)
__constant__ Sh cTw[8][16];    // per pass p: pass 0 = psi_{R0}^brv (negacyclic CT), others = omega_{Rp}^j (cyclic GS-DIF), j < Rp/2
__constant__ Sh cTwI[8][16];   // inverses: pass 0 = psi_{R0}^-brv (negacyclic GS), others = omega_{Rp}^-j (cyclic CT-DIT)
__device__ __forceinline__ u64 mulsl(u64 a, Sh w) { const u64 q = __umul64hi(a, w.q); return a * w.w - q * cP; }   // a w mod P in [0, 2P)
__device__ __forceinline__ u64 red2(u64 x) { const u64 d = x - 2 * cP; return ((int) (d >> 32)) < 0 ? x : d; }   // [0, 4P) -> [0, 2P)
__device__ __forceinline__ u64 red1(u64 x) { const u64 d = x - cP; return ((int) (d >> 32)) < 0 ? x : d; }       // [0, 2P) -> [0, P)
__device__ __forceinline__ u64 mont(u64 x, u64 bR) {   // x in [0, 2P), bR = b 2^64 mod P -> x b mod P in [0, 2P)
  const u64 hi = __umul64hi(x, bR), lo = x * bR, t = lo * cQI, r = hi - __umul64hi(t, cP); return (i64) r < 0 ? r + cP : r; }
__device__ __forceinline__ Sh neg(Sh s) { return Sh{cP - s.w, ~s.q}; }   // pair of -w (w != 0: w 2^64 / P is not an integer)
__host__ __device__ constexpr int brev(int k, int M) { int r = 0; for (int i = 0; i < M; i++) r |= ((k >> i) & 1) << (M - 1 - i); return r; }

// negacyclic R-point, Cooley-Tukey with merged psi (natural in, bit-reversed out), tw[m + i] = psi_R^brv
template <int M> __device__ __forceinline__ void nega_fwd(u64 *v, const Sh *tw) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int s = 0; s < M; s++) {
#pragma unroll
    for (int b = 0; b < R / 2; b++) { const int t = R >> (s + 1), i = b >> (M - 1 - s), j = b & (t - 1), p = i * 2 * t + j;
      const u64 u = v[p], w = mulsl(v[p + t], tw[(1 << s) + i]); v[p] = red2(u + w); v[p + t] = red2(u + 2 * cP - w); } }
}
// negacyclic inverse, Gentleman-Sande (bit-reversed in, natural out), itw[h + i] = psi_R^-brv; no 1/R here
template <int M> __device__ __forceinline__ void nega_inv(u64 *v, const Sh *itw) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int s = 0; s < M; s++) {
#pragma unroll
    for (int b = 0; b < R / 2; b++) { const int t = 1 << s, h = R >> (s + 1), i = b >> s, j = b & (t - 1), p = i * 2 * t + j;
      const u64 u = v[p], w = v[p + t]; v[p] = red2(u + w); v[p + t] = mulsl(u + 2 * cP - w, itw[h + i]); } }
}
// cyclic R-point, Gentleman-Sande DIF (natural in, bit-reversed out), tw[j] = omega_R^j
template <int M> __device__ __forceinline__ void cyc_fwd(u64 *v, const Sh *tw) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int s = 0; s < M; s++) {
#pragma unroll
    for (int b = 0; b < R / 2; b++) { const int t = R >> (s + 1), g = b >> (M - 1 - s), j = b & (t - 1), p = g * 2 * t + j;
      const u64 u = v[p], w = v[p + t]; v[p] = red2(u + w); v[p + t] = mulsl(u + 2 * cP - w, tw[j << s]); } }
}
// cyclic inverse, Cooley-Tukey DIT (bit-reversed in, natural out), itw[j] = omega_R^-j; no 1/R here
template <int M> __device__ __forceinline__ void cyc_inv(u64 *v, const Sh *itw) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int s = 0; s < M; s++) {
#pragma unroll
    for (int b = 0; b < R / 2; b++) { const int t = 1 << s, g = b >> s, j = b & (t - 1), p = g * 2 * t + j;
      const u64 u = v[p], w = mulsl(v[p + t], itw[j << (M - 1 - s)]); v[p] = red2(u + w); v[p + t] = red2(u + 2 * cP - w); } }
}
// the twist of output k of group g in pass p (m = g >> shift): psi^(m (2k + 1)) in pass 0, omega_Nsub^(k m) = psi^(2 k m stepN)
// after it (stepN = R_0 ... R_{p-1}); e < 2N, psi^N = -1
__device__ __forceinline__ Sh twist(const Sh *__restrict__ tab, size_t e, size_t N) { return e < N ? tab[e] : neg(tab[e - N]); }

#ifndef MP_BS
#define MP_BS 256   // threads (= groups) per block
#endif
#ifndef MP_THREADS
#define MP_THREADS 512   // resident threads per SM the register budget is set for (512: 128 registers)
#endif
#define MP_LB __launch_bounds__(MP_BS, MP_THREADS / MP_BS)
// staging: the block's MP_BS groups x R elements, row = group, one pad word per row (conflict-free row reads)
template <int M> __device__ __forceinline__ void stage_in(u64 *sm, const u64 *__restrict__ x) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int j = 0; j < R; j++) { const int idx = j * MP_BS + threadIdx.x; sm[(idx >> M) * (R + 1) + (idx & (R - 1))] = __ldcs(x + idx); }
}
template <int M> __device__ __forceinline__ void stage_out(u64 *__restrict__ x, const u64 *sm) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int j = 0; j < R; j++) { const int idx = j * MP_BS + threadIdx.x; __stcs(x + idx, sm[(idx >> M) * (R + 1) + (idx & (R - 1))]); }
}
template <int M> __device__ __forceinline__ void row_ld(u64 *v, const u64 *sm) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int i = 0; i < R; i++) v[i] = sm[threadIdx.x * (R + 1) + i]; }
template <int M> __device__ __forceinline__ void row_st(u64 *sm, const u64 *v) {
  constexpr int R = 1 << M;
#pragma unroll
  for (int i = 0; i < R; i++) sm[threadIdx.x * (R + 1) + i] = v[i]; }

// forward pass p: blockIdx.x = chunk of MP_BS groups, blockIdx.y = item; in place.  TWM: the output twist mode (0 = none:
// the last pass; 1 = block-uniform m, pairs from psiSh; 2 = two tables)
template <int M, int TWM> __device__ __forceinline__ void twist_ld(Sh *w, u64 *t1, const Sh *__restrict__ psiSh, const Sh *__restrict__ T1,
                                                                    size_t g, int shift, size_t stepN, size_t N, bool first) {
  constexpr int R = 1 << M; const size_t m = g >> shift;
  if (TWM == 1) {
#pragma unroll
    for (int k = 0; k < R; k++) w[k] = twist(psiSh, first ? m * (2 * k + 1) : 2 * k * m * stepN, N);
  } else if (TWM == 2) {   // T1 (32 x (R + 1) pairs, padded rows) to shared memory
    for (int i = threadIdx.x; i < 32 * (R + 1) * 2; i += MP_BS) t1[i] = ((const u64 *) T1)[i];
  }
}
template <int M, int TWM> __device__ __forceinline__ void twist_ap(u64 *v, const Sh *w, const u64 *t1, const Sh *__restrict__ T2, size_t g, int shift) {
  constexpr int R = 1 << M;
  if (TWM == 1) {
#pragma unroll
    for (int k = 0; k < R; k++) v[brev(k, M)] = mulsl(v[brev(k, M)], w[k]);
  } else if (TWM == 2) {
    const size_t m = g >> shift; const int ml = (int) (m & 31); const Sh *t2 = T2 + (m >> 5) * R; const Sh *s1 = (const Sh *) t1 + ml * (R + 1);
#pragma unroll
    for (int k = 0; k < R; k++) v[brev(k, M)] = mulsl(mulsl(v[brev(k, M)], s1[k]), t2[k]);
  } else {   // the last pass (a^ precomputation): reduced to [0, P) for the pointwise b^ product
#pragma unroll
    for (int i = 0; i < R; i++) v[i] = red1(v[i]);
  }
}
template <int M, bool FIRST, int TWM> __global__ void MP_LB pass_fwd(u64 *__restrict__ x, int p, int shift, size_t stepN, const Sh *__restrict__ psiSh,
                                                                     const Sh *__restrict__ T1, const Sh *__restrict__ T2, size_t N) {
  constexpr int R = 1 << M; __shared__ u64 sm[MP_BS * (R + 1)]; __shared__ u64 t1[TWM == 2 ? 32 * (R + 1) * 2 : 2];
  u64 *chunk = x + (size_t) blockIdx.y * N + (size_t) blockIdx.x * (MP_BS * R);
  const size_t g = (size_t) blockIdx.x * MP_BS + threadIdx.x;
  Sh w[TWM == 1 ? R : 1]; twist_ld<M, TWM>(w, t1, psiSh, T1, g, shift, stepN, N, FIRST);
  stage_in<M>(sm, chunk); __syncthreads();
  u64 v[R]; row_ld<M>(v, sm);
  if (FIRST) nega_fwd<M>(v, cTw[0]); else cyc_fwd<M>(v, cTw[p]);
  twist_ap<M, TWM>(v, w, t1, T2, g, shift);
  row_st<M>(sm, v); __syncthreads(); stage_out<M>(chunk, sm);
}
// last forward pass with the b^ product (Montgomery form, same layout) and the accumulate over the items in registers:
// acc = sum over items of NTT(x_item) . b^_item; blockIdx.x = chunk
template <int M, bool FIRST> __global__ void MP_LB pass_acc(const u64 *__restrict__ x, int p, size_t N,
                                                                                         const u64 *__restrict__ bR, u64 *__restrict__ acc, int items) {
  constexpr int R = 1 << M; __shared__ u64 sm[MP_BS * (R + 1)];
  const size_t off = (size_t) blockIdx.x * (MP_BS * R);
  u64 a[R], v[R];
#pragma unroll
  for (int i = 0; i < R; i++) a[i] = 0;
  for (int it = 0; it < items; it++) {
    stage_in<M>(sm, x + (size_t) it * N + off); __syncthreads();
    row_ld<M>(v, sm); __syncthreads();   // every row read before the b^ staging overwrites it
    if (FIRST) nega_fwd<M>(v, cTw[0]); else cyc_fwd<M>(v, cTw[p]);
    stage_in<M>(sm, bR + (size_t) it * N + off); __syncthreads();
#pragma unroll
    for (int i = 0; i < R; i++) a[i] = red2(a[i] + mont(v[i], sm[threadIdx.x * (R + 1) + i]));
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < R; i++) a[i] = red1(a[i]);
  row_st<M>(sm, a); __syncthreads(); stage_out<M>(acc + off, sm);
}
// inverse pass p (the mirror, one item): undo the twist (inverse tables, TWM as in the forward; 0 for the last forward
// pass), inverse DFT; pass 0 also applies 1/N and reduces to [0, P)
template <int M, bool FIRST, int TWM> __global__ void MP_LB pass_inv(u64 *__restrict__ x, int p, int shift, size_t stepN, const Sh *__restrict__ psiiSh,
                                                                     const Sh *__restrict__ T1, const Sh *__restrict__ T2, size_t N, Sh ninv) {
  constexpr int R = 1 << M; __shared__ u64 sm[MP_BS * (R + 1)]; __shared__ u64 t1[TWM == 2 ? 32 * (R + 1) * 2 : 2];
  u64 *chunk = x + (size_t) blockIdx.x * (MP_BS * R);
  const size_t g = (size_t) blockIdx.x * MP_BS + threadIdx.x;
  Sh w[TWM == 1 ? R : 1]; twist_ld<M, TWM>(w, t1, psiiSh, T1, g, shift, stepN, N, FIRST);
  stage_in<M>(sm, chunk); __syncthreads();
  u64 v[R]; row_ld<M>(v, sm);
  if (TWM) twist_ap<M, TWM>(v, w, t1, T2, g, shift);
  if (FIRST) { nega_inv<M>(v, cTwI[0]);
#pragma unroll
    for (int i = 0; i < R; i++) v[i] = red1(mulsl(v[i], ninv)); }
  else cyc_inv<M>(v, cTwI[p]);
  row_st<M>(sm, v); __syncthreads(); stage_out<M>(chunk, sm);
}
// re-layout (the DRU's job; the GPU version for the bench): per item (blockIdx.z) the [R] x [C] view transposed to [C] x [R]
// (FWD: dst[c R + r] = src[r C + c]) or back (dst[r C + c] = src[c R + r]); tiles of R x (1024 / R), both sides coalesced
template <int R, bool FWD> __global__ void __launch_bounds__(256) xpose(const u64 *__restrict__ src, u64 *__restrict__ dst, int C) {
  constexpr int TC = 1024 / R; __shared__ u64 tile[R][TC + 1];
  const size_t off = (size_t) blockIdx.z * R * C; src += off; dst += off; const int c0 = blockIdx.x * TC;
  if (FWD) {
#pragma unroll
    for (int q = threadIdx.x; q < 1024; q += 256) { const int r = q / TC, c = q % TC; tile[r][c] = src[(size_t) r * C + c0 + c]; }
    __syncthreads();
#pragma unroll
    for (int q = threadIdx.x; q < 1024; q += 256) { const int c = q / R, r = q % R; dst[(size_t) (c0 + c) * R + r] = tile[r][c]; }
  } else {
#pragma unroll
    for (int q = threadIdx.x; q < 1024; q += 256) { const int c = q / R, r = q % R; tile[r][c] = src[(size_t) (c0 + c) * R + r]; }
    __syncthreads();
#pragma unroll
    for (int q = threadIdx.x; q < 1024; q += 256) { const int r = q / TC, c = q % TC; dst[(size_t) r * C + c0 + c] = tile[r][c]; }
  }
}
// b^ -> Montgomery form (b^ 2^64 mod P): x b^ 2^64 = mont(x, (2^64)^2 mod P)
__global__ void to_mont(u64 *__restrict__ x, size_t n, u64 r2) {
  for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) x[i] = red1(mont(x[i], r2)); }

// host: the pass plan and tables for one L
struct Plan {
  int L, k; int Mp[8]; u64 psi, psii; Sh ninv; u64 r2; Sh *psiSh = nullptr, *psiiSh = nullptr;
  Sh *T1[8] = {}, *T2[8] = {}, *T1i[8] = {}, *T2i[8] = {};   // two-table twists of the passes whose m is not block-uniform
  bool uni(int p) const { return (1ull << shift(p)) >= MP_BS; }   // m = g >> shift uniform over a block of MP_BS groups
  static u64 mulm_h(u64 a, u64 b, u64 P) { return (u64) ((u128) a * b % P); }
  static u64 powm_h(u64 a, u64 e, u64 P) { u64 r = 1; while (e) { if (e & 1) r = mulm_h(r, a, P); a = mulm_h(a, a, P); e >>= 1; } return r; }
  static Sh sh(u64 w, u64 P) { return Sh{w, (u64) (((u128) w << 64) / P)}; }
  static void plan(int L, int k, int *Mp) { for (int p = 0; p < k; p++) Mp[p] = L / k + (p < L % k ? 1 : 0); }   // k balanced passes
  Plan(int L_, int k_, u64 PP) : L(L_), k(k_) {
    plan(L, k, Mp); const size_t N = 1ull << L;
    u64 g = 3; while (powm_h(g, (PP - 1) / 2, PP) == 1) g++;
    psi = powm_h(g, (PP - 1) / (2 * N), PP); psii = powm_h(psi, PP - 2, PP); ninv = sh(powm_h(N % PP, PP - 2, PP), PP);
    const u64 r1 = (u64) (((u128) 1 << 64) % PP); r2 = mulm_h(r1, r1, PP);
    u64 qi = PP; for (int i = 0; i < 6; i++) qi *= 2 - PP * qi;   // P^-1 mod 2^64 (Newton, P odd)
    cudaMemcpyToSymbol(cP, &PP, 8); cudaMemcpyToSymbol(cQI, &qi, 8);
    const u64 om = mulm_h(psi, psi, PP), omi = mulm_h(psii, psii, PP);
    Sh tw[8][16], twi[8][16]; memset(tw, 0, sizeof tw); memset(twi, 0, sizeof twi);
    size_t S = N;
    for (int p = 0; p < k; p++) {
      const int R = 1 << Mp[p]; S /= R;   // S = S_p
      if (p == 0) {   // negacyclic R-point: psi_R = psi^S_0 (primitive 2R-th root), bit-reversed powers
        const u64 pr = powm_h(psi, S, PP), pri = powm_h(psii, S, PP);
        for (int j = 0; j < R; j++) { const int r = brev(j, Mp[0]); tw[0][j] = sh(powm_h(pr, r, PP), PP); twi[0][j] = sh(powm_h(pri, r, PP), PP); }
      } else {        // cyclic R-point: omega_R = omega^(N/R)
        const u64 wr = powm_h(om, N / R, PP), wri = powm_h(omi, N / R, PP); u64 w = 1, wi = 1;
        for (int j = 0; j < R / 2; j++) { tw[p][j] = sh(w, PP); twi[p][j] = sh(wi, PP); w = mulm_h(w, wr, PP); wi = mulm_h(wi, wri, PP); }
      }
    }
    cudaMemcpyToSymbol(cTw, tw, sizeof tw); cudaMemcpyToSymbol(cTwI, twi, sizeof twi);
    std::vector<Sh> a(N), ai(N); u64 w = 1, wi = 1;
    for (size_t j = 0; j < N; j++) { a[j] = sh(w, PP); ai[j] = sh(wi, PP); w = mulm_h(w, psi, PP); wi = mulm_h(wi, psii, PP); }
    cudaMalloc(&psiSh, N * sizeof(Sh)); cudaMalloc(&psiiSh, N * sizeof(Sh));
    cudaMemcpy(psiSh, a.data(), N * sizeof(Sh), cudaMemcpyHostToDevice); cudaMemcpy(psiiSh, ai.data(), N * sizeof(Sh), cudaMemcpyHostToDevice);
    // two-table twists: psi^(m c_k) = T1[m mod 32][k] T2[m / 32][k], c_k = 2k + 1 (pass 0) or 2 k stepN; T1 rows padded to R + 1
    for (int p = 0; p + 1 < k; p++) {
      if (uni(p)) continue;
      const int R = 1 << Mp[p]; const size_t sN = stepN(p), maxm = N / ((size_t) R << shift(p)), nh = (maxm + 31) / 32;
      std::vector<Sh> t1(32 * (R + 1)), t1i(32 * (R + 1)), t2(nh * R), t2i(nh * R);
      auto ck = [&](int kk) { return p == 0 ? (u64) (2 * kk + 1) : (u64) 2 * kk * sN; };
      for (int ml = 0; ml < 32; ml++) for (int kk = 0; kk < R; kk++) { const u64 e = ((u64) ml * ck(kk)) % (2 * N);
        t1[ml * (R + 1) + kk] = sh(powm_h(psi, e, PP), PP); t1i[ml * (R + 1) + kk] = sh(powm_h(psii, e, PP), PP); }
      for (size_t mh = 0; mh < nh; mh++) for (int kk = 0; kk < R; kk++) { const u64 e = (u64) ((((u128) 32 * mh) * ck(kk)) % (2 * N));
        t2[mh * R + kk] = sh(powm_h(psi, e, PP), PP); t2i[mh * R + kk] = sh(powm_h(psii, e, PP), PP); }
      auto up = [&](Sh **d, const std::vector<Sh> &h) { cudaMalloc(d, h.size() * sizeof(Sh)); cudaMemcpy(*d, h.data(), h.size() * sizeof(Sh), cudaMemcpyHostToDevice); };
      up(&T1[p], t1); up(&T2[p], t2); up(&T1i[p], t1i); up(&T2i[p], t2i);
    }
    cudaDeviceSynchronize();
  }
  ~Plan() { cudaFree(psiSh); cudaFree(psiiSh); for (int p = 0; p < 8; p++) { cudaFree(T1[p]); cudaFree(T2[p]); cudaFree(T1i[p]); cudaFree(T2i[p]); } }
  int shift(int p) const { int s = 0; for (int q = 0; q < p; q++) s += Mp[q]; return s; }
  size_t stepN(int p) const { return 1ull << shift(p); }
};
}   // namespace mp
