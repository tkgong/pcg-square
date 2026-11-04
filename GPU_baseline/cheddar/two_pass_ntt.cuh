// Two-pass streaming NTT for PCG^2 (no Cheddar code).  N = N1 x N2.  The N-point negacyclic transform (Cooley-Tukey,
// natural order in, bit-reversed out, psi merged into one bit-reversed twiddle table) split by stage: the first log N1
// stages pair elements N2 apart, i.e. independent N1-point transforms of the columns, contiguous in the layout the SPUs
// write; the last log N2 stages pair elements inside one row.  The DRU transposes column-major -> row-major in memory
// between the two passes.  Row k1 (its position in the buffer) uses the slice tw[(N1 + k1) m + i] of the table, so there
// is no twist multiply and nothing to precompute per row; the slice (N2 entries) stays in L1 across the items.  Every
// block streams one contiguous 2^P-element line: 2^P/8 threads, 8 elements per thread, 3 radix-2 stages per register
// phase, shared-memory exchange between phases (padded), the next item's line prefetched into registers while the
// current one is transformed.  Twiddles: Shoup pairs.  The b^ product: Montgomery (b^ precomputed in Montgomery form).
// The row pass accumulates over the items in registers; one inverse (row, transpose, column) per output.
#pragma once
namespace tp {
using u64 = ::u64;
typedef long long i64;
typedef unsigned __int128 u128;
__constant__ u64 tP, tMU, tQI;   // modulus; floor(2^124 / P) (Barrett, table building only); P^-1 mod 2^64 (Montgomery)
struct Sh { u64 w, q; };
__device__ __forceinline__ u64 addm(u64 a, u64 b) { u64 s = a + b; return s >= tP ? s - tP : s; }
__device__ __forceinline__ u64 subm(u64 a, u64 b) { return a >= b ? a - b : a + tP - b; }
__device__ __forceinline__ u64 mulm(u64 a, u64 b) {
  const u64 hi = __umul64hi(a, b), lo = a * b, x1 = (hi << 4) | (lo >> 60), q = __umul64hi(x1, tMU);
  u64 r = lo - q * tP; while (r >= tP) r -= tP; return r; }
__device__ __forceinline__ u64 muls(u64 a, Sh w) { const u64 q = __umul64hi(a, w.q), r = a * w.w - q * tP; return r >= tP ? r - tP : r; }
// x in [0, P), bR = b 2^64 mod P -> x b mod P in [0, P): hi(x bR) - hi(t P) with t = lo(x bR) P^-1 mod 2^64 (no borrow:
// the low words are equal), the difference lies in (-P, P)
__device__ __forceinline__ u64 mont(u64 x, u64 bR) {
  const u64 hi = __umul64hi(x, bR), lo = x * bR, t = lo * tQI, r = hi - __umul64hi(t, tP); return (i64) r < 0 ? r + tP : r; }
__device__ __forceinline__ u64 powm_d(u64 a, u64 e) { u64 r = 1; while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; } return r; }
__device__ __forceinline__ int pad(int p) { return p + (p >> 3); }   // shared-memory index with one pad word per 8

// one register phase of up to 3 CT stages on the thread's 8 elements v[0..7] (element i at logical offset i * s from the
// thread's base), largest distance first: u + w v / u - w v with w = tw[rm * m + i], m = 2^P / (2 d) groups of the stage,
// i the butterfly's group, rm = 1 for a column line, N1 + k1 for row k1.  NS = stages in this phase (3, 2 or 1).
template <int NS> __device__ __forceinline__ void phase(u64 *v, const Sh *__restrict__ tw, int base, int s, int P, int rm) {
  const int d0 = (NS == 3 ? 4 : NS == 2 ? 2 : 1) * s, q = rm * ((1 << P) / (2 * d0)) + base / (2 * d0);   // first stage: distance d0
#pragma unroll
  for (int st = 0; st < NS; st++) {
    const int D = NS == 3 ? (4 >> st) : NS == 2 ? (2 >> st) : 1;   // slot distance
    const int d = D * s;                                             // logical distance
#pragma unroll
    for (int b = 0; b < (1 << (NS - 1)); b++) {
      const int lo = (b / D) * 2 * D + (b % D), hi = lo + D;
      // twiddle index rm * m + i with m = 2^P / (2 d), i = (base + lo s) / (2 d): one index q for the phase's first
      // stage, (q << st) + (b >> (NS - 1 - st)) for the later ones (the group index doubles per stage)
      const u64 u = v[lo], w = muls(v[hi], tw[(q << st) + (b >> (NS - 1 - st))]); v[lo] = addm(u, w); v[hi] = subm(u, w);
    }
  }
}
// forward line transform in place in shared memory sm[pad(.)], 2^P elements.  Phases: distances (2^(P-1), 2^(P-2), 2^(P-3)), ... down to 1.
template <int P> __device__ __forceinline__ void line_fwd(u64 *sm, const Sh *__restrict__ tw, int rm) {
  const int t = threadIdx.x; u64 v[8];
#pragma unroll
  for (int e = P - 1; e >= 0; e -= 3) {         // this phase covers distances 2^e, 2^(e-1), 2^(e-2) (as many as remain)
    const int ns = e >= 2 ? 3 : e + 1;         // stages in this phase
    const int s = 1 << (e - ns + 1);           // smallest distance of the phase = slot stride
    if (ns == 3) {   // 8 slots = 8 consecutive multiples of s: base = (t / s) * 8s + t % s
      const int base = (t / s) * 8 * s + (t % s);
#pragma unroll
      for (int i = 0; i < 8; i++) v[i] = sm[pad(base + i * s)];
      phase<3>(v, tw, base, s, P, rm);
#pragma unroll
      for (int i = 0; i < 8; i++) sm[pad(base + i * s)] = v[i];
    } else if (ns == 2) {   // two groups of 4 (slots 0..3 and 4..7): thread handles groups 2t, 2t+1
#pragma unroll
      for (int g = 0; g < 2; g++) { const int gi = 2 * t + g, base = (gi / s) * 4 * s + (gi % s);
#pragma unroll
        for (int i = 0; i < 4; i++) v[4 * g + i] = sm[pad(base + i * s)];
        phase<2>(v + 4 * g, tw, base, s, P, rm);
#pragma unroll
        for (int i = 0; i < 4; i++) sm[pad(base + i * s)] = v[4 * g + i]; }
    } else {   // ns == 1: four pairs
#pragma unroll
      for (int g = 0; g < 4; g++) { const int gi = 4 * t + g, base = (gi / s) * 2 * s + (gi % s);
        v[2 * g] = sm[pad(base)]; v[2 * g + 1] = sm[pad(base + s)];
        phase<1>(v + 2 * g, tw, base, s, P, rm);
        sm[pad(base)] = v[2 * g]; sm[pad(base + s)] = v[2 * g + 1]; }
    }
    __syncthreads();
  }
}
// inverse (Gentleman-Sande) phase, distances increasing: u + v / (u - v) w with w = itw[rm * h + i], h = 2^P / (2 d)
template <int NS> __device__ __forceinline__ void iphase(u64 *v, const Sh *__restrict__ itw, int base, int s, int P, int rm) {
  const int dl = (NS == 1 ? 1 : (1 << (NS - 1))) * s, q = rm * ((1 << P) / (2 * dl)) + base / (2 * dl);   // last stage: distance dl
#pragma unroll
  for (int st = 0; st < NS; st++) {
    const int D = NS == 1 ? 1 : (1 << st); const int d = D * s;
#pragma unroll
    for (int b = 0; b < (1 << (NS - 1)); b++) {
      const int lo = (b / D) * 2 * D + (b % D), hi = lo + D;
      // twiddle index rm * h + i, h = 2^P / (2 d): q for the last stage, (q << (NS - 1 - st)) + (b >> st) before it
      const u64 u = v[lo], w = v[hi]; v[lo] = addm(u, w); v[hi] = muls(subm(u, w), itw[(q << (NS - 1 - st)) + (b >> st)]);
    }
  }
}
template <int P> __device__ __forceinline__ void line_inv(u64 *sm, const Sh *__restrict__ itw, int rm) {
  const int t = threadIdx.x; u64 v[8];
#pragma unroll
  for (int e = 0; e < P; e += 3) {           // this phase covers distances 2^e, 2^(e+1), 2^(e+2) (as many as remain)
    const int ns = P - e >= 3 ? 3 : P - e; const int s = 1 << e;
    if (ns == 3) { const int base = (t / s) * 8 * s + (t % s);
#pragma unroll
      for (int i = 0; i < 8; i++) v[i] = sm[pad(base + i * s)];
      iphase<3>(v, itw, base, s, P, rm);
#pragma unroll
      for (int i = 0; i < 8; i++) sm[pad(base + i * s)] = v[i];
    } else if (ns == 2) {
#pragma unroll
      for (int g = 0; g < 2; g++) { const int gi = 2 * t + g, base = (gi / s) * 4 * s + (gi % s);
#pragma unroll
        for (int i = 0; i < 4; i++) v[4 * g + i] = sm[pad(base + i * s)];
        iphase<2>(v + 4 * g, itw, base, s, P, rm);
#pragma unroll
        for (int i = 0; i < 4; i++) sm[pad(base + i * s)] = v[4 * g + i]; }
    } else {
#pragma unroll
      for (int g = 0; g < 4; g++) { const int gi = 4 * t + g, base = (gi / s) * 2 * s + (gi % s);
        v[2 * g] = sm[pad(base)]; v[2 * g + 1] = sm[pad(base + s)];
        iphase<1>(v + 2 * g, itw, base, s, P, rm);
        sm[pad(base)] = v[2 * g]; sm[pad(base + s)] = v[2 * g + 1]; }
    }
    __syncthreads();
  }
}
// a thread's 8 coalesced elements of a 2^P line: i = threadIdx.x + e * 2^P / 8 (streaming loads/stores: the data passes
// through L1 once, the twiddle slice stays)
template <int P> __device__ __forceinline__ void ld8(u64 *r, const u64 *__restrict__ line) {
#pragma unroll
  for (int e = 0; e < 8; e++) r[e] = __ldcs(line + threadIdx.x + e * ((1 << P) / 8)); }
template <int P> __device__ __forceinline__ void to_sm(u64 *sm, const u64 *r) {
#pragma unroll
  for (int e = 0; e < 8; e++) sm[pad(threadIdx.x + e * ((1 << P) / 8))] = r[e]; }
template <int P> __device__ __forceinline__ void st8(u64 *line, const u64 *sm) {
#pragma unroll
  for (int e = 0; e < 8; e++) { const int i = threadIdx.x + e * ((1 << P) / 8); __stcs(line + i, sm[pad(i)]); } }
#ifndef TP_PREF_X
#define TP_PREF_X 1   // prefetch the next item's line into registers
#endif
#ifndef TP_PREF_B
#define TP_PREF_B 1   // load this item's b^ before the line transform
#endif
#define TP_LB(P) __launch_bounds__((1 << (P)) / 8, 4096 >> (P))   // 2^P/8 threads, registers capped at 128 (4 blocks of 128 threads ... 1 block of 512)

// column pass: blockIdx.x = column (contiguous N1 elements at x + item * N + col * N1), blockIdx.y = group of items;
// in place, the next item's column prefetched
template <int P1> __global__ void TP_LB(P1) col_fwd(u64 *__restrict__ x, const Sh *__restrict__ tw, size_t N, int items) {
  constexpr int N1 = 1 << P1; extern __shared__ u64 sm[];
  const int i0 = (int) (((long) blockIdx.y * items) / gridDim.y), i1 = (int) (((long) (blockIdx.y + 1) * items) / gridDim.y);
  u64 *col = x + (size_t) blockIdx.x * N1; u64 r[8];
#if TP_PREF_X
  if (i0 < i1) ld8<P1>(r, col + (size_t) i0 * N);
#endif
  for (int it = i0; it < i1; it++) {
#if TP_PREF_X
    u64 n[8]; if (it + 1 < i1) ld8<P1>(n, col + (size_t) (it + 1) * N);
#else
    ld8<P1>(r, col + (size_t) it * N);
#endif
    to_sm<P1>(sm, r); __syncthreads(); line_fwd<P1>(sm, tw, 1); st8<P1>(col + (size_t) it * N, sm); __syncthreads();
#if TP_PREF_X
#pragma unroll
    for (int e = 0; e < 8; e++) r[e] = n[e];
#endif
  }
}
// Row-pass twiddle modes.  SL (slice): row k1 reads the slice tw[(N1 + k1) m + i] of the merged N-entry table (no twist
// multiply; 16 N2 bytes per row per pass, from L2/HBM).  !SL (twist): the shared cyclic N2-entry table (L1-resident) and
// the row's twist zeta^n2, zeta = psi^(2 brev(k1) + 1), built per block in shared memory (Montgomery form) from two small
// per-row power tables (64 + N2/64 entries), one Montgomery multiply per element per item.  Both compute the same
// transform (merged table of row k1 = zeta^(N2 / 2m) times the cyclic table); the faster mode differs by GPU.
struct Twist { const u64 *A, *B; };   // A[k1 * 64 + i] = zeta^i, B[k1 * (N2/64) + j] = zeta^(64 j) 2^64 (Montgomery form)
template <int P2> __device__ __forceinline__ void build_twist(u64 *wt, Twist tz) {   // wt[n2] = zeta^n2 2^64 mod P for this block's row
  constexpr int N2 = 1 << P2, NT = N2 / 8; const u64 *A = tz.A + blockIdx.x * 64, *B = tz.B + blockIdx.x * (N2 / 64);
#pragma unroll
  for (int e = 0; e < 8; e++) { const int i = threadIdx.x + e * NT; wt[i] = mulm(A[i & 63], B[i >> 6]); }   // each thread writes the entries it reads
}
template <int P, bool SL> __device__ __forceinline__ void to_sm_tw(u64 *sm, const u64 *r, const u64 *wt) {
#pragma unroll
  for (int e = 0; e < 8; e++) { const int i = threadIdx.x + e * ((1 << P) / 8); sm[pad(i)] = SL ? r[e] : mont(r[e], wt[i]); } }
// row pass with the multiply-accumulate over the items: blockIdx.x = row k1 (position); the next item's row and this
// item's b^ (Montgomery form) prefetched; acc[row k1] = sum over items of NTT_row(x) . b^
template <int P2, bool SL> __global__ void TP_LB(P2) row_acc(const u64 *__restrict__ x, const Sh *__restrict__ tw, size_t N, int N1,
                                                              const u64 *__restrict__ bR, u64 *__restrict__ acc, int items, Twist tz) {
  constexpr int N2 = 1 << P2, NT = N2 / 8; extern __shared__ u64 sm[]; u64 *wt = sm + pad(N2) + 8;
  const int rm = SL ? N1 + blockIdx.x : 1; const size_t off = (size_t) blockIdx.x * N2;
  if (!SL) build_twist<P2>(wt, tz);
  u64 a[8], r[8];
#pragma unroll
  for (int e = 0; e < 8; e++) a[e] = 0;
#if TP_PREF_X
  ld8<P2>(r, x + off);
#endif
  for (int it = 0; it < items; it++) {
    u64 n[8], b[8];
#if TP_PREF_X
    if (it + 1 < items) ld8<P2>(n, x + (size_t) (it + 1) * N + off);
#else
    ld8<P2>(r, x + (size_t) it * N + off);
#endif
#if TP_PREF_B
    ld8<P2>(b, bR + (size_t) it * N + off);
#endif
    to_sm_tw<P2, SL>(sm, r, wt); __syncthreads(); line_fwd<P2>(sm, tw, rm);
#if !TP_PREF_B
    ld8<P2>(b, bR + (size_t) it * N + off);
#endif
#pragma unroll
    for (int e = 0; e < 8; e++) a[e] = addm(a[e], mont(sm[pad(threadIdx.x + e * NT)], b[e]));
    __syncthreads();
#if TP_PREF_X
#pragma unroll
    for (int e = 0; e < 8; e++) r[e] = n[e];
#endif
  }
#pragma unroll
  for (int e = 0; e < 8; e++) acc[off + threadIdx.x + e * NT] = a[e];
}
// row pass without the accumulate (the public a, precomputation): in place
template <int P2, bool SL> __global__ void TP_LB(P2) row_fwd(u64 *__restrict__ x, const Sh *__restrict__ tw, int N1, Twist tz) {
  constexpr int N2 = 1 << P2; extern __shared__ u64 sm[]; u64 *wt = sm + pad(N2) + 8; u64 *line = x + (size_t) blockIdx.x * N2; u64 r[8];
  if (!SL) build_twist<P2>(wt, tz);
  ld8<P2>(r, line); to_sm_tw<P2, SL>(sm, r, wt); __syncthreads(); line_fwd<P2>(sm, tw, SL ? N1 + blockIdx.x : 1); st8<P2>(line, sm);
}
// inverse row pass (once per output): bit-reversed in; SL: the row's inverse slice, !SL: cyclic inverse then the inverse twist; in place
template <int P2, bool SL> __global__ void TP_LB(P2) row_inv(u64 *__restrict__ x, const Sh *__restrict__ itw, int N1, Twist tzi) {
  constexpr int N2 = 1 << P2, NT = N2 / 8; extern __shared__ u64 sm[]; u64 *wt = sm + pad(N2) + 8; u64 *line = x + (size_t) blockIdx.x * N2; u64 r[8];
  if (!SL) build_twist<P2>(wt, tzi);
  ld8<P2>(r, line); to_sm<P2>(sm, r); __syncthreads(); line_inv<P2>(sm, itw, SL ? N1 + blockIdx.x : 1);
  if (SL) st8<P2>(line, sm);
  else {
#pragma unroll
    for (int e = 0; e < 8; e++) { const int i = threadIdx.x + e * NT; __stcs(line + i, mont(sm[pad(i)], wt[i])); }
  }
}
// inverse column pass (once per output): column contiguous after the transpose; GS inverse, times 1/N
template <int P1> __global__ void TP_LB(P1) col_inv(u64 *__restrict__ x, const Sh *__restrict__ itw, Sh ninv) {
  constexpr int N1 = 1 << P1, NT = N1 / 8; extern __shared__ u64 sm[]; u64 *line = x + (size_t) blockIdx.x * N1; u64 r[8];
  ld8<P1>(r, line); to_sm<P1>(sm, r); __syncthreads(); line_inv<P1>(sm, itw, 1);
#pragma unroll
  for (int e = 0; e < 8; e++) { const int i = threadIdx.x + e * NT; __stcs(line + i, muls(sm[pad(i)], ninv)); }
}
// tables: tw[k] = base^brev_L(k) as a Shoup pair (base = psi for the forward table, psi^-1 for the inverse one)
__global__ void build_tw(Sh *__restrict__ out, u64 base, int L) {
  const size_t k = (size_t) blockIdx.x * blockDim.x + threadIdx.x; if (k >= (1ull << L)) return;
  const u64 w = powm_d(base, __brev((unsigned) k) >> (32 - L)); out[k] = Sh{w, (u64) (((u128) w << 64) / tP)};
}
// twist power tables of every row: block = row k1, zeta = base^(2 brev_P1(k1) + 1); A: zeta^i (i < 64), B: zeta^(64 j) scale
__global__ void build_twist_tab(u64 *__restrict__ A, u64 *__restrict__ B, u64 base, int P1, int N2, u64 scale) {
  const u64 z = powm_d(base, 2ull * (__brev(blockIdx.x) >> (32 - P1)) + 1); const int t = threadIdx.x;
  if (t < 64) A[blockIdx.x * 64 + t] = powm_d(z, t);
  else if (t < 64 + N2 / 64) B[blockIdx.x * (N2 / 64) + t - 64] = mulm(powm_d(z, 64ull * (t - 64)), scale);
}
// b^ -> Montgomery form (b^ 2^64 mod P), part of the b^ precomputation
__global__ void to_mont(u64 *__restrict__ x, size_t n, u64 rmodp) {
  for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) x[i] = mulm(x[i], rmodp); }

struct Plan {
  int L, P1, P2; u64 psi, psii, ninv, rmodp; Sh ninvs;
  Sh *twf = nullptr, *itwf = nullptr;      // merged N-entry tables (slice mode; the column passes use the first N1 entries)
  Sh *twc = nullptr, *itwc = nullptr;      // cyclic N2-entry tables (twist mode)
  u64 *twA = nullptr, *twB = nullptr, *twAi = nullptr, *twBi = nullptr;   // twist power tables of every row (twist mode)
  Twist tz() const { return Twist{twA, twB}; } Twist tzi() const { return Twist{twAi, twBi}; }
  static u64 mulm_h(u64 a, u64 b, u64 P) { return (u64) ((u128) a * b % P); }
  static u64 powm_h(u64 a, u64 e, u64 P) { u64 r = 1; while (e) { if (e & 1) r = mulm_h(r, a, P); a = mulm_h(a, a, P); e >>= 1; } return r; }
  static Sh sh(u64 w, u64 P) { return Sh{w, (u64) (((u128) w << 64) / P)}; }
  Plan(int L_, int P1_, u64 PP) : L(L_), P1(P1_), P2(L_ - P1_) {
    const size_t N = 1ull << L; const int N1 = 1 << P1, N2 = 1 << P2;
    u64 g = 3; while (powm_h(g, (PP - 1) / 2, PP) == 1) g++;
    psi = powm_h(g, (PP - 1) / (2 * N), PP); psii = powm_h(psi, PP - 2, PP); ninv = powm_h(N % PP, PP - 2, PP); ninvs = sh(ninv, PP);
    const u64 mu = (u64) (((u128) 1 << 124) / PP); rmodp = (u64) (((u128) 1 << 64) % PP);
    u64 qi = PP; for (int i = 0; i < 6; i++) qi *= 2 - PP * qi;   // P^-1 mod 2^64 (Newton, P odd)
    cudaMemcpyToSymbol(tP, &PP, 8); cudaMemcpyToSymbol(tMU, &mu, 8); cudaMemcpyToSymbol(tQI, &qi, 8);
    cudaMalloc(&twf, N * sizeof(Sh)); cudaMalloc(&itwf, N * sizeof(Sh));
    build_tw<<<(unsigned) ((N + 255) / 256), 256>>>(twf, psi, L); build_tw<<<(unsigned) ((N + 255) / 256), 256>>>(itwf, psii, L);
    // cyclic tables: c[m + i] = omega_2m^brev_j(i) (m = 2^j), omega_2m = psi^(N / m)
    std::vector<Sh> c(N2), ci(N2); c[0] = ci[0] = sh(1, PP);
    for (int m = 1, j = 0; m < N2; m <<= 1, j++) for (int i = 0; i < m; i++) {
      int r = 0; for (int q = 0; q < j; q++) r |= ((i >> q) & 1) << (j - 1 - q);
      c[m + i] = sh(powm_h(psi, (N / m) * (u64) r, PP), PP); ci[m + i] = sh(powm_h(psii, (N / m) * (u64) r, PP), PP); }
    cudaMalloc(&twc, N2 * sizeof(Sh)); cudaMalloc(&itwc, N2 * sizeof(Sh));
    cudaMemcpy(twc, c.data(), N2 * sizeof(Sh), cudaMemcpyHostToDevice); cudaMemcpy(itwc, ci.data(), N2 * sizeof(Sh), cudaMemcpyHostToDevice);
    cudaMalloc(&twA, (size_t) N1 * 64 * 8); cudaMalloc(&twB, (size_t) N1 * (N2 / 64) * 8); cudaMalloc(&twAi, (size_t) N1 * 64 * 8); cudaMalloc(&twBi, (size_t) N1 * (N2 / 64) * 8);
    build_twist_tab<<<N1, 64 + N2 / 64>>>(twA, twB, psi, P1, N2, rmodp); build_twist_tab<<<N1, 64 + N2 / 64>>>(twAi, twBi, psii, P1, N2, rmodp);
    cudaDeviceSynchronize();
  }
  ~Plan() { for (void *p : {(void *) twf, (void *) itwf, (void *) twc, (void *) itwc, (void *) twA, (void *) twB, (void *) twAi, (void *) twBi}) cudaFree(p); }
};
template <int P, bool SL = true> static constexpr int line_smem() { return ((1 << P) + ((1 << P) >> 3) + 8 + (SL ? 0 : (1 << P))) * 8; }
}   // namespace tp
