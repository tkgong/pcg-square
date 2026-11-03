// Two-pass streaming NTT for PCG^2 (no Cheddar code).  N = N1 x N2.  Column pass: N1-point negacyclic transforms on
// contiguous columns (the layout the SPUs write), output contiguous (column-major); the DRU transposes to row-major in
// memory; row pass: N2-point cyclic transforms on contiguous rows with the four-step twist folded into the load, the b^
// product and the accumulate fused into the row pass over the items.  Every block streams one contiguous 2^P-element
// line: 2^P/8 threads, 8 elements per thread, 3 radix-2 stages per register phase, shared-memory exchange between phases
// (padded, 2-way conflicts at worst), twiddles as Shoup pairs from an L1-resident table.  Data multiplies: Barrett.
#pragma once
namespace tp {
using u64 = ::u64;
typedef unsigned __int128 u128;
__constant__ u64 tP, tMU;
struct Sh { u64 w, q; };
__device__ __forceinline__ u64 addm(u64 a, u64 b) { u64 s = a + b; return s >= tP ? s - tP : s; }
__device__ __forceinline__ u64 subm(u64 a, u64 b) { return a >= b ? a - b : a + tP - b; }
__device__ __forceinline__ u64 mulm(u64 a, u64 b) {
  const u64 hi = __umul64hi(a, b), lo = a * b, x1 = (hi << 4) | (lo >> 60), q = __umul64hi(x1, tMU);
  u64 r = lo - q * tP; while (r >= tP) r -= tP; return r; }
__device__ __forceinline__ u64 muls(u64 a, Sh w) { const u64 q = __umul64hi(a, w.q), r = a * w.w - q * tP; return r >= tP ? r - tP : r; }
__device__ __forceinline__ u64 powm_d(u64 a, u64 e) { u64 r = 1; while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; } return r; }
__device__ __forceinline__ int pad(int p) { return p + (p >> 3); }   // shared-memory index with one pad word per 8

// one register phase of up to 3 stages on the thread's 8 elements v[0..7] (element i at logical offset i * s from the
// thread's base), largest distance first.  CT (negacyclic, merged psi: u + w v / u - w v, w = tw[m + i]) when CT, else
// GS-DIF (u + v / (u - v) w, w = tw[j << (log2 of the stage's "m")]).  NS = stages in this phase (3, 2 or 1).
// Generic radix-2 on registers: distance D in elements (4, 2, 1 register slots = 4s, 2s, s logical), group index i and
// position j of each butterfly in the line are derived from the thread's base and the slot.
template <bool CT, int NS> __device__ __forceinline__ void phase(u64 *v, const Sh *__restrict__ tw, int base, int s, int P) {
  // stage distances in logical elements: 4s, 2s, s (NS = 3); 2s, s (NS = 2: slots [0..3] and [4..7] are two groups); s (NS = 1: four pairs)
#pragma unroll
  for (int st = 0; st < NS; st++) {
    const int D = NS == 3 ? (4 >> st) : NS == 2 ? (2 >> st) : 1;   // slot distance
    const int d = D * s;                                             // logical distance t
#pragma unroll
    for (int b = 0; b < (1 << (NS - 1)); b++) {                      // slots/2 butterflies per stage
      const int lo = (b / D) * 2 * D + (b % D), hi = lo + D;
      const int p = base + lo * s;                                   // logical position of the butterfly's upper element
      const int i = p / (2 * d), j = p % d;                           // group, position within the group (d = t)
      if (CT) { const int m = (1 << P) / (2 * d); const u64 u = v[lo], w = muls(v[hi], tw[m + i]); v[lo] = addm(u, w); v[hi] = subm(u, w); }
      else    { const int lg = __ffs((1 << P) / (2 * d)) - 1; const u64 u = v[lo], w = v[hi]; v[lo] = addm(u, w); v[hi] = muls(subm(u, w), tw[j << lg]); }
    }
  }
}
// forward line transform in place in shared memory sm[pad(.)], 2^P elements, P in 10..13; CT = negacyclic column pass
// (tw = psi_brv table), !CT = cyclic row pass (tw = omega powers).  Phases: distances (2^(P-1), 2^(P-2), 2^(P-3)), ... down to 1.
template <bool CT, int P> __device__ __forceinline__ void line_fwd(u64 *sm, const Sh *__restrict__ tw) {
  constexpr int NT = (1 << P) / 8; const int t = threadIdx.x;
  u64 v[8];
#pragma unroll
  for (int e = P - 1; e >= 0; e -= 3) {         // this phase covers distances 2^e, 2^(e-1), 2^(e-2) (as many as remain)
    const int ns = e >= 2 ? 3 : e + 1;         // stages in this phase
    const int s = 1 << (e - ns + 1);           // smallest distance of the phase = slot stride
    if (ns == 3) {   // 8 slots = 8 consecutive multiples of s: base = (t / s) * 8s + t % s
      const int base = (t / s) * 8 * s + (t % s);
#pragma unroll
      for (int i = 0; i < 8; i++) v[i] = sm[pad(base + i * s)];
      phase<CT, 3>(v, tw, base, s, P);
#pragma unroll
      for (int i = 0; i < 8; i++) sm[pad(base + i * s)] = v[i];
    } else if (ns == 2) {   // two groups of 4 (slots 0..3 and 4..7), each 4 consecutive multiples of s: thread handles groups 2t, 2t+1
#pragma unroll
      for (int g = 0; g < 2; g++) { const int gi = 2 * t + g, base = (gi / s) * 4 * s + (gi % s);
#pragma unroll
        for (int i = 0; i < 4; i++) v[4 * g + i] = sm[pad(base + i * s)];
        phase<CT, 2>(v + 4 * g, tw, base, s, P);
#pragma unroll
        for (int i = 0; i < 4; i++) sm[pad(base + i * s)] = v[4 * g + i]; }
    } else {   // ns == 1: four pairs
#pragma unroll
      for (int g = 0; g < 4; g++) { const int gi = 4 * t + g, base = (gi / s) * 2 * s + (gi % s);
        v[2 * g] = sm[pad(base)]; v[2 * g + 1] = sm[pad(base + s)];
        phase<CT, 1>(v + 2 * g, tw, base, s, P);
        sm[pad(base)] = v[2 * g]; sm[pad(base + s)] = v[2 * g + 1]; }
    }
    __syncthreads();
  }
  (void) NT;
}
// column pass: blockIdx.x = column (contiguous N1 elements at in + col * N1), blockIdx.y = item; in place
template <int P1> __global__ void __launch_bounds__((1 << P1) / 8, 1) col_fwd(u64 *__restrict__ x, const Sh *__restrict__ tw, size_t N) {
  constexpr int N1 = 1 << P1; extern __shared__ u64 sm[];
  u64 *line = x + (size_t) blockIdx.y * N + (size_t) blockIdx.x * N1;
  for (int i = threadIdx.x; i < N1; i += blockDim.x) sm[pad(i)] = line[i];
  __syncthreads();
  line_fwd<true, P1>(sm, tw);
  for (int i = threadIdx.x; i < N1; i += blockDim.x) line[i] = sm[pad(i)];
}
// row pass with the twist psi^((2 k1 + 1) n2) on the load and the multiply-accumulate over the items; blockIdx.x = row k1
template <int P2> __global__ void __launch_bounds__((1 << P2) / 8, 1) row_acc(const u64 *__restrict__ x, const Sh *__restrict__ tw, u64 psi, size_t N,
                                                                 const u64 *__restrict__ bhat, u64 *__restrict__ acc, int items, int P1) {
  constexpr int N2 = 1 << P2; extern __shared__ u64 sm[]; u64 *A = sm + pad(N2) + 8, *B = A + 64;
  const int k1 = __brev(blockIdx.x) >> (32 - P1); constexpr int NT = N2 / 8;   // the column pass left k1 bit-reversed in the row index
  { const u64 base = powm_d(psi, 2 * (u64) k1 + 1);
    if (threadIdx.x < 64) A[threadIdx.x] = powm_d(base, threadIdx.x);
    else if (threadIdx.x < 64 + N2 / 64) B[threadIdx.x - 64] = powm_d(base, 64ull * (threadIdx.x - 64)); }
  __syncthreads();
  u64 a[8], wt[8];
#pragma unroll
  for (int e = 0; e < 8; e++) { const int i = threadIdx.x + e * NT; wt[e] = mulm(A[i & 63], B[i >> 6]); a[e] = 0; }
  for (int it = 0; it < items; it++) {
    const u64 *line = x + (size_t) it * N + (size_t) blockIdx.x * N2;
#pragma unroll
    for (int e = 0; e < 8; e++) { const int i = threadIdx.x + e * NT; sm[pad(i)] = mulm(line[i], wt[e]); }
    __syncthreads();
    line_fwd<false, P2>(sm, tw);
    const u64 *bh = bhat + (size_t) it * N + (size_t) blockIdx.x * N2;
#pragma unroll
    for (int e = 0; e < 8; e++) { const int i = threadIdx.x + e * NT; a[e] = addm(a[e], mulm(sm[pad(i)], bh[i])); }
    __syncthreads();
  }
  u64 *dst = acc + (size_t) blockIdx.x * N2;
#pragma unroll
  for (int e = 0; e < 8; e++) dst[threadIdx.x + e * NT] = a[e];
}
// row pass without the accumulate (the public a): in place, twist applied, output stays in the row (bit-reversed order within it)
template <int P2> __global__ void __launch_bounds__((1 << P2) / 8, 1) row_fwd(u64 *__restrict__ x, const Sh *__restrict__ tw, u64 psi, size_t N, int P1) {
  constexpr int N2 = 1 << P2; extern __shared__ u64 sm[]; u64 *A = sm + pad(N2) + 8, *B = A + 64;
  const int k1 = __brev(blockIdx.x) >> (32 - P1); constexpr int NT = N2 / 8;   // the column pass left k1 bit-reversed in the row index
  { const u64 base = powm_d(psi, 2 * (u64) k1 + 1);
    if (threadIdx.x < 64) A[threadIdx.x] = powm_d(base, threadIdx.x);
    else if (threadIdx.x < 64 + N2 / 64) B[threadIdx.x - 64] = powm_d(base, 64ull * (threadIdx.x - 64)); }
  __syncthreads();
  u64 *line = x + (size_t) blockIdx.x * N2;
  for (int i = threadIdx.x; i < N2; i += NT) sm[pad(i)] = mulm(line[i], mulm(A[i & 63], B[i >> 6]));
  __syncthreads();
  line_fwd<false, P2>(sm, tw);
  for (int i = threadIdx.x; i < N2; i += NT) line[i] = sm[pad(i)];
}
// inverse line transform: distances increasing (1, 2, 4, ... 2^(P-1)); CT = negacyclic GS inverse (itw = psi^-1 brv), !CT = cyclic CT-DIT (itw = omega^-j)
template <bool NEGA, int NS> __device__ __forceinline__ void iphase(u64 *v, const Sh *__restrict__ itw, int base, int s, int P) {
#pragma unroll
  for (int st = 0; st < NS; st++) {
    const int D = NS == 3 ? (1 << st) : NS == 2 ? (1 << st) : 1; const int d = D * s;
#pragma unroll
    for (int b = 0; b < (1 << (NS - 1)); b++) {
      const int lo = (b / D) * 2 * D + (b % D), hi = lo + D; const int p = base + lo * s; const int i = p / (2 * d), j = p % d;
      if (NEGA) { const int h = (1 << P) / (2 * d); const u64 u = v[lo], w = v[hi]; v[lo] = addm(u, w); v[hi] = muls(subm(u, w), itw[h + i]); }
      else      { const int lg = __ffs((1 << P) / (2 * d)) - 1; const u64 u = v[lo], w = muls(v[hi], itw[j << lg]); v[lo] = addm(u, w); v[hi] = subm(u, w); }
    }
  }
}
template <bool NEGA, int P> __device__ __forceinline__ void line_inv(u64 *sm, const Sh *__restrict__ itw) {
  const int t = threadIdx.x; u64 v[8];
#pragma unroll
  for (int e = 0; e < P; e += 3) {           // this phase covers distances 2^e, 2^(e+1), 2^(e+2) (as many as remain)
    const int ns = P - e >= 3 ? 3 : P - e; const int s = 1 << e;
    if (ns == 3) { const int base = (t / s) * 8 * s + (t % s);
#pragma unroll
      for (int i = 0; i < 8; i++) v[i] = sm[pad(base + i * s)];
      iphase<NEGA, 3>(v, itw, base, s, P);
#pragma unroll
      for (int i = 0; i < 8; i++) sm[pad(base + i * s)] = v[i];
    } else if (ns == 2) {
#pragma unroll
      for (int g = 0; g < 2; g++) { const int gi = 2 * t + g, base = (gi / s) * 4 * s + (gi % s);
#pragma unroll
        for (int i = 0; i < 4; i++) v[4 * g + i] = sm[pad(base + i * s)];
        iphase<NEGA, 2>(v + 4 * g, itw, base, s, P);
#pragma unroll
        for (int i = 0; i < 4; i++) sm[pad(base + i * s)] = v[4 * g + i]; }
    } else {
#pragma unroll
      for (int g = 0; g < 4; g++) { const int gi = 4 * t + g, base = (gi / s) * 2 * s + (gi % s);
        v[2 * g] = sm[pad(base)]; v[2 * g + 1] = sm[pad(base + s)];
        iphase<NEGA, 1>(v + 2 * g, itw, base, s, P);
        sm[pad(base)] = v[2 * g]; sm[pad(base + s)] = v[2 * g + 1]; }
    }
    __syncthreads();
  }
}
// inverse row pass (once per output): bit-reversed in -> natural, then the inverse twist; in place
template <int P2> __global__ void __launch_bounds__((1 << P2) / 8, 1) row_inv(u64 *__restrict__ x, const Sh *__restrict__ itw, u64 psii, size_t N, int P1) {
  constexpr int N2 = 1 << P2; extern __shared__ u64 sm[]; u64 *A = sm + pad(N2) + 8, *B = A + 64;
  const int k1 = __brev(blockIdx.x) >> (32 - P1); constexpr int NT = N2 / 8;   // the column pass left k1 bit-reversed in the row index
  { const u64 base = powm_d(psii, 2 * (u64) k1 + 1);
    if (threadIdx.x < 64) A[threadIdx.x] = powm_d(base, threadIdx.x);
    else if (threadIdx.x < 64 + N2 / 64) B[threadIdx.x - 64] = powm_d(base, 64ull * (threadIdx.x - 64)); }
  u64 *line = x + (size_t) blockIdx.x * N2;
  for (int i = threadIdx.x; i < N2; i += NT) sm[pad(i)] = line[i];
  __syncthreads();
  line_inv<false, P2>(sm, itw);
  for (int i = threadIdx.x; i < N2; i += NT) line[i] = mulm(sm[pad(i)], mulm(A[i & 63], B[i >> 6]));
}
// inverse column pass (once per output): column contiguous with k1 natural (after the transpose); loads bit-reversed, GS inverse, 1/N
template <int P1> __global__ void __launch_bounds__((1 << P1) / 8, 1) col_inv(u64 *__restrict__ x, const Sh *__restrict__ itw, u64 ninv) {
  constexpr int N1 = 1 << P1; extern __shared__ u64 sm[];
  u64 *line = x + (size_t) blockIdx.x * N1;
  for (int i = threadIdx.x; i < N1; i += blockDim.x) sm[pad(i)] = line[i];   // the CT output order the GS inverse expects
  __syncthreads();
  line_inv<true, P1>(sm, itw);
  for (int i = threadIdx.x; i < N1; i += blockDim.x) line[i] = mulm(sm[pad(i)], ninv);
}

struct Plan {
  int L, P1, P2; u64 psi, psii, ninv; Sh *tw1 = nullptr, *tw1i = nullptr, *tw2 = nullptr, *tw2i = nullptr;
  static u64 mulm_h(u64 a, u64 b, u64 P) { return (u64) ((u128) a * b % P); }
  static u64 powm_h(u64 a, u64 e, u64 P) { u64 r = 1; while (e) { if (e & 1) r = mulm_h(r, a, P); a = mulm_h(a, a, P); e >>= 1; } return r; }
  static Sh sh(u64 w, u64 P) { return Sh{w, (u64) (((u128) w << 64) / P)}; }
  Plan(int L_, int P1_, u64 PP) : L(L_), P1(P1_), P2(L_ - P1_) {
    const size_t N = 1ull << L; const int N1 = 1 << P1, N2 = 1 << P2;
    u64 g = 3; while (powm_h(g, (PP - 1) / 2, PP) == 1) g++;
    psi = powm_h(g, (PP - 1) / (2 * N), PP); psii = powm_h(psi, PP - 2, PP); ninv = powm_h(N % PP, PP - 2, PP);
    const u64 mu = (u64) (((u128) 1 << 124) / PP);
    cudaMemcpyToSymbol(tP, &PP, 8); cudaMemcpyToSymbol(tMU, &mu, 8);
    const u64 psi1 = powm_h(psi, N2, PP), psi1i = powm_h(psii, N2, PP), om2 = powm_h(psi, 2ull * N1, PP), om2i = powm_h(psii, 2ull * N1, PP);
    std::vector<Sh> a(N1), ai(N1), b(N2 / 2), bi(N2 / 2);
    for (int k = 0; k < N1; k++) { int r = 0; for (int q = 0; q < P1; q++) r |= ((k >> q) & 1) << (P1 - 1 - q); a[k] = sh(powm_h(psi1, r, PP), PP); ai[k] = sh(powm_h(psi1i, r, PP), PP); }
    u64 w = 1, wi = 1; for (int j = 0; j < N2 / 2; j++) { b[j] = sh(w, PP); bi[j] = sh(wi, PP); w = mulm_h(w, om2, PP); wi = mulm_h(wi, om2i, PP); }
    auto up = [](Sh **d, const std::vector<Sh> &h) { cudaMalloc(d, h.size() * sizeof(Sh)); cudaMemcpy(*d, h.data(), h.size() * sizeof(Sh), cudaMemcpyHostToDevice); };
    up(&tw1, a); up(&tw1i, ai); up(&tw2, b); up(&tw2i, bi);
  }
  ~Plan() { cudaFree(tw1); cudaFree(tw1i); cudaFree(tw2); cudaFree(tw2i); }
};
template <int P> static constexpr int line_smem() { return ((1 << P) + ((1 << P) >> 3) + 8 + 64 + ((1 << P) >> 6) + 8) * 8; }
}   // namespace tp
