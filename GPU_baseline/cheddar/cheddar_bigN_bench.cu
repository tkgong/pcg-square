// Cheddar's two-kernel NTT vs the four-step on Cheddar's kernels (transposes on the DRU) vs GPU-NTT merge, per negacyclic
// multiply, at one logN (compile with -DBIGN_LOGN=24..28). Same kernels, dataflow and timing as cheddar_f4dru_bench.cu; the
// differences are only what large N needs:
//   * configs are swept here (no forward sweep): Cheddar  P2 (row stages) in [logN-14, 14], column M1, warp batching B1, row M2;
//     four-step  P1 (column stages) in [logN-14, 14], column M1, warp batching B1 (2^B1 contiguous columns per block), row M2.  Stage merging M in {3, 4, 5} on both sides (a pass of
//     14 stages needs M >= 4 to stay within 1024 threads); the row kernel's M2 is swept for both designs (fixed 3 in the 2^20..2^24 bench).
//   * a config is skipped when a kernel cannot launch: threads > the kernel's maxThreadsPerBlock (register file) or shared memory
//     > the opt-in limit; any other launch error is reported on stderr and the config is skipped.
//   * the reference product is computed one polynomial at a time with GPU-NTT merge (its 32-bit addressing overflows at
//     batch x N > 2^32 elements); the batched merge product is checked against it too (merge row's mismatch column).
//   * Cheddar's per-polynomial offset is 64-bit (extract.sh), so batch x N > 2^31 is addressed correctly.
//   * a 14-stage pass (needed at 2^27, 2^28) needs 70..96 registers per thread at M 4 (1024 threads) and 122..228 at M 5
//     (512 threads), over the 64K-register file of a block; -DBIGN_MAXRREG=64 / 128 (nvcc -maxrregcount) builds a variant that
//     caps every kernel of the binary (spilling to local memory), recorded in the maxrreg column (0 = no cap).
// Output: logN,batch,design,P1,P2,M1,B1,M2,maxrreg,ms_per_poly,mismatches
#include "fused4/fused4_ntt.cuh"   // -I.. (GPU_baseline)
#include "gpuntt/common/nttparameters.cuh"
#include "gen/cheddar_ntt_contig.cuh"
#include "gen/cheddar_ntt_acc.cuh"
#include <cstdio>
#ifndef BIGN_MAXRREG
#define BIGN_MAXRREG 0
#endif
#ifndef BIGN_LOGN
#error "compile with -DBIGN_LOGN=<20..28>"
#endif
#include <cstring>
#include <vector>
#include <utility>
#include <functional>
using namespace gpuntt;
using u64 = uint64_t; using i64 = int64_t; using u128 = unsigned __int128;
// PCG's prime 2^62 - 3*2^25 + 1 has 2-adicity 25 (negacyclic N <= 2^24); above that the 62-bit prime 4611685941117976577
// (2-adicity 33) of the 2^29/2^30 merge job. Kernel time does not depend on which 62-bit prime is used.
static constexpr u64 P = BIGN_LOGN <= 24 ? 4611686018326724609ULL : 4611685941117976577ULL;
static void ck(cudaError_t e, const char *w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }
static u64 mulm(u64 a, u64 b) { return (u64) ((u128) a * b % P); }
static u64 powm(u64 a, u64 e) { u64 r = 1; while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; } return r; }
static u64 tomont(u64 a) { return (u64) (((u128) a << 64) % P); }
__global__ void fill(u64 *x, size_t n, u64 seed) {
  for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
    u64 z = seed + 0x9e3779b97f4a7c15ULL * (i + 1); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; z ^= z >> 31;
    x[i] = z % P;
  }
}
__global__ void count_diff(const u64 *a, const u64 *b, size_t n, unsigned long long *cnt) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) c += a[i] != b[i];
  if (c) atomicAdd(cnt, c);
}
__global__ void addmod_kernel(u64 *z, const u64 *x, size_t n, Modulus<Data64> mod) {   // z += x
  for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) z[i] = OPERATOR_GPU<Data64>::add(z[i], x[i], mod);
}
__global__ void reduce_partials(u64 *p, int G, size_t n, Modulus<Data64> mod) {   // p[0] = sum of the G partial polynomials
  for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
    u64 s = p[i]; for (int g = 1; g < G; g++) s = OPERATOR_GPU<Data64>::add(s, p[g * n + i], mod); p[i] = s;
  }
}
__global__ void set_one(u64 *x, size_t n) {   // the constant polynomial 1 (a_0 = 1 in the PCG output)
  for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) x[i] = i == 0;
}
__global__ void transpose(const u64 *__restrict__ src, u64 *__restrict__ dst, int R, int C) {   // per polynomial (blockIdx.z): R x C -> C x R
  __shared__ u64 t[32][33];
  const size_t off = (size_t) blockIdx.z * R * C; src += off; dst += off;
  int c = blockIdx.x * 32 + threadIdx.x, r = blockIdx.y * 32 + threadIdx.y;
  for (int k = 0; k < 32; k += 8) t[threadIdx.y + k][threadIdx.x] = src[(size_t) (r + k) * C + c];
  __syncthreads();
  r = blockIdx.x * 32 + threadIdx.y; c = blockIdx.y * 32 + threadIdx.x;
  for (int k = 0; k < 32; k += 8) dst[(size_t) (r + k) * R + c] = t[threadIdx.x][threadIdx.y + k];
}
static void T(const u64 *s, u64 *d, int R, int C, int batch) { transpose<<<dim3(C / 32, R / 32, batch), dim3(32, 8)>>>(s, d, R, C); }
// column-contiguous layout -> the register order of Cheddar's column kernel (NTTPhase1RegIn): within every column of 2^P1
// elements, element n1 = s + i * 2^(P1-M) (thread s, register i) moves to s * 2^M + i
__global__ void perm_reg(const u64 *src, u64 *dst, int P1, int M, size_t n) {
  for (size_t idx = blockIdx.x * (size_t) blockDim.x + threadIdx.x; idx < n; idx += (size_t) gridDim.x * blockDim.x) {
    const size_t col = idx >> P1; const int n1 = (int) (idx & ((1u << P1) - 1)); const int i = n1 >> (P1 - M), sidx = n1 & ((1 << (P1 - M)) - 1);
    dst[(col << P1) + ((size_t) sidx << M) + i] = src[idx];
  }
}
struct MergePlan {   // fused4_bench.cu:73-117
  int logN; NTTParameters<Data64> params; Modulus<Data64> modulus; Root<Data64> *fwd = nullptr, *inv = nullptr; ntt_configuration<Data64> cfg_fwd{}, cfg_inv{};
  static NTTParameters<Data64> make(int lg) {
    const int N = 1 << lg; const uint64_t psi = pcg_cuda::negacyclic_psi(P, N); const uint64_t omega = pcg_cuda::mod_mul_u64(psi, psi, P);
    NTTFactors<Data64> f(Modulus<Data64>(P), omega, psi); return NTTParameters<Data64>(lg, f, ReductionPolynomial::X_N_plus);
  }
  explicit MergePlan(int lg) : logN(lg), params(make(lg)), modulus(params.modulus) {
    auto f = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
    auto i = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);
    const size_t bytes = static_cast<size_t>(params.root_of_unity_size) * sizeof(Root<Data64>);
    ck(cudaMalloc(&fwd, bytes), "fwd"); ck(cudaMalloc(&inv, bytes), "inv");
    cudaMemcpy(fwd, f.data(), bytes, cudaMemcpyHostToDevice); cudaMemcpy(inv, i.data(), bytes, cudaMemcpyHostToDevice);
    cfg_fwd.n_power = lg; cfg_fwd.ntt_type = FORWARD; cfg_fwd.ntt_layout = PerPolynomial; cfg_fwd.reduction_poly = ReductionPolynomial::X_N_plus;
    cfg_fwd.zero_padding = false; cfg_fwd.stream = 0; cfg_inv = cfg_fwd; cfg_inv.ntt_type = INVERSE; cfg_inv.mod_inverse = params.n_inv;
  }
  ~MergePlan() { cudaFree(fwd); cudaFree(inv); }
  void multiply(Data64 *a, Data64 *b, int batch) const {
    GPU_NTT_Inplace<Data64>(a, fwd, modulus, cfg_fwd, batch); GPU_NTT_Inplace<Data64>(b, fwd, modulus, cfg_fwd, batch);
    const size_t total = static_cast<size_t>(batch) << logN;
    fused4::pointwise_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(a, b, modulus, total);
    GPU_INTT_Inplace<Data64>(a, inv, modulus, cfg_inv, batch);
  }
};
template <int Pn, int M, int Bt> struct CCfg {
  __host__ __device__ static constexpr int RadixStages() { return Pn; }
  __host__ __device__ static constexpr int StageMerging() { return M; }
  __host__ __device__ static constexpr int LogWarpBatching() { return Bt; }
  __host__ __device__ static constexpr int LsbSize() { return 32; }
  __host__ __device__ static constexpr bool OFTwiddle() { return true; }
};
template <int S, int E, class F> constexpr void cfor(F &&f) { if constexpr (S < E) { f(std::integral_constant<int, S>()); cfor<S + 1, E>(f); } }
constexpr bool thr_ok(int lg) { return lg >= 5 && lg <= 10; }   // 32..1024 threads
static std::string DESIGNS = "merge,cheddar,f4c";
static bool wantd(const char *v) { return ("," + DESIGNS + ",").find("," + std::string(v) + ",") != std::string::npos; }
static bool EXT_ONLY = false;   // --ext: only the configs outside the original sweep (M1 = 6, M2 = 6 or B1 = 4)
static bool ext_skip(int m1, int b1, int m2) { return EXT_ONLY && !(m1 == 6 || m2 == 6 || b1 == 4); }
struct Tables {   // NTT.cu PopulateTwiddleFactors for one prime; per-polynomial constants for up to maxb polynomials
  int L; u64 *tw, *twm, *itw, *itwm, *primes, *ninv; i64 *qinv;
  Tables(int L_, int maxb) : L(L_) {
    const size_t N = 1ull << L; u64 g = 3; while (powm(g, (P - 1) / 2) == 1) g++;
    u64 psi = powm(g, (P - 1) / (2 * N)), psii = powm(psi, P - 2);
    std::vector<u64> f(N), fi(N), m(N / 32), mi(N / 32);
    { std::vector<u64> a(N), ai(N);
      a[0] = ai[0] = 1; for (size_t j = 1; j < N; j++) { a[j] = mulm(a[j - 1], psi); ai[j] = mulm(ai[j - 1], psii); }
      for (size_t j = 0; j < N; j++) { size_t r = 0; for (int k = 0; k < L; k++) r |= ((j >> k) & 1) << (L - 1 - k); f[r] = a[j]; fi[r] = ai[j]; } }
    for (size_t j = 0; j < N; j++) { f[j] = tomont(f[j]); fi[j] = tomont(fi[j]); }
    for (size_t j = 0; j < N / 32; j++) { m[j] = f[j * 32]; mi[j] = fi[j * 32]; }
    u64 qi = 1; for (int k = 0; k < 7; k++) qi *= 2 - P * qi;
    std::vector<u64> pr(maxb, P), nv(maxb, tomont(powm(N % P, P - 2))); std::vector<i64> qv(maxb, (i64) qi);
    auto up = [](u64 **d, const void *h, size_t bytes) { ck(cudaMalloc(d, bytes), "tbl"); cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice); };
    up(&tw, f.data(), N * 8); up(&itw, fi.data(), N * 8); up(&twm, m.data(), N / 32 * 8); up(&itwm, mi.data(), N / 32 * 8);
    up(&primes, pr.data(), maxb * 8); up(&ninv, nv.data(), maxb * 8); up((u64 **) &qinv, qv.data(), maxb * 8);
  }
  ~Tables() { cudaFree(tw); cudaFree(twm); cudaFree(itw); cudaFree(itwm); cudaFree(primes); cudaFree(ninv); cudaFree(qinv); }
};
template <class K> static bool fits(K k, int th, int sm, int maxs) {   // can this kernel launch with th threads and sm bytes?
  if (sm > maxs) return false;
  ck(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm), "smem attr");
  cudaFuncAttributes fa; ck(cudaFuncGetAttributes(&fa, k), "func attr");
  return th <= fa.maxThreadsPerBlock;
}
template <int L, int P2, int M2> struct Rows {   // Cheddar's contiguous row kernels (last P2 stages), 2^(P2-M2) threads
  using C = CCfg<P2, M2, 0>; static constexpr int th = 1 << (P2 - M2), sm = th * (1 << M2) * 8;
  static bool setup(int maxs) { return fits(cheddar::kernel::NTTPhase2<u64, L, C>, th, sm, maxs) && fits(cheddar::kernel::INTTPhase1<u64, L, C>, th, sm, maxs); }
  static void fwd(const Tables &t, u64 *x, int batch) { const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase2<u64, L, C><<<dim3(N / (1 << M2) / th, batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.tw, t.twm, 0, batch, 0, 0, s); }
  static void inv(const Tables &t, u64 *x, int batch) { const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::INTTPhase1<u64, L, C><<<dim3(N / (1 << M2) / th, batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.itw, t.itwm, 0, batch, s); }
};
template <int L, int P1, int M1, int B1> struct ColsContig {   // column stages on transposed data: 2^B1 contiguous column vectors per block
  using C = CCfg<P1, M1, B1>; static constexpr int th = 1 << (P1 - M1 + B1), sm = th * (1 << M1) * 8;
  static bool setup(int maxs) { return fits(cheddar::kernel::NTTPhase1Contig<u64, L, C>, th, sm, maxs) &&
                                       fits(cheddar::kernel::INTTPhase2Contig<u64, L, C, cheddar::kernel::MultConst<u64>>, th, sm, maxs); }
  static void fwd(const Tables &t, u64 *x, int batch) { cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase1Contig<u64, L, C><<<dim3(1u << (L - P1 - B1), batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.tw, 0, batch, 0, 0, s, cheddar::InputPtrList<u64, 1>()); }
  static void inv(const Tables &t, u64 *x, int batch) { cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x; cheddar::InputPtrList<u64, 1> c; c.ptrs_[0] = t.ninv;
    cheddar::kernel::INTTPhase2Contig<u64, L, C, cheddar::kernel::MultConst<u64>><<<dim3(1u << (L - P1 - B1), batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.itw, 0, batch, s, c); }
};
template <int L, int P1, int M1, int B1> struct Cols {   // Cheddar's own strided column kernels (first P1 stages)
  using C = CCfg<P1, M1, B1>; static constexpr int th = 1 << (P1 - M1 + B1), sm = th * (1 << M1) * 8;
  static bool setup(int maxs) { return fits(cheddar::kernel::NTTPhase1<u64, L, C>, th, sm, maxs) &&
                                       fits(cheddar::kernel::INTTPhase2<u64, L, C, cheddar::kernel::MultConst<u64>>, th, sm, maxs); }
  static void fwd(const Tables &t, u64 *x, int batch) { const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase1<u64, L, C><<<dim3(N / (1 << M1) / th, batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.tw, 0, batch, 0, 0, s, cheddar::InputPtrList<u64, 1>()); }
  static void inv(const Tables &t, u64 *x, int batch) { const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x; cheddar::InputPtrList<u64, 1> c; c.ptrs_[0] = t.ninv;
    cheddar::kernel::INTTPhase2<u64, L, C, cheddar::kernel::MultConst<u64>><<<dim3(N / (1 << M1) / th, batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.itw, 0, batch, s, c); }
};
template <int L, int P2, int M2> struct RowsAcc {   // Cheddar's row kernel with the PCG multiply-accumulate epilogue (extract_acc.py)
  using C = CCfg<P2, M2, 0>; static constexpr int th = 1 << (P2 - M2), sm = th * (1 << M2) * 8;
  static bool setup(int maxs) { return fits(cheddar::kernel::NTTPhase2Acc<u64, L, C>, th, sm, maxs); }
  static int groups(int items) {   // partial-sum groups: enough blocks (2^P1 rows x G) to fill the GPU, at most one item per block, G | items
    int G = 1; while ((1 << (L - P2)) * G < 1024 && G < items) G *= 2;
    while (items % G) G--; return G; }
  static void fwd(const Tables &t, const u64 *g, const u64 *ahat, int c, int items, u64 *partials, int G, Modulus<Data64> mod, bool pairs = false, const u64 *bhat = nullptr) {
    cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) g;
    cheddar::kernel::NTTPhase2Acc<u64, L, C><<<dim3(1u << (L - P2), G), th, sm>>>(nullptr, t.primes, t.qinv, t.tw, t.twm, 0, items, 0, 0, s, ahat, c, items / G, partials, mod, pairs, bhat); }
};
template <int L, int P1, int M1, int B1> struct ColsContigIn {   // Cheddar's column kernel reading the SPU's column-contiguous layout, storing Cheddar's natural layout
  using C = CCfg<P1, M1, B1>; static constexpr int th = 1 << (P1 - M1 + B1), sm = th * (1 << M1) * 8;
  static bool setup(int maxs) { return fits(cheddar::kernel::NTTPhase1ContigIn<u64, L, C>, th, sm, maxs); }
  // out of place: the layout changes, so a block's stores are not the addresses it loaded (in place would race with other blocks)
  static void fwd(const Tables &t, const u64 *x, u64 *y, int batch) { cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase1ContigIn<u64, L, C><<<dim3(1u << (L - P1 - B1), batch), th, sm>>>((i64 *) y, t.primes, t.qinv, t.tw, 0, batch, 0, 0, s, cheddar::InputPtrList<u64, 1>()); }
};
template <int L, int P1, int M1, int B1> struct ColsRegIn {   // Cheddar's column kernel reading its own register order (the SPUs write g_ij in it), storing Cheddar's natural layout
  using C = CCfg<P1, M1, B1>; static constexpr int th = 1 << (P1 - M1 + B1), sm = th * (1 << M1) * 8;
  static bool setup(int maxs) { return fits(cheddar::kernel::NTTPhase1RegIn<u64, L, C>, th, sm, maxs); }
  static void fwd(const Tables &t, const u64 *x, u64 *y, int batch) { cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase1RegIn<u64, L, C><<<dim3(1u << (L - P1 - B1), batch), th, sm>>>((i64 *) y, t.primes, t.qinv, t.tw, 0, batch, 0, 0, s, cheddar::InputPtrList<u64, 1>()); }
};
template <typename F> static double time_ms(F &&fn, int iters) {
  fn(); ck(cudaDeviceSynchronize(), "warm"); cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s); for (int i = 0; i < iters; ++i) fn(); cudaEventRecord(e); ck(cudaEventSynchronize(e), "time");
  float ms; cudaEventElapsedTime(&ms, s, e); cudaEventDestroy(s); cudaEventDestroy(e); return ms / iters;
}
struct Ctx { int batch, iters, maxs; size_t tot; u64 *a, *b, *r, *tmp, *zr = nullptr, *zrp = nullptr, *gn = nullptr, *bh = nullptr; unsigned long long *cnt; MergePlan *mp; Tables *t; std::function<void()> refill, refill_ole; };
static unsigned long long diff(Ctx &c) { cudaMemset(c.cnt, 0, 8); count_diff<<<4096, 256>>>(c.a, c.r, c.tot, c.cnt); unsigned long long v; cudaMemcpy(&v, c.cnt, 8, cudaMemcpyDeviceToHost); return v; }
static bool launched(const char *what, int P1, int P2, int M1, int B1, int M2) {   // the check run's launches all succeeded?
  cudaError_t e = cudaDeviceSynchronize(); if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "skip %s P1 %d P2 %d M1 %d B1 %d M2 %d: %s\n", what, P1, P2, M1, B1, M2, cudaGetErrorString(e)); cudaGetLastError(); return false; }
  return true;
}
template <int L, int P2, int M1, int B1, int M2> static void run_cheddar(Ctx &c) {
  constexpr int P1 = L - P2;
  if constexpr (P1 < M1 || P2 < M2 || !thr_ok(P1 - M1 + B1) || !thr_ok(P2 - M2)) return; else {
    if (ext_skip(M1, B1, M2)) return;
    using CC = Cols<L, P1, M1, B1>; using RC = Rows<L, P2, M2>;
    if (!CC::setup(c.maxs) || !RC::setup(c.maxs)) { fprintf(stderr, "skip cheddar P1 %d P2 %d M1 %d B1 %d M2 %d: cannot launch (threads x registers or shared memory)\n", P1, P2, M1, B1, M2); return; }
    const Tables &t = *c.t; const int batch = c.batch;
    auto mul = [&] { CC::fwd(t, c.a, batch); RC::fwd(t, c.a, batch); CC::fwd(t, c.b, batch); RC::fwd(t, c.b, batch);
      fused4::pointwise_kernel<<<static_cast<unsigned>((c.tot + 255) / 256), 256>>>(c.a, c.b, c.mp->modulus, c.tot); RC::inv(t, c.a, batch); CC::inv(t, c.a, batch); };
    if (wantd("cheddar")) {
      c.refill(); mul(); if (!launched("cheddar", P1, P2, M1, B1, M2)) return; const unsigned long long bad = diff(c);
      printf("%d,%d,cheddar,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, batch, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(mul, c.iters) / batch, bad); fflush(stdout);
    }
    if (wantd("cheddar_t")) {   // cheddar_t: the same multiply inside PCG^2 - the g operand (b) is read by the column kernel directly in the
      // column-contiguous layout the SPUs write (ColsContigIn, out of place into tmp); everything else is Cheddar's pipeline unchanged
      using CI = ColsContigIn<L, P1, M1, B1>;
      if (!CI::setup(c.maxs)) { fprintf(stderr, "skip cheddar_t P1 %d P2 %d M1 %d B1 %d M2 %d: cannot launch\n", P1, P2, M1, B1, M2); return; }
      const size_t N = 1ull << L;
      auto mul_t = [&] { CC::fwd(t, c.a, batch); RC::fwd(t, c.a, batch); CI::fwd(t, c.b, c.tmp, batch); RC::fwd(t, c.tmp, batch);
        fused4::pointwise_kernel<<<static_cast<unsigned>((c.tot + 255) / 256), 256>>>(c.a, c.tmp, c.mp->modulus, c.tot); RC::inv(t, c.a, batch); CC::inv(t, c.a, batch); };
      c.refill();   // the SPUs' layout for b, prepared outside the timed region
      for (int k = 0; k < batch; k++) { T(c.b + k * N, c.tmp, 1 << P1, 1 << P2, 1); cudaMemcpyAsync(c.b + k * N, c.tmp, N * 8, cudaMemcpyDeviceToDevice, 0); }
      mul_t(); if (!launched("cheddar_t", P1, P2, M1, B1, M2)) return; const unsigned long long badt = diff(c);
      printf("%d,%d,cheddar_t,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, batch, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(mul_t, c.iters) / batch, badt); fflush(stdout);
    }
  }
}
template <int L, int P1, int M1, int B1, int M2> static void run_f4c(Ctx &c) {
  constexpr int P2 = L - P1;
  if constexpr (P1 < M1 || P2 < M2 || P2 < B1 || !thr_ok(P1 - M1 + B1) || !thr_ok(P2 - M2)) return; else {
    using RC = Rows<L, P2, M2>; using CC = ColsContig<L, P1, M1, B1>;
    if (!CC::setup(c.maxs) || !RC::setup(c.maxs)) { fprintf(stderr, "skip f4c P1 %d P2 %d M1 %d B1 %d M2 %d: cannot launch (threads x registers or shared memory)\n", P1, P2, M1, B1, M2); return; }
    const int R = 1 << P1, Cn = 1 << P2, batch = c.batch; const Tables &t = *c.t;
    auto pw = [&] { fused4::pointwise_kernel<<<static_cast<unsigned>((c.tot + 255) / 256), 256>>>(c.a, c.b, c.mp->modulus, c.tot); };
    auto full = [&] {   // transposes on the SMs (functional reference of the DRU dataflow)
      for (u64 *x : {c.a, c.b}) { T(x, c.tmp, R, Cn, batch); CC::fwd(t, c.tmp, batch); T(c.tmp, x, Cn, R, batch); RC::fwd(t, x, batch); }
      pw(); RC::inv(t, c.a, batch); T(c.a, c.tmp, R, Cn, batch); CC::inv(t, c.tmp, batch); T(c.tmp, c.a, Cn, R, batch);
    };
    auto sm = [&] {     // SM lane: the same kernels, transposes left to the DRU
      CC::fwd(t, c.a, batch); RC::fwd(t, c.a, batch); CC::fwd(t, c.b, batch); RC::fwd(t, c.b, batch); pw(); RC::inv(t, c.a, batch); CC::inv(t, c.a, batch);
    };
    c.refill(); full(); if (!launched("f4c", P1, P2, M1, B1, M2)) return; const unsigned long long bad = diff(c);
    const double tf = time_ms(full, c.iters) / batch, ts = time_ms(sm, c.iters) / batch;
    printf("%d,%d,f4c_full,%d,%d,%d,%d,%d,%d,%.4f,%llu\n%d,%d,f4c_sm,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, batch, P1, P2, M1, B1, M2, BIGN_MAXRREG, tf, bad, L, batch, P1, P2, M1, B1, M2, BIGN_MAXRREG, ts, bad); fflush(stdout);
  }
}
static unsigned long long diff_z(Ctx &c) {   // the PCG output z (one polynomial, in c.tmp) against its reference c.zr
  const size_t N = 1ull << BIGN_LOGN; cudaMemset(c.cnt, 0, 8); count_diff<<<4096, 256>>>(c.tmp, c.zr, N, c.cnt); unsigned long long v; cudaMemcpy(&v, c.cnt, 8, cudaMemcpyDeviceToHost); return v; }
// PCG output on Cheddar's kernels: z = INTT( sum_ij a^_i . a^_j . NTT(g_ij) ) for c^2 = batch polynomials g_ij and c public
// polynomials a_i (a_0 = 1).  Per output: c forward transforms of a (Cols + Rows), one column pass over all g_ij, the row
// pass with the multiply-accumulate epilogue (RowsAcc, G partial sums), the partial-sum reduction, one inverse transform.
// ole = g_ij read in Cheddar's natural layout (strided columns); ole_t = g_ij read in the column-contiguous layout the SPUs
// write (ColsContigIn), stores unchanged.  ms_per_poly = time per output / c^2 (per g_ij; the a transforms and the inverse
// are amortised over the c^2 polynomials).  Reference: z computed with merge one product at a time (c.zr).
template <int L, int P2, int M1, int B1, int M2> static void run_ole(Ctx &c) {
  constexpr int P1 = L - P2;
  if constexpr (P1 < M1 || P2 < M2 || !thr_ok(P1 - M1 + B1) || !thr_ok(P2 - M2)) return; else {
    using CC = Cols<L, P1, M1, B1>; using CI = ColsContigIn<L, P1, M1, B1>; using CR = ColsRegIn<L, P1, M1, B1>; using RC = Rows<L, P2, M2>; using RA = RowsAcc<L, P2, M2>;
    if (!CC::setup(c.maxs) || !RC::setup(c.maxs) || !RA::setup(c.maxs)) { fprintf(stderr, "skip ole P1 %d P2 %d M1 %d B1 %d M2 %d: cannot launch (threads x registers or shared memory)\n", P1, P2, M1, B1, M2); return; }
    const bool ci_ok = CI::setup(c.maxs), cr_ok = CR::setup(c.maxs);
    const int c2 = c.batch; int cc = 1; while (cc * cc < c2) cc++; if (cc * cc != c2) return;
    if (ext_skip(M1, B1, M2)) return;
    const int G = RA::groups(c2); const size_t N = 1ull << L; const Tables &t = *c.t; const Modulus<Data64> mod = c.mp->modulus;
    u64 *g = c.a, *ahat = c.b, *z = c.tmp, *gn = c.gn;   // g_ij: batch polynomials; a_i: the first c polynomials of b (a_0 = 1); z partials: G polynomials of tmp; gn: natural-layout columns of the tiled g
    auto tail = [&](const u64 *gs, int items, int Gi, bool pairs, const u64 *bh = nullptr) { RA::fwd(t, gs, ahat, cc, items, z, Gi, mod, pairs, bh); if (Gi > 1) reduce_partials<<<4096, 256>>>(z, Gi, N, mod); RC::inv(t, z, 1); CC::inv(t, z, 1); };
    auto ole = [&] { CC::fwd(t, ahat, cc); RC::fwd(t, ahat, cc); CC::fwd(t, g, c2); tail(g, c2, G, false); };
    auto ole_t = [&] { CC::fwd(t, ahat, cc); RC::fwd(t, ahat, cc); CI::fwd(t, g, gn, c2); tail(gn, c2, G, false); };
    auto ole_r = [&] { CC::fwd(t, ahat, cc); RC::fwd(t, ahat, cc); CR::fwd(t, g, gn, c2); tail(gn, c2, G, false); };   // ole_r: g_ij in the column kernel's register order
    // olep: the SPUs accumulated g_ij + g_ji (i < j) into one buffer, so c(c+1)/2 unordered pairs instead of c^2 polynomials
    const int np = cc * (cc + 1) / 2, Gp = RA::groups(np);
    auto olep = [&] { CC::fwd(t, ahat, cc); RC::fwd(t, ahat, cc); CC::fwd(t, g, np); tail(g, np, Gp, true); };
    if (wantd("ole")) {
      c.refill_ole(); ole(); if (!launched("ole", P1, P2, M1, B1, M2)) return; const unsigned long long bad = diff_z(c);
      printf("%d,%d,ole,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, c2, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(ole, c.iters) / c2, bad); fflush(stdout);
    }
    if (c.zrp) {   // paired reference available
      if (wantd("olep")) {
        c.refill_ole(); olep(); if (!launched("olep", P1, P2, M1, B1, M2)) return;
        cudaMemset(c.cnt, 0, 8); count_diff<<<4096, 256>>>(c.tmp, c.zrp, N, c.cnt); unsigned long long badp; cudaMemcpy(&badp, c.cnt, 8, cudaMemcpyDeviceToHost);
        printf("%d,%d,olep,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, c2, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(olep, c.iters) / c2, badp); fflush(stdout);
      }
      if (c.bh) {   // olepb: b^_ij = a^_i . a^_j precomputed once per public a (Cheddar Seq.2: constant products folded ahead of the kernel)
        // the a transforms are part of that precomputation, so olepb runs only the column pass, the accumulate pass and the inverse
        auto prep = [&] { CC::fwd(t, ahat, cc); RC::fwd(t, ahat, cc); int k = 0;
          for (int i = 0; i < cc; i++) for (int j = i; j < cc; j++, k++) { cudaMemcpyAsync(c.bh + k * N, ahat + i * N, N * 8, cudaMemcpyDeviceToDevice, 0);
            fused4::pointwise_kernel<<<static_cast<unsigned>((N + 255) / 256), 256>>>(c.bh + k * N, ahat + j * N, mod, N); } };
        auto olepb = [&] { CC::fwd(t, g, np); tail(g, np, Gp, true, c.bh); };
        c.refill_ole(); prep(); olepb(); if (!launched("olepb", P1, P2, M1, B1, M2)) return;
        cudaMemset(c.cnt, 0, 8); count_diff<<<4096, 256>>>(c.tmp, c.zrp, N, c.cnt); unsigned long long badb; cudaMemcpy(&badb, c.cnt, 8, cudaMemcpyDeviceToHost);
        c.refill_ole(); prep(); ck(cudaDeviceSynchronize(), "olepb prep");
        printf("%d,%d,olepb,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, c2, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(olepb, c.iters) / c2, badb); fflush(stdout);
      }
    }
    if (wantd("ole")) {
      if (!ci_ok) { fprintf(stderr, "skip ole_t P1 %d P2 %d M1 %d B1 %d M2 %d: cannot launch\n", P1, P2, M1, B1, M2); return; }
      c.refill_ole();   // the SPUs' layout: every g_ij column-contiguous (transposed R x C -> C x R), prepared outside the timed region
      for (int k = 0; k < c2; k++) { T(g + k * N, z, 1 << P1, 1 << P2, 1); cudaMemcpyAsync(g + k * N, z, N * 8, cudaMemcpyDeviceToDevice, 0); }
      ole_t(); if (!launched("ole_t", P1, P2, M1, B1, M2)) return; const unsigned long long badt = diff_z(c);
      printf("%d,%d,ole_t,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, c2, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(ole_t, c.iters) / c2, badt); fflush(stdout);
      if (wantd("oleu")) {   // oleu / olepu: the SM lane of the channel-unit design (the per-channel unit does the row pass, the a^ products
        // and the accumulate in-stream, rows placed whole in a channel by the SPUs): the column pass over the g batch (SPU layout, c^2 or
        // c(c+1)/2 paired items) + the inverse column pass once per output. Same kernels as ole_t; mismatch count = ole_t's in this config.
        auto oleu = [&] { CI::fwd(t, g, gn, c2); CC::inv(t, z, 1); }; auto olepu = [&] { CI::fwd(t, g, gn, np); CC::inv(t, z, 1); };
        printf("%d,%d,oleu,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, c2, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(oleu, c.iters) / c2, badt); fflush(stdout);
        printf("%d,%d,olepu,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, c2, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(olepu, c.iters) / c2, badt); fflush(stdout);
      }
    }
    if (wantd("ole_r")) {
      if (!cr_ok) { fprintf(stderr, "skip ole_r P1 %d P2 %d M1 %d B1 %d M2 %d: cannot launch\n", P1, P2, M1, B1, M2); return; }
      c.refill_ole();   // the SPUs' layout: every g_ij in the column kernel's register order (column-contiguous, then per-column s/i transpose), outside the timed region
      for (int k = 0; k < c2; k++) { T(g + k * N, z, 1 << P1, 1 << P2, 1); perm_reg<<<4096, 256>>>(z, g + k * N, P1, M1, N); }
      ole_r(); if (!launched("ole_r", P1, P2, M1, B1, M2)) return; const unsigned long long badr = diff_z(c);
      printf("%d,%d,ole_r,%d,%d,%d,%d,%d,%d,%.4f,%llu\n", L, c2, P1, P2, M1, B1, M2, BIGN_MAXRREG, time_ms(ole_r, c.iters) / c2, badr); fflush(stdout);
    }
  }
}
#ifndef BIGN_NO_MAIN   // cheddar_interfere.cu reuses everything above
int main(int argc, char **argv) {
  constexpr int L = BIGN_LOGN; std::vector<int> batches = {4, 16, 64}; int iters = L >= 27 ? 3 : 5; std::string designs = "merge,cheddar,f4c";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) { batches.clear(); for (char *p = argv[++i]; *p;) { batches.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; } }
    else if (!strcmp(argv[i], "--designs") && i + 1 < argc) designs = argv[++i];   // merge,cheddar,f4c (per multiply) and/or ole (PCG output)
    else if (!strcmp(argv[i], "--ext")) EXT_ONLY = true;
  }
  DESIGNS = designs; auto want = [&](const char *v) { return wantd(v); };
  int dev, maxs; cudaGetDevice(&dev); cudaDeviceGetAttribute(&maxs, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev); fprintf(stderr, "%s, L2 %d B, logN %d, maxrregcount %d\n", p.name, p.l2CacheSize, L, BIGN_MAXRREG);
  printf("logN,batch,design,P1,P2,M1,B1,M2,maxrreg,ms_per_poly,mismatches\n"); fflush(stdout);
  MergePlan mp(L); int maxb = 1; for (int b : batches) maxb = b > maxb ? b : maxb;
  Tables t(L, maxb);
  for (int batch : batches) {
    Ctx c; c.batch = batch; c.iters = iters; c.maxs = maxs; c.tot = (1ull << L) * batch; c.mp = &mp; c.t = &t; c.a = c.b = c.r = c.tmp = nullptr;
    auto release = [&] { cudaFree(c.a); cudaFree(c.b); cudaFree(c.r); cudaFree(c.tmp); c.a = c.b = c.r = c.tmp = nullptr; };
    if (cudaMalloc(&c.a, c.tot * 8) || cudaMalloc(&c.b, c.tot * 8) || cudaMalloc(&c.r, c.tot * 8) || cudaMalloc(&c.tmp, c.tot * 8)) {
      printf("%d,%d,OOM,0,0,0,0,0,%d,0,0\n", L, batch, BIGN_MAXRREG); fflush(stdout); cudaGetLastError(); release(); continue; }
    ck(cudaMalloc(&c.cnt, 8), "cnt");
    c.refill = [&] { fill<<<4096, 256>>>(c.a, c.tot, 111 + L * 7 + batch); fill<<<4096, 256>>>(c.b, c.tot, 999 + L * 13 + batch); };
    const size_t N = 1ull << L;
    if (want("merge") || want("cheddar") || want("cheddar_t") || want("f4c")) {
      // reference: merge one polynomial at a time
      c.refill();
      for (int i = 0; i < batch; i++) mp.multiply(c.a + i * N, c.b + i * N, 1);
      ck(cudaDeviceSynchronize(), "ref"); ck(cudaMemcpy(c.r, c.a, c.tot * 8, cudaMemcpyDeviceToDevice), "ref copy");
      c.refill(); mp.multiply(c.a, c.b, batch); ck(cudaDeviceSynchronize(), "merge batched"); const unsigned long long mbad = diff(c);
      if (want("merge")) { printf("%d,%d,merge,0,0,0,0,0,%d,%.4f,%llu\n", L, batch, BIGN_MAXRREG, time_ms([&] { mp.multiply(c.a, c.b, batch); }, iters) / batch, mbad); fflush(stdout); }
    }
    // Cheddar: P2 in [L-14, 14], M1, B1, M2
    // M1, M2 in 3..6 (3..5 in the original sweep; 6 = 64 elements per thread), B1 in 0..4 (0..3 originally)
    if (want("cheddar") || want("cheddar_t")) cfor<9, 15>([&](auto P2) { if constexpr (L - P2 >= 7 && L - P2 <= 14) cfor<3, 7>([&](auto M1) { cfor<0, 5>([&](auto B1) { cfor<3, 7>([&](auto M2) {
      run_cheddar<L, P2, M1, B1, M2>(c); }); }); }); });
    // four-step: P1 in [L-14, 14], M1, M2
    if (want("f4c")) cfor<9, 15>([&](auto P1) { if constexpr (L - P1 >= 7 && L - P1 <= 14) cfor<3, 6>([&](auto M1) { cfor<0, 4>([&](auto B1) { cfor<3, 6>([&](auto M2) { run_f4c<L, P1, M1, B1, M2>(c); }); }); }); });
    if (want("ole") || want("ole_r") || want("olep") || want("olepb")) {   // PCG output z over c^2 = batch polynomials g_ij (c.a) and c public a_i (first c polynomials of c.b, a_0 = 1)
      int cc = 1; while (cc * cc < batch) cc++;
      if (cc * cc == batch) {
        ck(cudaMalloc(&c.zr, N * 8), "zr");
        const bool ref_busy = want("merge") || want("cheddar") || want("cheddar_t") || want("f4c");   // c.r holds the per-multiply reference: ole_t's natural-layout output needs its own buffer
        if (ref_busy) { if (cudaMalloc(&c.gn, c.tot * 8)) { printf("%d,%d,OOM,0,0,0,0,0,%d,0,0\n", L, batch, BIGN_MAXRREG); fflush(stdout); cudaGetLastError(); c.gn = nullptr; } } else c.gn = c.r;
        c.refill_ole = [&, cc] { c.refill(); set_one<<<4096, 256>>>(c.b, N); };
        // reference: z = sum_ij (a_i a_j) g_ij with merge, one product at a time (tmp[0], tmp[1] scratch; batch >= 4 polynomials)
        c.refill_ole(); cudaMemset(c.zr, 0, N * 8);
        for (int i = 0; i < cc; i++) for (int j = 0; j < cc; j++) {
          cudaMemcpy(c.tmp, c.b + i * N, N * 8, cudaMemcpyDeviceToDevice); cudaMemcpy(c.tmp + N, c.b + j * N, N * 8, cudaMemcpyDeviceToDevice);
          mp.multiply(c.tmp, c.tmp + N, 1);
          cudaMemcpy(c.tmp + N, c.a + (i * cc + j) * N, N * 8, cudaMemcpyDeviceToDevice); mp.multiply(c.tmp, c.tmp + N, 1);
          addmod_kernel<<<4096, 256>>>(c.zr, c.tmp, N, mp.modulus);
        }
        ck(cudaDeviceSynchronize(), "ole ref");
        if (want("olep") || want("olepb")) {   // paired reference: z = sum_{i<=j} (a_i a_j) g_k, k enumerating the pairs as the kernel does
          ck(cudaMalloc(&c.zrp, N * 8), "zrp"); c.refill_ole(); cudaMemset(c.zrp, 0, N * 8); int k = 0;
          for (int i = 0; i < cc; i++) for (int j = i; j < cc; j++, k++) {
            cudaMemcpy(c.tmp, c.b + i * N, N * 8, cudaMemcpyDeviceToDevice); cudaMemcpy(c.tmp + N, c.b + j * N, N * 8, cudaMemcpyDeviceToDevice);
            mp.multiply(c.tmp, c.tmp + N, 1);
            cudaMemcpy(c.tmp + N, c.a + k * N, N * 8, cudaMemcpyDeviceToDevice); mp.multiply(c.tmp, c.tmp + N, 1);
            addmod_kernel<<<4096, 256>>>(c.zrp, c.tmp, N, mp.modulus);
          }
          ck(cudaDeviceSynchronize(), "olep ref");
          if (want("olepb") && cudaMalloc(&c.bh, (size_t) (cc * (cc + 1) / 2) * N * 8)) { fprintf(stderr, "olepb: b^ does not fit, skipped\n"); cudaGetLastError(); c.bh = nullptr; }
        }
        if (c.gn) cfor<9, 15>([&](auto P2) { if constexpr (L - P2 >= 7 && L - P2 <= 14) cfor<3, 7>([&](auto M1) { cfor<0, 5>([&](auto B1) { cfor<3, 7>([&](auto M2) {
          run_ole<L, P2, M1, B1, M2>(c); }); }); }); });
        if (ref_busy) cudaFree(c.gn);
        cudaFree(c.zr); cudaFree(c.zrp); cudaFree(c.bh); c.zr = c.zrp = c.gn = c.bh = nullptr;
      } else fprintf(stderr, "ole: batch %d is not a square c^2, skipped\n", batch);
    }
    release(); cudaFree(c.cnt);
  }
  return 0;
}
#endif
