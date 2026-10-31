// Four-step NTT on Cheddar's kernels with the transposes on the DRU, vs Cheddar's two-kernel NTT and GPU-NTT merge.
// N = R x C, R = 2^P1, C = 2^P2. Forward: T(R x C -> C x R) | column stages: Cheddar's column kernel with contiguous addressing
// (gen/cheddar_ntt_contig.cuh) | T back | row stages: Cheddar's NTTPhase2. Inverse: Cheddar's INTTPhase1 | T | contiguous
// INTTPhase2 (with 1/N) | T back. 6 transposes per multiply, as in the paper's DRU design.
//   f4c_sm   SM lane: the SM kernels + pointwise only (transposes on the DRU, fully overlapped; same definition as f4g_sm)
//   f4c_full the same with the transposes executed on the SMs (32x33 shared-memory tile transpose)
//   cheddar  Cheddar's own two-kernel NTT, the 3 best configs of the sweep (gen/cfgs.h)
// Correctness: the f4c_full product and every Cheddar product are compared element-wise with the GPU-NTT merge product.
// Output: logN,batch,design,P1,P2,M1,B1,ms_per_poly,mismatches
#include "fused4/fused4_ntt.cuh"   // -I.. (GPU_baseline)
#include "gpuntt/common/nttparameters.cuh"
#include "gen/cheddar_ntt_contig.cuh"
#include "gen/cfgs.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <utility>
#include <functional>
using namespace gpuntt;
using u64 = uint64_t; using i64 = int64_t; using u128 = unsigned __int128;
static const u64 P = 4611686018326724609ULL;   // 2^62 - 3*2^25 + 1
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
__global__ void transpose(const u64 *__restrict__ src, u64 *__restrict__ dst, int R, int C) {   // per limb (blockIdx.z): R x C -> C x R
  __shared__ u64 t[32][33];
  const size_t off = (size_t) blockIdx.z * R * C; src += off; dst += off;
  int c = blockIdx.x * 32 + threadIdx.x, r = blockIdx.y * 32 + threadIdx.y;
  for (int k = 0; k < 32; k += 8) t[threadIdx.y + k][threadIdx.x] = src[(size_t) (r + k) * C + c];
  __syncthreads();
  r = blockIdx.x * 32 + threadIdx.y; c = blockIdx.y * 32 + threadIdx.x;
  for (int k = 0; k < 32; k += 8) dst[(size_t) (r + k) * R + c] = t[threadIdx.x][threadIdx.y + k];
}
static void T(const u64 *s, u64 *d, int R, int C, int batch) { transpose<<<dim3(C / 32, R / 32, batch), dim3(32, 8)>>>(s, d, R, C); }
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
struct Tables {   // NTT.cu PopulateTwiddleFactors for one prime
  int L; u64 *tw, *twm, *itw, *itwm, *primes, *ninv; i64 *qinv;
  Tables(int L_, int batch) : L(L_) {
    const size_t N = 1ull << L; u64 g = 3; while (powm(g, (P - 1) / 2) == 1) g++;
    u64 psi = powm(g, (P - 1) / (2 * N)), psii = powm(psi, P - 2);
    std::vector<u64> f(N), fi(N), fm(N), fim(N), m(N / 32), mi(N / 32), a(N), ai(N);
    a[0] = ai[0] = 1; for (size_t j = 1; j < N; j++) { a[j] = mulm(a[j - 1], psi); ai[j] = mulm(ai[j - 1], psii); }
    for (size_t j = 0; j < N; j++) { size_t r = 0; for (int k = 0; k < L; k++) r |= ((j >> k) & 1) << (L - 1 - k); f[r] = a[j]; fi[r] = ai[j]; }
    for (size_t j = 0; j < N; j++) { fm[j] = tomont(f[j]); fim[j] = tomont(fi[j]); }
    for (size_t j = 0; j < N / 32; j++) { m[j] = fm[j * 32]; mi[j] = fim[j * 32]; }
    u64 qi = 1; for (int k = 0; k < 7; k++) qi *= 2 - P * qi;
    std::vector<u64> pr(batch, P), nv(batch, tomont(powm(N % P, P - 2))); std::vector<i64> qv(batch, (i64) qi);
    auto up = [](u64 **d, const void *h, size_t bytes) { ck(cudaMalloc(d, bytes), "tbl"); cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice); };
    up(&tw, fm.data(), N * 8); up(&itw, fim.data(), N * 8); up(&twm, m.data(), N / 32 * 8); up(&itwm, mi.data(), N / 32 * 8);
    up(&primes, pr.data(), batch * 8); up(&ninv, nv.data(), batch * 8); up((u64 **) &qinv, qv.data(), batch * 8);
  }
  ~Tables() { cudaFree(tw); cudaFree(twm); cudaFree(itw); cudaFree(itwm); cudaFree(primes); cudaFree(ninv); cudaFree(qinv); }
};
template <int L, int P2, int M2> struct Rows {   // Cheddar's contiguous row kernels (last P2 stages), 2^(P2-M2) threads
  using C = CCfg<P2, M2, 0>; static constexpr int th = 1 << (P2 - M2), sm = th * (1 << M2) * 8;
  static void setup() {
    ck(cudaFuncSetAttribute(cheddar::kernel::NTTPhase2<u64, L, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm), "r1");
    ck(cudaFuncSetAttribute(cheddar::kernel::INTTPhase1<u64, L, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm), "r2");
  }
  static void fwd(const Tables &t, u64 *x, int batch) { const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase2<u64, L, C><<<dim3(N / (1 << M2) / th, batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.tw, t.twm, 0, batch, 0, 0, s); }
  static void inv(const Tables &t, u64 *x, int batch) { const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::INTTPhase1<u64, L, C><<<dim3(N / (1 << M2) / th, batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.itw, t.itwm, 0, batch, s); }
};
template <int L, int P1, int M1> struct ColsContig {   // column stages on transposed data: one block per contiguous column vector
  using C = CCfg<P1, M1, 0>; static constexpr int th = 1 << (P1 - M1), sm = th * (1 << M1) * 8;
  static void setup() {
    ck(cudaFuncSetAttribute(cheddar::kernel::NTTPhase1Contig<u64, L, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm), "c1");
    ck(cudaFuncSetAttribute(cheddar::kernel::INTTPhase2Contig<u64, L, C, cheddar::kernel::MultConst<u64>>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm), "c2");
  }
  static void fwd(const Tables &t, u64 *x, int batch) { cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase1Contig<u64, L, C><<<dim3(1u << (L - P1), batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.tw, 0, batch, 0, 0, s, cheddar::InputPtrList<u64, 1>()); }
  static void inv(const Tables &t, u64 *x, int batch) { cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x; cheddar::InputPtrList<u64, 1> c; c.ptrs_[0] = t.ninv;
    cheddar::kernel::INTTPhase2Contig<u64, L, C, cheddar::kernel::MultConst<u64>><<<dim3(1u << (L - P1), batch), th, sm>>>((i64 *) x, t.primes, t.qinv, t.itw, 0, batch, s, c); }
};
template <int L, int P2, int M1, int B1> struct CheddarMul {   // Cheddar's own two-kernel NTT (strided column kernel)
  static constexpr int P1 = L - P2, M2 = 3; using CC = CCfg<P1, M1, B1>; using CR = CCfg<P2, M2, 0>;
  static constexpr int thc = 1 << (P1 - M1 + B1), thr = 1 << (P2 - M2), smc = thc * (1 << M1) * 8, smr = thr * (1 << M2) * 8;
  static void setup() {
    ck(cudaFuncSetAttribute(cheddar::kernel::NTTPhase1<u64, L, CC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smc), "a1");
    ck(cudaFuncSetAttribute(cheddar::kernel::NTTPhase2<u64, L, CR>, cudaFuncAttributeMaxDynamicSharedMemorySize, smr), "a2");
    ck(cudaFuncSetAttribute(cheddar::kernel::INTTPhase1<u64, L, CR>, cudaFuncAttributeMaxDynamicSharedMemorySize, smr), "a3");
    ck(cudaFuncSetAttribute(cheddar::kernel::INTTPhase2<u64, L, CC, cheddar::kernel::MultConst<u64>>, cudaFuncAttributeMaxDynamicSharedMemorySize, smc), "a4");
  }
  static void fwd(const Tables &t, u64 *x, int batch) { const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase1<u64, L, CC><<<dim3(N / (1 << M1) / thc, batch), thc, smc>>>((i64 *) x, t.primes, t.qinv, t.tw, 0, batch, 0, 0, s, cheddar::InputPtrList<u64, 1>());
    cheddar::kernel::NTTPhase2<u64, L, CR><<<dim3(N / (1 << M2) / thr, batch), thr, smr>>>((i64 *) x, t.primes, t.qinv, t.tw, t.twm, 0, batch, 0, 0, s); }
  static void inv(const Tables &t, u64 *x, int batch) { const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x; cheddar::InputPtrList<u64, 1> c; c.ptrs_[0] = t.ninv;
    cheddar::kernel::INTTPhase1<u64, L, CR><<<dim3(N / (1 << M2) / thr, batch), thr, smr>>>((i64 *) x, t.primes, t.qinv, t.itw, t.itwm, 0, batch, s);
    cheddar::kernel::INTTPhase2<u64, L, CC, cheddar::kernel::MultConst<u64>><<<dim3(N / (1 << M1) / thc, batch), thc, smc>>>((i64 *) x, t.primes, t.qinv, t.itw, 0, batch, s, c); }
};
template <typename F> static double time_ms(F &&fn, int iters) {
  fn(); ck(cudaDeviceSynchronize(), "warm"); cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s); for (int i = 0; i < iters; ++i) fn(); cudaEventRecord(e); ck(cudaEventSynchronize(e), "time");
  float ms; cudaEventElapsedTime(&ms, s, e); cudaEventDestroy(s); cudaEventDestroy(e); return ms / iters;
}
struct Ctx { int batch, iters, maxs; size_t tot; u64 *a, *b, *r, *tmp; unsigned long long *cnt; MergePlan *mp; Tables *t; std::function<void()> refill; };
static unsigned long long diff(Ctx &c) { cudaMemset(c.cnt, 0, 8); count_diff<<<1024, 256>>>(c.a, c.r, c.tot, c.cnt); unsigned long long v; cudaMemcpy(&v, c.cnt, 8, cudaMemcpyDeviceToHost); return v; }
template <int L, int P1, int M1> static void run_f4c(Ctx &c) {
  constexpr int P2 = L - P1, M2 = 3;
  if constexpr (P1 < M1 || P2 < M2 || (1 << (P1 - M1)) > 1024 || (1 << (P1 - M1)) < 32 || (1 << (P2 - M2)) > 1024 || (1 << (P2 - M2)) < 32) return; else {
    using RC = Rows<L, P2, M2>; using CC = ColsContig<L, P1, M1>;
    if (CC::sm > c.maxs || RC::sm > c.maxs) return;
    RC::setup(); CC::setup(); const int R = 1 << P1, Cn = 1 << P2, batch = c.batch; const Tables &t = *c.t;
    auto pw = [&] { fused4::pointwise_kernel<<<static_cast<unsigned>((c.tot + 255) / 256), 256>>>(c.a, c.b, c.mp->modulus, c.tot); };
    auto full = [&] {   // transposes on the SMs (functional reference of the DRU dataflow)
      for (u64 *x : {c.a, c.b}) { T(x, c.tmp, R, Cn, batch); CC::fwd(t, c.tmp, batch); T(c.tmp, x, Cn, R, batch); RC::fwd(t, x, batch); }
      pw(); RC::inv(t, c.a, batch); T(c.a, c.tmp, R, Cn, batch); CC::inv(t, c.tmp, batch); T(c.tmp, c.a, Cn, R, batch);
    };
    auto sm = [&] {     // SM lane: the same kernels, transposes left to the DRU
      CC::fwd(t, c.a, batch); RC::fwd(t, c.a, batch); CC::fwd(t, c.b, batch); RC::fwd(t, c.b, batch); pw(); RC::inv(t, c.a, batch); CC::inv(t, c.a, batch);
    };
    c.refill(); full(); ck(cudaDeviceSynchronize(), "f4c check"); const unsigned long long bad = diff(c);
    const double tf = time_ms(full, c.iters) / batch, ts = time_ms(sm, c.iters) / batch;
    printf("%d,%d,f4c_full,%d,%d,%d,0,%.4f,%llu\n%d,%d,f4c_sm,%d,%d,%d,0,%.4f,%llu\n", L, batch, P1, P2, M1, tf, bad, L, batch, P1, P2, M1, ts, bad); fflush(stdout);
  }
}
int main(int argc, char **argv) {
  std::vector<int> batches = {4, 16, 64}, lgs = {20, 21, 22, 23, 24}; int iters = 5;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--logN") && i + 1 < argc) { lgs.clear(); for (char *p = argv[++i]; *p;) { lgs.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; } }
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) { batches.clear(); for (char *p = argv[++i]; *p;) { batches.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; } }
  }
  int dev, maxs; cudaGetDevice(&dev); cudaDeviceGetAttribute(&maxs, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev); fprintf(stderr, "%s\n", p.name);
  printf("logN,batch,design,P1,P2,M1,B1,ms_per_poly,mismatches\n");
  for (int L : lgs) {
    MergePlan mp(L);
    for (int batch : batches) {
      Ctx c; c.batch = batch; c.iters = iters; c.maxs = maxs; c.tot = (1ull << L) * batch; c.mp = &mp;
      if (cudaMalloc(&c.a, c.tot * 8) || cudaMalloc(&c.b, c.tot * 8) || cudaMalloc(&c.r, c.tot * 8) || cudaMalloc(&c.tmp, c.tot * 8)) { printf("%d,%d,OOM\n", L, batch); cudaGetLastError(); continue; }
      ck(cudaMalloc(&c.cnt, 8), "cnt");
      c.refill = [&] { fill<<<1024, 256>>>(c.a, c.tot, 111 + L * 7 + batch); fill<<<1024, 256>>>(c.b, c.tot, 999 + L * 13 + batch); };
      c.refill(); mp.multiply(c.a, c.b, batch); ck(cudaMemcpy(c.r, c.a, c.tot * 8, cudaMemcpyDeviceToDevice), "ref");
      printf("%d,%d,merge,0,0,0,0,%.4f,0\n", L, batch, time_ms([&] { mp.multiply(c.a, c.b, batch); }, iters) / batch); fflush(stdout);
      Tables t(L, batch); c.t = &t;
#define X(LL, P2, M1, B1) if (LL == L) { using C = CheddarMul<LL, P2, M1, B1>; C::setup(); \
        auto mul = [&] { C::fwd(t, c.a, batch); C::fwd(t, c.b, batch); \
          fused4::pointwise_kernel<<<static_cast<unsigned>((c.tot + 255) / 256), 256>>>(c.a, c.b, mp.modulus, c.tot); C::inv(t, c.a, batch); }; \
        c.refill(); mul(); ck(cudaDeviceSynchronize(), "chk"); const unsigned long long bad = diff(c); \
        printf("%d,%d,cheddar,%d,%d,%d,%d,%.4f,%llu\n", L, batch, C::P1, P2, M1, B1, time_ms(mul, iters) / batch, bad); fflush(stdout); }
      CHEDDAR_CFGS(X)
#undef X
      cfor<20, 25>([&](auto LL) { if (LL == L) cfor<7, 14>([&](auto P1) { cfor<3, 5>([&](auto M1) { run_f4c<LL, P1, M1>(c); }); }); });
      cudaFree(c.a); cudaFree(c.b); cudaFree(c.r); cudaFree(c.tmp); cudaFree(c.cnt);
    }
  }
  return 0;
}
