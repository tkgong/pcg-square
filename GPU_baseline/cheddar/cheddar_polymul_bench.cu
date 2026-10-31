// Negacyclic polynomial multiply (2 forward NTT + pointwise + 1 inverse NTT, both operands transformed: the PCG
// baseline's work unit) at PCG's sizes, logN 20..24, batch 4/16/64, prime 2^62 - 3*2^25 + 1, on one GPU:
//   merge    GPU-NTT merge (HEonGPU's NTT), exactly as GPU_baseline/fused4/fused4_bench.cu MergePlan
//   cheddar  Cheddar's NTT kernels (two passes per transform) with the configs in gen/cfgs.h, same pointwise kernel
// Every Cheddar result is compared element-wise against the merge product of the same inputs (whole batch).
// Output: logN,batch,merge_ms_per_poly,cheddar_P1,P2,M1,B1,cheddar_ms_per_poly,mismatches,merge_over_cheddar
#include "fused4/fused4_ntt.cuh"   // -I.. (GPU_baseline)
#include "gpuntt/common/nttparameters.cuh"
#include "gen/cheddar_ntt_kernels.cuh"
#include "gen/cfgs.h"
#include <cstdio>
#include <cstring>
#include <vector>
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
struct CheddarTables {   // NTT.cu PopulateTwiddleFactors for one prime
  int L; u64 *tw, *twm, *itw, *itwm, *primes, *ninv; i64 *qinv;
  CheddarTables(int L_, int batch) : L(L_) {
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
  ~CheddarTables() { cudaFree(tw); cudaFree(twm); cudaFree(itw); cudaFree(itwm); cudaFree(primes); cudaFree(ninv); cudaFree(qinv); }
};
template <int L, int P2, int M1, int B1> struct CheddarMul {
  static constexpr int P1 = L - P2, M2 = 3;
  using CC = CCfg<P1, M1, B1>; using CR = CCfg<P2, M2, 0>;   // column (strided) pass, row (contiguous) pass
  static constexpr int thc = 1 << (P1 - M1 + B1), thr = 1 << (P2 - M2);
  static constexpr int smc = thc * (1 << M1) * 8, smr = thr * (1 << M2) * 8;
  static void setup() {
    ck(cudaFuncSetAttribute(cheddar::kernel::NTTPhase1<u64, L, CC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smc), "a1");
    ck(cudaFuncSetAttribute(cheddar::kernel::NTTPhase2<u64, L, CR>, cudaFuncAttributeMaxDynamicSharedMemorySize, smr), "a2");
    ck(cudaFuncSetAttribute(cheddar::kernel::INTTPhase1<u64, L, CR>, cudaFuncAttributeMaxDynamicSharedMemorySize, smr), "a3");
    ck(cudaFuncSetAttribute(cheddar::kernel::INTTPhase2<u64, L, CC, cheddar::kernel::MultConst<u64>>, cudaFuncAttributeMaxDynamicSharedMemorySize, smc), "a4");
  }
  static void fwd(const CheddarTables &t, u64 *x, int batch) {
    const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x;
    cheddar::kernel::NTTPhase1<u64, L, CC><<<dim3(N / (1 << M1) / thc, batch), thc, smc>>>((i64 *) x, t.primes, t.qinv, t.tw, 0, batch, 0, 0, s, cheddar::InputPtrList<u64, 1>());
    cheddar::kernel::NTTPhase2<u64, L, CR><<<dim3(N / (1 << M2) / thr, batch), thr, smr>>>((i64 *) x, t.primes, t.qinv, t.tw, t.twm, 0, batch, 0, 0, s);
  }
  static void inv(const CheddarTables &t, u64 *x, int batch) {
    const size_t N = 1ull << L; cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = (i64 *) x; cheddar::InputPtrList<u64, 1> c; c.ptrs_[0] = t.ninv;
    cheddar::kernel::INTTPhase1<u64, L, CR><<<dim3(N / (1 << M2) / thr, batch), thr, smr>>>((i64 *) x, t.primes, t.qinv, t.itw, t.itwm, 0, batch, s);
    cheddar::kernel::INTTPhase2<u64, L, CC, cheddar::kernel::MultConst<u64>><<<dim3(N / (1 << M1) / thc, batch), thc, smc>>>((i64 *) x, t.primes, t.qinv, t.itw, 0, batch, s, c);
  }
};
template <typename F> static double time_ms(F &&fn, int iters) {
  fn(); ck(cudaDeviceSynchronize(), "warm"); cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s); for (int i = 0; i < iters; ++i) fn(); cudaEventRecord(e); ck(cudaEventSynchronize(e), "time");
  float ms; cudaEventElapsedTime(&ms, s, e); cudaEventDestroy(s); cudaEventDestroy(e); return ms / iters;
}
int main(int argc, char **argv) {
  std::vector<int> batches = {4, 16, 64}; int iters = 5;
  for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
  int dev; cudaGetDevice(&dev); cudaDeviceProp p; cudaGetDeviceProperties(&p, dev); fprintf(stderr, "%s\n", p.name);
  printf("logN,batch,merge_ms_per_poly,cheddar_P1,P2,M1,B1,cheddar_ms_per_poly,mismatches,merge_over_cheddar\n");
  for (int L = 20; L <= 24; L++) {
    MergePlan mp(L);
    for (int batch : batches) {
      const size_t N = 1ull << L, tot = N * batch; u64 *a, *b, *r; unsigned long long *cnt;
      if (cudaMalloc(&a, tot * 8) != cudaSuccess || cudaMalloc(&b, tot * 8) != cudaSuccess || cudaMalloc(&r, tot * 8) != cudaSuccess) {
        printf("%d,%d,OOM\n", L, batch); cudaGetLastError(); cudaFree(a); cudaFree(b); continue; }
      ck(cudaMalloc(&cnt, 8), "cnt");
      auto refill = [&] { fill<<<1024, 256>>>(a, tot, 111 + L * 7 + batch); fill<<<1024, 256>>>(b, tot, 999 + L * 13 + batch); };
      refill(); mp.multiply(a, b, batch); ck(cudaMemcpy(r, a, tot * 8, cudaMemcpyDeviceToDevice), "ref");
      const double tm = time_ms([&] { mp.multiply(a, b, batch); }, iters) / batch;
      CheddarTables t(L, batch);
#define X(LL, P2, M1, B1) if (LL == L) { using C = CheddarMul<LL, P2, M1, B1>; C::setup(); \
        auto mul = [&] { C::fwd(t, a, batch); C::fwd(t, b, batch); \
          fused4::pointwise_kernel<<<static_cast<unsigned>((tot + 255) / 256), 256>>>(a, b, mp.modulus, tot); C::inv(t, a, batch); }; \
        refill(); mul(); ck(cudaDeviceSynchronize(), "chk"); cudaMemset(cnt, 0, 8); count_diff<<<1024, 256>>>(a, r, tot, cnt); \
        unsigned long long bad; cudaMemcpy(&bad, cnt, 8, cudaMemcpyDeviceToHost); const double tc = time_ms(mul, iters) / batch; \
        printf("%d,%d,%.4f,%d,%d,%d,%d,%.4f,%llu,%.3f\n", L, batch, tm, C::P1, P2, M1, B1, tc, bad, tm / tc); fflush(stdout); }
      CHEDDAR_CFGS(X)
#undef X
      cudaFree(a); cudaFree(b); cudaFree(r); cudaFree(cnt);
    }
  }
  return 0;
}
