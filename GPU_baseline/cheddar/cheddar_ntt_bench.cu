// Cheddar's forward NTT (two kernels: Phase1 = the first P1 stages on strided columns with warp batching, Phase2 = the last
// P2 stages on contiguous rows with on-the-fly twiddles), 64-bit words, PCG's prime 2^62 - 3*2^25 + 1, one twiddle table.
// For each logN, every feasible (P2, Phase1 stage merging, Phase1 warp batching) on this GPU is checked element-wise
// against a CPU Cooley-Tukey reference and timed on 1 GB of polynomials (same bytes as merge_scale_bench).
// Output: logN,batch,P1,P2,M1,B1,threads1,smem1_KB,threads2,smem2_KB,ok,ms_per_transform,ps_per_elem
//   ./cheddar_ntt_bench --logN 16,17,...,24 --iters 10
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include "gen/cheddar_ntt_kernels.cuh"
using namespace cheddar;
using u64 = uint64_t; using i64 = int64_t; using u128 = unsigned __int128;
static void ck(cudaError_t e, const char *w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }
static const u64 Q = 4611686018326724609ULL;   // 2^62 - 3*2^25 + 1 (PCG's NTT prime), 2-adicity 25
static u64 mulm(u64 a, u64 b) { return (u64) ((u128) a * b % Q); }
static u64 powm(u64 a, u64 e) { u64 r = 1; while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; } return r; }
static u64 tomont(u64 a) { return (u64) (((u128) a << 64) % Q); }
template <int P, int M, int Bt> struct Cfg {
  __host__ __device__ static constexpr int RadixStages() { return P; }
  __host__ __device__ static constexpr int StageMerging() { return M; }
  __host__ __device__ static constexpr int LogWarpBatching() { return Bt; }
  __host__ __device__ static constexpr int LsbSize() { return 32; }
  __host__ __device__ static constexpr bool OFTwiddle() { return true; }
};
template <int S, int E, class F> constexpr void cfor(F &&f) { if constexpr (S < E) { f(std::integral_constant<int, S>()); cfor<S + 1, E>(f); } }
__global__ void fill(i64 *x, size_t n, u64 seed) {
  for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
    u64 z = seed + 0x9e3779b97f4a7c15ULL * (i + 1); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; z ^= z >> 31;
    x[i] = (i64) (z % Q);
  }
}
static void ref_ntt(std::vector<u64> &a, const std::vector<u64> &psi_rev) {   // Longa-Naehrig CT, bit-reversed output
  size_t N = a.size(), t = N;
  for (size_t m = 1; m < N; m <<= 1) {
    t >>= 1;
    for (size_t i = 0; i < m; i++) {
      u64 S = psi_rev[m + i];
      for (size_t j = 2 * i * t; j < 2 * i * t + t; j++) { u64 U = a[j], V = mulm(a[j + t], S); a[j] = U + V >= Q ? U + V - Q : U + V; a[j + t] = U >= V ? U - V : U + Q - V; }
    }
  }
}
struct Ctx { int L, batch; i64 *x; u64 *primes, *tw, *twm; i64 *qinv; std::vector<u64> in0, inL, ref0, refL; int iters; int max_smem; };
template <int L, int P2, int M1, int B1> static void run_cfg(Ctx &c) {
  constexpr int P1 = L - P2, M2 = 3;
  if constexpr (P1 < M1 || P1 < 1) return; else {
    using C1 = Cfg<P1, M1, B1>; using C2 = Cfg<P2, M2, 0>;
    constexpr int th1 = 1 << (P1 - M1 + B1), th2 = 1 << (P2 - M2);
    constexpr size_t sm1 = (size_t) th1 * (1 << M1) * 8, sm2 = (size_t) th2 * (1 << M2) * 8;
    if (th1 > 1024 || th1 < 32 || th2 > 1024 || th2 < 32 || (int) sm1 > c.max_smem || (int) sm2 > c.max_smem) return;
    auto k1 = kernel::NTTPhase1<u64, L, C1>; auto k2 = kernel::NTTPhase2<u64, L, C2>;
    ck(cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm1), "attr1");
    ck(cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm2), "attr2");
    const size_t N = 1ull << L, tot = N * c.batch;
    dim3 g1(N / (1 << M1) / th1, c.batch), g2(N / (1 << M2) / th2, c.batch);
    InputPtrList<i64, 1> src; src.ptrs_[0] = c.x;
    auto go = [&] {
      k1<<<g1, th1, sm1>>>(c.x, c.primes, c.qinv, c.tw, 0, c.batch, 0, 0, src, InputPtrList<u64, 1>());
      k2<<<g2, th2, sm2>>>(c.x, c.primes, c.qinv, c.tw, c.twm, 0, c.batch, 0, 0, src);
    };
    fill<<<1024, 256>>>(c.x, tot, 12345); go(); ck(cudaDeviceSynchronize(), "check run");
    std::vector<u64> o0(N), oL(N);
    ck(cudaMemcpy(o0.data(), c.x, N * 8, cudaMemcpyDeviceToHost), "d2h0"); ck(cudaMemcpy(oL.data(), c.x + (tot - N), N * 8, cudaMemcpyDeviceToHost), "d2hL");
    bool ok = o0 == c.ref0 && oL == c.refL;
    go(); ck(cudaDeviceSynchronize(), "warm");
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < c.iters; i++) go(); cudaEventRecord(b); ck(cudaEventSynchronize(b), "time");
    float ms; cudaEventElapsedTime(&ms, a, b); double t = ms / c.iters / c.batch;
    printf("%d,%d,%d,%d,%d,%d,%d,%.0f,%d,%.0f,%d,%.4f,%.1f\n", L, c.batch, P1, P2, M1, B1, th1, sm1 / 1024.0, th2, sm2 / 1024.0, ok, t, t * 1e9 / N / 2);
    fflush(stdout); cudaEventDestroy(a); cudaEventDestroy(b);
  }
}
template <int L> static void run_L(int iters, int max_smem) {
  Ctx c; c.L = L; c.batch = 1 << (27 - L); c.iters = iters; c.max_smem = max_smem;
  const size_t N = 1ull << L, tot = N * c.batch;
  u64 g = 3; while (powm(g, (Q - 1) / 2) == 1) g++;
  u64 psi = powm(g, (Q - 1) / (2 * N)); if (powm(psi, N) != Q - 1) { fprintf(stderr, "bad psi\n"); exit(1); }
  std::vector<u64> pw(N), psi_rev(N), mont(N), msb(N / 32);
  pw[0] = 1; for (size_t j = 1; j < N; j++) pw[j] = mulm(pw[j - 1], psi);
  for (size_t j = 0; j < N; j++) { size_t r = 0; for (int k = 0; k < L; k++) r |= ((j >> k) & 1) << (L - 1 - k); psi_rev[r] = pw[j]; }
  for (size_t j = 0; j < N; j++) mont[j] = tomont(psi_rev[j]);
  for (size_t j = 0; j < N / 32; j++) msb[j] = mont[j * 32];   // NTT.cu PopulateTwiddleFactors, lsb_size 32
  u64 qi = 1; for (int k = 0; k < 7; k++) qi *= 2 - Q * qi;      // Q^-1 mod 2^64
  std::vector<u64> pr(c.batch, Q); std::vector<i64> qv(c.batch, (i64) qi);
  ck(cudaMalloc(&c.x, tot * 8), "x"); ck(cudaMalloc(&c.primes, c.batch * 8), "p"); ck(cudaMalloc(&c.qinv, c.batch * 8), "qi");
  ck(cudaMalloc(&c.tw, N * 8), "tw"); ck(cudaMalloc(&c.twm, N / 32 * 8), "twm");
  cudaMemcpy(c.primes, pr.data(), c.batch * 8, cudaMemcpyHostToDevice); cudaMemcpy(c.qinv, qv.data(), c.batch * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(c.tw, mont.data(), N * 8, cudaMemcpyHostToDevice); cudaMemcpy(c.twm, msb.data(), N / 32 * 8, cudaMemcpyHostToDevice);
  fill<<<1024, 256>>>(c.x, tot, 12345); ck(cudaDeviceSynchronize(), "fill");
  c.ref0.resize(N); c.refL.resize(N);
  cudaMemcpy(c.ref0.data(), c.x, N * 8, cudaMemcpyDeviceToHost); cudaMemcpy(c.refL.data(), c.x + (tot - N), N * 8, cudaMemcpyDeviceToHost);
  ref_ntt(c.ref0, psi_rev); ref_ntt(c.refL, psi_rev);
  run_cfg<L, 9, 4, 4>(c);   // Cheddar's own 2^16 setting first (P2 = 9, M1 = 4, B1 = 4)
  cfor<9, 14>([&](auto P2) { cfor<3, 5>([&](auto M1) { cfor<1, 5>([&](auto B1) {
    if constexpr (!(P2 == 9 && M1 == 4 && B1 == 4)) run_cfg<L, P2, M1, B1>(c); }); }); });
  cudaFree(c.x); cudaFree(c.primes); cudaFree(c.qinv); cudaFree(c.tw); cudaFree(c.twm);
}
int main(int argc, char **argv) {
  std::vector<int> lgs = {16, 17, 18, 19, 20, 21, 22, 23, 24}; int iters = 10;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--logN") && i + 1 < argc) { lgs.clear(); for (char *p = argv[++i]; *p;) { lgs.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; } }
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
  }
  int dev; cudaGetDevice(&dev); int max_smem; cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev); fprintf(stderr, "%s, max opt-in shared memory per block %d B\n", p.name, max_smem);
  printf("logN,batch,P1,P2,M1,B1,threads1,smem1_KB,threads2,smem2_KB,ok,ms_per_transform,ps_per_elem_per_pass\n");
  for (int L : lgs) cfor<16, 25>([&](auto LL) { if (LL == L) run_L<LL>(iters, max_smem); });
  return 0;
}
