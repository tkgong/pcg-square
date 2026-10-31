// What a DRU can buy Cheddar's two-kernel NTT: per transform, time Cheddar's strided column kernel (NTTPhase1) and contiguous
// row kernel (NTTPhase2) separately at the best config of the sweep, and the contiguous-pass cost of P stages on rows of 2^P
// elements (NTTPhase2 with RadixStages = P over the same bytes: the access pattern the column pass gets once a DRU has
// transposed the data). Timing only (kernel time is data-independent); 1 GB of 64-bit words, same layout as cheddar_ntt_bench.
// Output: logN,kind,P,M,B,ms_per_transform
#include <cstdio>
#include <cstring>
#include <utility>
#include <cuda_runtime.h>
#include "gen/cheddar_ntt_kernels.cuh"
using u64 = uint64_t; using i64 = int64_t;
static void ck(cudaError_t e, const char *w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }
template <int Pn, int M, int Bt> struct CCfg {
  __host__ __device__ static constexpr int RadixStages() { return Pn; }
  __host__ __device__ static constexpr int StageMerging() { return M; }
  __host__ __device__ static constexpr int LogWarpBatching() { return Bt; }
  __host__ __device__ static constexpr int LsbSize() { return 32; }
  __host__ __device__ static constexpr bool OFTwiddle() { return true; }
};
template <int S, int E, class F> constexpr void cfor(F &&f) { if constexpr (S < E) { f(std::integral_constant<int, S>()); cfor<S + 1, E>(f); } }
struct Buf { i64 *x; u64 *pr, *tw, *twm; i64 *qi; int batch; };
template <typename F> static double tms(F &&f, int it) {
  f(); ck(cudaDeviceSynchronize(), "warm"); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); for (int i = 0; i < it; i++) f(); cudaEventRecord(b); ck(cudaEventSynchronize(b), "t");
  float ms; cudaEventElapsedTime(&ms, a, b); return ms / it;
}
template <int L, int P, int M, int Bt> static void col(Buf &b, int it, int maxs) {   // Cheddar's strided column kernel
  constexpr int th = 1 << (P - M + Bt); constexpr int sm = th * (1 << M) * 8;
  if constexpr (P >= M && th <= 1024 && th >= 32) { if (sm > maxs) return;
    auto k = cheddar::kernel::NTTPhase1<u64, L, CCfg<P, M, Bt>>; ck(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm), "a");
    cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = b.x; const size_t N = 1ull << L;
    double t = tms([&] { k<<<dim3(N / (1 << M) / th, b.batch), th, sm>>>(b.x, b.pr, b.qi, b.tw, 0, b.batch, 0, 0, s, cheddar::InputPtrList<u64, 1>()); }, it);
    printf("%d,col_strided,%d,%d,%d,%.4f\n", L, P, M, Bt, t / b.batch); fflush(stdout); }
}
template <int L, int P, int M> static void row(Buf &b, int it, int maxs) {   // contiguous P-stage pass on rows of 2^P elements
  constexpr int th = 1 << (P - M); constexpr int sm = th * (1 << M) * 8;
  if constexpr (P >= M && th <= 1024 && th >= 32) { if (sm > maxs) return;
    auto k = cheddar::kernel::NTTPhase2<u64, L, CCfg<P, M, 0>>; ck(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm), "a");
    cheddar::InputPtrList<i64, 1> s; s.ptrs_[0] = b.x; const size_t N = 1ull << L;
    double t = tms([&] { k<<<dim3(N / (1 << M) / th, b.batch), th, sm>>>(b.x, b.pr, b.qi, b.tw, b.twm, 0, b.batch, 0, 0, s); }, it);
    printf("%d,row_contig,%d,%d,0,%.4f\n", L, P, M, t / b.batch); fflush(stdout); }
}
template <int L> static void runL(int it, int maxs) {
  Buf b; b.batch = 1 << (27 - L); const size_t N = 1ull << L, tot = N * b.batch;
  ck(cudaMalloc(&b.x, tot * 8), "x"); cudaMemset(b.x, 0, tot * 8);
  ck(cudaMalloc(&b.pr, b.batch * 8), "p"); ck(cudaMalloc(&b.qi, b.batch * 8), "q"); ck(cudaMalloc(&b.tw, N * 8), "tw"); ck(cudaMalloc(&b.twm, N / 32 * 8), "twm");
  cudaMemset(b.pr, 0xff, b.batch * 8); cudaMemset(b.qi, 0x11, b.batch * 8); cudaMemset(b.tw, 0x22, N * 8); cudaMemset(b.twm, 0x33, N / 32 * 8);
  cfor<7, 15>([&](auto P) { cfor<3, 5>([&](auto M) { row<L, P, M>(b, it, maxs); cfor<1, 5>([&](auto Bt) { col<L, P, M, Bt>(b, it, maxs); }); }); });
  cudaFree(b.x); cudaFree(b.pr); cudaFree(b.qi); cudaFree(b.tw); cudaFree(b.twm);
}
int main(int argc, char **argv) {
  int it = 10; for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--iters") && i + 1 < argc) it = atoi(argv[++i]);
  int dev, maxs; cudaGetDevice(&dev); cudaDeviceGetAttribute(&maxs, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev); fprintf(stderr, "%s, max smem/block %d\n", p.name, maxs);
  printf("logN,kind,P,M,B,ms_per_transform\n");
  cfor<20, 25>([&](auto L) { runL<L>(it, maxs); });
  return 0;
}
