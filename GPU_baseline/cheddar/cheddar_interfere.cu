// cheddar_interfere: DRAM-interference sensitivity, measured on silicon, of Cheddar's two-kernel NTT and of the four-step
// on Cheddar's kernels (SM lane = transposes left to the DRU; full = transposes on the SMs), next to GPU-NTT merge, per
// negacyclic multiply. Same victims and kernels as cheddar_bigN_bench.cu, each at the config given on the command line (the
// best of cheddar_bigN_<tag>.csv, register cap = this build's -DBIGN_MAXRREG); same aggressor and protocol as ../fused4/interfere_ntt.cu:
// a persistent aggressor beside the victim, streaming (16 B read + write per step), SPU-like row burst (random 8 KB chunk,
// contiguous) or a sleep-only control with the same grid (SM occupancy), blocks x sleep sweeping its bandwidth.
// net slowdown = victim(stream | rowburst) / victim(control at the same blocks).  Every victim's product is checked once
// against merge computed one polynomial at a time (the SM lane through f4c_full, whose kernels it runs).
//   nvcc -DBIGN_LOGN=24 ...;  ./cheddar_interfere --batch 16 --cheddar P1,M1,B1,M2 --f4c P1,M1,B1,M2 --victims merge,cheddar,f4c --blocks 8,16,32,64,128
//   (--victims f4c times the SM lane and the full four-step, whose product checks the SM lane's kernels; f4csm times the SM
//   lane only and runs the full four-step once for the check)
// CSV: gpu,victim,logN,batch,P1,M1,B1,M2,maxrreg,blocks,sleep_ns,mode,agg_GBps,victim_ms,slowdown_raw,mismatches
#define BIGN_NO_MAIN
#include "cheddar_bigN_bench.cu"
#include <chrono>
#include <string>
#include <thread>

// ../fused4/interfere_ntt.cu:74-130 (mode 0 stream, 1 sleep only, 3 row burst)
__global__ void aggressor(const uint4 *__restrict__ src, uint4 *__restrict__ dst, size_t n16, volatile int *stop, unsigned sleep_ns, int mode,
                          unsigned long long *bytes) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  __shared__ int s_stop;
  for (;;) {
    if (threadIdx.x == 0) s_stop = *stop;
    __syncthreads();
    if (s_stop) break;
    if (mode == 0) {
#pragma unroll 4
      for (int k = 0; k < 16; ++k) { uint4 v = __ldcs(src + i); __stcs(dst + i, v); i += stride; if (i >= n16) i -= n16; }
      local += 16 * 32;
    } else if (mode == 3) {
      const size_t nchunk = n16 / blockDim.x;
      for (int k = 0; k < 16; ++k) {
        const size_t c = ((static_cast<size_t>(blockIdx.x) + 1) * 0x9E3779B97F4A7C15ULL + (local + k) * 0xBF58476D1CE4E5B9ULL) % nchunk;
        const size_t h = c * blockDim.x + threadIdx.x;
        uint4 v = __ldcs(src + h); __stcs(dst + h, v);
      }
      local += 16 * 32;
    }
    if (sleep_ns) __nanosleep(sleep_ns);
    else if (mode == 1) __nanosleep(1000);
    __syncthreads();
  }
  atomicAdd(bytes, local);
}
struct Agg {
  uint4 *src = nullptr, *dst = nullptr; size_t n16 = 0; int *d_stop = nullptr; unsigned long long *d_bytes = nullptr;
  cudaStream_t s, ctl; std::chrono::steady_clock::time_point t0;
  explicit Agg(size_t bytes) {
    n16 = bytes / 16; ck(cudaMalloc(&src, n16 * 16), "agg src"); ck(cudaMalloc(&dst, n16 * 16), "agg dst"); cudaMemset(src, 1, n16 * 16);
    ck(cudaMalloc(&d_stop, sizeof(int)), "stop"); ck(cudaMalloc(&d_bytes, 8), "bytes");
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream"); ck(cudaStreamCreateWithFlags(&ctl, cudaStreamNonBlocking), "ctl");
  }
  void start(int blocks, unsigned sleep_ns, int mode) {
    cudaMemsetAsync(d_stop, 0, sizeof(int), s); cudaMemsetAsync(d_bytes, 0, 8, s);
    aggressor<<<blocks, 512, 0, s>>>(src, dst, n16, d_stop, sleep_ns, mode, d_bytes); ck(cudaGetLastError(), "aggressor launch");
    t0 = std::chrono::steady_clock::now();
  }
  double stop_GBps() {
    static const int one = 1;
    cudaMemcpyAsync(d_stop, &one, sizeof(int), cudaMemcpyHostToDevice, ctl); ck(cudaStreamSynchronize(s), "agg sync");
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    unsigned long long b = 0; cudaMemcpy(&b, d_bytes, 8, cudaMemcpyDeviceToHost); return b / sec / 1e9;
  }
};
struct Bufs { u64 *a, *b, *a0, *b0, *tmp; size_t tot; int batch; };
using Mul = std::function<void()>;
template <int L> static bool make_cheddar(int p1, int m1, int b1, int m2, const Tables &t, Bufs &x, Mul &out) {   // Cheddar
  bool found = false;
  cfor<7, 15>([&](auto P1) { cfor<3, 6>([&](auto M1) { cfor<0, 5>([&](auto B1) { cfor<3, 6>([&](auto M2) {
    constexpr int P2 = L - P1;
    if constexpr (P2 >= 7 && P2 <= 14 && P1 >= M1 && P2 >= M2 && thr_ok(P1 - M1 + B1) && thr_ok(P2 - M2)) {
      if (P1 == p1 && M1 == m1 && B1 == b1 && M2 == m2) {
        using CC = Cols<L, P1, M1, B1>; using RC = Rows<L, P2, M2>;
        if (!CC::setup(1 << 30) || !RC::setup(1 << 30)) return;
        found = true; const Tables *tp = &t; Bufs *bp = &x;
        out = [tp, bp] { const Tables &t = *tp; Bufs &x = *bp; CC::fwd(t, x.a, x.batch); RC::fwd(t, x.a, x.batch); CC::fwd(t, x.b, x.batch); RC::fwd(t, x.b, x.batch);
          fused4::pointwise_kernel<<<static_cast<unsigned>((x.tot + 255) / 256), 256>>>(x.a, x.b, Modulus<Data64>(P), x.tot); RC::inv(t, x.a, x.batch); CC::inv(t, x.a, x.batch); };
      }
    } }); }); }); });
  return found;
}
template <int L> static bool make_f4c(int p1, int m1, int b1, int m2, bool full, const Tables &t, Bufs &x, Mul &out) {   // four-step
  bool found = false;
  cfor<7, 15>([&](auto P1) { cfor<3, 6>([&](auto M1) { cfor<0, 5>([&](auto B1) { cfor<3, 6>([&](auto M2) {
    constexpr int P2 = L - P1;
    if constexpr (P2 >= 7 && P2 <= 14 && P1 >= M1 && P2 >= M2 && P2 >= B1 && thr_ok(P1 - M1 + B1) && thr_ok(P2 - M2)) {
      if (P1 == p1 && M1 == m1 && B1 == b1 && M2 == m2) {
        using CC = ColsContig<L, P1, M1, B1>; using RC = Rows<L, P2, M2>;
        if (!CC::setup(1 << 30) || !RC::setup(1 << 30)) return;
        found = true; const Tables *tp = &t; Bufs *bp = &x; constexpr int R = 1 << P1, Cn = 1 << P2;
        if (full) out = [tp, bp] { const Tables &t = *tp; Bufs &x = *bp;
          for (u64 *v : {x.a, x.b}) { T(v, x.tmp, R, Cn, x.batch); CC::fwd(t, x.tmp, x.batch); T(x.tmp, v, Cn, R, x.batch); RC::fwd(t, v, x.batch); }
          fused4::pointwise_kernel<<<static_cast<unsigned>((x.tot + 255) / 256), 256>>>(x.a, x.b, Modulus<Data64>(P), x.tot);
          RC::inv(t, x.a, x.batch); T(x.a, x.tmp, R, Cn, x.batch); CC::inv(t, x.tmp, x.batch); T(x.tmp, x.a, Cn, R, x.batch); };
        else out = [tp, bp] { const Tables &t = *tp; Bufs &x = *bp;
          CC::fwd(t, x.a, x.batch); RC::fwd(t, x.a, x.batch); CC::fwd(t, x.b, x.batch); RC::fwd(t, x.b, x.batch);
          fused4::pointwise_kernel<<<static_cast<unsigned>((x.tot + 255) / 256), 256>>>(x.a, x.b, Modulus<Data64>(P), x.tot);
          RC::inv(t, x.a, x.batch); CC::inv(t, x.a, x.batch); };
      }
    } }); }); }); });
  return found;
}
template <int L> static bool make_f4b(int p1, int m2, const Tables &t, Bufs &x, Mul &out, Mul &check) {   // butterfly DRU: SM lane = row kernel only
  // check = Cheddar's product with Cols on the SMs standing in for the DRU's column pass (first column config that launches)
  bool found = false;
  cfor<7, 15>([&](auto P1) { cfor<3, 6>([&](auto M2) {
    constexpr int P2 = L - P1;
    if constexpr (P2 >= 7 && P2 <= 14 && P2 >= M2 && thr_ok(P2 - M2)) {
      if (P1 == p1 && M2 == m2) {
        using RC = Rows<L, P2, M2>;
        if (!RC::setup(1 << 30)) return;
        const Tables *tp = &t; Bufs *bp = &x;
        out = [tp, bp] { const Tables &t = *tp; Bufs &x = *bp; RC::fwd(t, x.a, x.batch); RC::fwd(t, x.b, x.batch);
          fused4::pointwise_kernel<<<static_cast<unsigned>((x.tot + 255) / 256), 256>>>(x.a, x.b, Modulus<Data64>(P), x.tot); RC::inv(t, x.a, x.batch); };
        cfor<3, 6>([&](auto M1) { cfor<0, 5>([&](auto B1) {
          if constexpr (P1 >= M1 && thr_ok(P1 - M1 + B1)) {
            if (found) return; using CC = Cols<L, P1, M1, B1>; if (!CC::setup(1 << 30)) return;
            found = true;
            check = [tp, bp] { const Tables &t = *tp; Bufs &x = *bp; CC::fwd(t, x.a, x.batch); RC::fwd(t, x.a, x.batch); CC::fwd(t, x.b, x.batch); RC::fwd(t, x.b, x.batch);
              fused4::pointwise_kernel<<<static_cast<unsigned>((x.tot + 255) / 256), 256>>>(x.a, x.b, Modulus<Data64>(P), x.tot); RC::inv(t, x.a, x.batch); CC::inv(t, x.a, x.batch); };
          } }); });
      }
    } }); });
  return found;
}
static std::vector<int> ints(const char *s) { std::vector<int> v; for (const char *p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; } return v; }
int main(int argc, char **argv) {
  constexpr int L = BIGN_LOGN; int batch = 16, iters = 10; std::vector<int> ch = {L / 2, 3, 0, 3}, fc = {L / 2, 3, 0, 3}, fb = {L / 2, 3}, blocks = {8, 16, 32, 64, 128};
  std::string victims = "merge,cheddar,f4c";
  std::vector<unsigned> sleeps = {0, 10000, 30000};
  for (int i = 1; i + 1 < argc; i++) {
    if (!strcmp(argv[i], "--batch")) batch = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cheddar")) ch = ints(argv[++i]);
    else if (!strcmp(argv[i], "--f4c")) fc = ints(argv[++i]);
    else if (!strcmp(argv[i], "--f4b")) fb = ints(argv[++i]);   // P1,M2 (row kernel only; "-" = default)
    else if (!strcmp(argv[i], "--blocks")) blocks = ints(argv[++i]);
    else if (!strcmp(argv[i], "--victims")) victims = argv[++i];
  }
  if (ch.size() < 4) ch.push_back(3);
  if (fc.size() == 3) fc.insert(fc.begin() + 2, 0);   // P1,M1,M2 (before B1): B1 = 0
  if (fc.size() < 4) fc.push_back(3);
  auto want = [&](const char *v) { return ("," + victims + ",").find("," + std::string(v) + ",") != std::string::npos; };
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0); std::string gpu = prop.name; for (auto &c : gpu) if (c == ' ') c = '_';
  const size_t N = 1ull << L; Bufs x; x.batch = batch; x.tot = N * batch;
  for (u64 **p : {&x.a, &x.b, &x.a0, &x.b0, &x.tmp}) ck(cudaMalloc(p, x.tot * 8), "buf");
  u64 *ref; ck(cudaMalloc(&ref, x.tot * 8), "ref"); unsigned long long *cnt; ck(cudaMalloc(&cnt, 8), "cnt");
  fill<<<4096, 256>>>(x.a0, x.tot, 111 + L); fill<<<4096, 256>>>(x.b0, x.tot, 999 + L);
  MergePlan mp(L); Tables t(L, batch);
  auto load = [&] { cudaMemcpyAsync(x.a, x.a0, x.tot * 8, cudaMemcpyDeviceToDevice, 0); cudaMemcpyAsync(x.b, x.b0, x.tot * 8, cudaMemcpyDeviceToDevice, 0); };
  load(); for (int i = 0; i < batch; i++) mp.multiply(x.a + i * N, x.b + i * N, 1);
  ck(cudaDeviceSynchronize(), "ref"); cudaMemcpy(ref, x.a, x.tot * 8, cudaMemcpyDeviceToDevice);
  struct V { const char *name; int p1, m1, b1, m2; Mul mul; Mul check; };   // check: the product that validates mul's kernels (empty = mul itself)
  std::vector<V> vs; Mul m, mf;
  if (want("merge")) vs.push_back({"merge", 0, 0, 0, 0, [&] { mp.multiply(x.a, x.b, batch); }, {}});
  if (want("cheddar")) {
    if (make_cheddar<L>(ch[0], ch[1], ch[2], ch[3], t, x, m)) vs.push_back({"cheddar", ch[0], ch[1], ch[2], ch[3], m, {}});
    else fprintf(stderr, "cheddar P1 %d M1 %d B1 %d M2 %d: not instantiated or cannot launch\n", ch[0], ch[1], ch[2], ch[3]);
  }
  if (want("f4c") || want("f4csm")) {   // the SM lane is timed; the full four-step (transposes on the SMs) checks its kernels' product
    const bool sm_ok = make_f4c<L>(fc[0], fc[1], fc[2], fc[3], false, t, x, m), full_ok = make_f4c<L>(fc[0], fc[1], fc[2], fc[3], true, t, x, mf);
    if (sm_ok) vs.push_back({"f4c_sm", fc[0], fc[1], fc[2], fc[3], m, full_ok ? mf : Mul{}});
    else fprintf(stderr, "f4c P1 %d M1 %d B1 %d M2 %d: not instantiated or cannot launch\n", fc[0], fc[1], fc[2], fc[3]);
    if (full_ok && want("f4c")) vs.push_back({"f4c_full", fc[0], fc[1], fc[2], fc[3], mf, {}});
  }
  if (want("f4b") && fb.size() >= 2) {   // butterfly DRU: row kernel only on the SMs, checked through Cheddar's product
    if (make_f4b<L>(fb[0], fb[1], t, x, m, mf)) vs.push_back({"f4b_sm", fb[0], 0, 0, fb[1], m, mf});
    else fprintf(stderr, "f4b P1 %d M2 %d: not instantiated or cannot launch\n", fb[0], fb[1]);
  }
  Agg agg(1ull << 30);
  auto victim_ms = [&](const Mul &mul) {   // per polynomial; operand reload outside the timed region
    cudaEvent_t c1, e; cudaEventCreate(&c1); cudaEventCreate(&e); load(); mul(); float tot = 0;
    for (int i = 0; i < iters; ++i) { load(); cudaEventRecord(c1, 0); mul(); cudaEventRecord(e, 0); cudaEventSynchronize(e); float a = 0; cudaEventElapsedTime(&a, c1, e); tot += a; }
    cudaEventDestroy(c1); cudaEventDestroy(e); return tot / iters / batch;
  };
  printf("gpu,victim,logN,batch,P1,M1,B1,M2,maxrreg,blocks,sleep_ns,mode,agg_GBps,victim_ms,slowdown_raw,mismatches\n");
  std::vector<unsigned long long> bads;
  for (auto &v : vs) {   // an SM lane alone leaves the product incomplete: it is checked through its check product (same kernels)
    load(); v.mul(); ck(cudaDeviceSynchronize(), v.name);
    if (v.check) { load(); v.check(); ck(cudaDeviceSynchronize(), v.name); }
    cudaMemset(cnt, 0, 8); count_diff<<<4096, 256>>>(x.a, ref, x.tot, cnt);
    unsigned long long b; cudaMemcpy(&b, cnt, 8, cudaMemcpyDeviceToHost); bads.push_back(v.check || strcmp(v.name, "f4c_sm") ? b : ~0ull);
  }
  for (size_t vi = 0; vi < vs.size(); vi++) {
    auto &v = vs[vi]; const unsigned long long bad = bads[vi];
    const double base = victim_ms(v.mul);
    printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,0,-,none,0,%.4f,1.000,%llu\n", gpu.c_str(), v.name, L, batch, v.p1, v.m1, v.b1, v.m2, BIGN_MAXRREG, base, bad);
    for (int b : blocks) for (int mode : {1, 0, 3}) for (unsigned sl : sleeps) {
      if (mode == 1 && sl != 0) continue;
      agg.start(b, sl, mode); std::this_thread::sleep_for(std::chrono::milliseconds(20));
      const double tm = victim_ms(v.mul), bw = agg.stop_GBps();
      printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%u,%s,%.1f,%.4f,%.3f,%llu\n", gpu.c_str(), v.name, L, batch, v.p1, v.m1, v.b1, v.m2, BIGN_MAXRREG, b, sl,
             mode == 0 ? "stream" : mode == 3 ? "rowburst" : "sleep_only", bw, tm, tm / base, bad);
      fflush(stdout);
    }
  }
  return 0;
}
