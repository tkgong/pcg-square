// cheddar_power_bench: run one Cheddar-kernel NTT poly-mul variant back to back for a fixed wall time so the board power
// can be sampled over a steady phase (run_power.sh). Same kernels, configs and dataflow as cheddar_interfere.cu:
//   cheddar : Cheddar's two-kernel NTT (the GPU baseline's NTT)             --cfg P1,M1,B1,M2
//   f4c_sm  : four-step on Cheddar's kernels, SM lane (transposes on the DRU) --cfg P1,M1,B1,M2
//   f4b_sm  : butterfly DRU, SM lane = row kernel + pointwise only            --cfg P1,0,0,M2
// Compile with -DBIGN_LOGN=<logN> [-DBIGN_MAXRREG=<cap>] like the interference bench.
//   ./cheddar_power_bench --what cheddar --cfg 11,4,2,3 --batch 16 --secs 10
// Prints: what,logN,batch,secs,muls,ms_per_mul
#define main cheddar_interfere_main
#include "cheddar_interfere.cu"
#undef main
#include <chrono>
int main(int argc, char **argv) {
  constexpr int L = BIGN_LOGN; int batch = 16; double secs = 10.0; std::string what = "cheddar"; std::vector<int> cfg = {L / 2, 3, 0, 3};
  for (int i = 1; i + 1 < argc; i++) {
    if (!strcmp(argv[i], "--batch")) batch = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--secs")) secs = atof(argv[++i]);
    else if (!strcmp(argv[i], "--what")) what = argv[++i];
    else if (!strcmp(argv[i], "--cfg")) cfg = ints(argv[++i]);
  }
  while (cfg.size() < 4) cfg.push_back(cfg.size() == 3 ? 3 : 0);
  const size_t N = 1ull << L; Bufs x; x.batch = batch; x.tot = N * batch;
  for (u64 **p : {&x.a, &x.b, &x.a0, &x.b0, &x.tmp}) ck(cudaMalloc(p, x.tot * 8), "buf");
  fill<<<4096, 256>>>(x.a, x.tot, 111 + L); fill<<<4096, 256>>>(x.b, x.tot, 999 + L);
  Tables t(L, batch); Mul m, mf; bool ok;
  if (what == "cheddar") ok = make_cheddar<L>(cfg[0], cfg[1], cfg[2], cfg[3], t, x, m);
  else if (what == "f4c_sm") ok = make_f4c<L>(cfg[0], cfg[1], cfg[2], cfg[3], false, t, x, m);
  else if (what == "f4b_sm") ok = make_f4b<L>(cfg[0], cfg[3], t, x, m, mf);
  else { fprintf(stderr, "unknown --what %s\n", what.c_str()); return 2; }
  if (!ok) { fprintf(stderr, "%s P1 %d M1 %d B1 %d M2 %d: not instantiated or cannot launch\n", what.c_str(), cfg[0], cfg[1], cfg[2], cfg[3]); return 3; }
  m(); ck(cudaDeviceSynchronize(), "warm-up");   // the timed loop runs the kernels on whatever is in the buffers: power does not depend on the data
  const auto t0 = std::chrono::steady_clock::now(); long muls = 0;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
    for (int k = 0; k < 4; ++k) m();
    ck(cudaDeviceSynchronize(), "run"); muls += 4L * batch;
  }
  const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("%s,%d,%d,%.2f,%ld,%.4f\n", what.c_str(), L, batch, el, muls, el * 1e3 / muls);
  return 0;
}
