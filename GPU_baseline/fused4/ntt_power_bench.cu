// ntt_power_bench: run one NTT poly-mul variant back to back for a fixed wall time so the
// board power can be sampled (nvidia-smi) over a steady phase. Reuses fused4_bench's plans.
//
//   ./ntt_power_bench --what merge|f4g_sm|sleep --logN 22 --batch 16 --secs 10
//
// Prints: what,logN,batch,secs,muls,ms_per_mul
#define main fused4_bench_main
#include "fused4_bench.cu"
#undef main
#include <thread>

int main(int argc, char** argv) {
    int lg = 22, batch = 16; double secs = 10.0; std::string what = "merge";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--logN") && i + 1 < argc) lg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--secs") && i + 1 < argc) secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--what") && i + 1 < argc) what = argv[++i];
    }
    cudaFree(0);                                   // context up (the "idle with context" phase)
    if (what == "sleep") {
        std::this_thread::sleep_for(std::chrono::milliseconds((long)(secs * 1000)));
        printf("sleep,%d,%d,%.1f,0,0\n", lg, batch, secs);
        return 0;
    }
    const size_t total = static_cast<size_t>(batch) << lg;
    Data64 *x, *y, *tx, *ty;
    fused4::check(cudaMalloc(&x, total * 8), "cudaMalloc x");
    fused4::check(cudaMalloc(&y, total * 8), "cudaMalloc y");
    fused4::check(cudaMalloc(&tx, total * 8), "cudaMalloc tx");
    fused4::check(cudaMalloc(&ty, total * 8), "cudaMalloc ty");
    fill(x, total, 1234 + lg); fill(y, total, 5678 + lg);
    std::unique_ptr<MergePlan> merge;
    std::unique_ptr<fused4g::PlanG> plang;
    if (what == "merge") merge.reset(new MergePlan(lg));
    else if (what == "f4g_sm") plang.reset(new fused4g::PlanG(lg, P));
    else { fprintf(stderr, "unknown --what %s\n", what.c_str()); return 2; }
    auto run = [&] { if (merge) merge->multiply(x, y, batch); else plang->multiply(x, y, tx, ty, batch, 1); };
    run(); fused4::check(cudaDeviceSynchronize(), "warm-up");
    const auto t0 = std::chrono::steady_clock::now();
    long muls = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        for (int k = 0; k < 4; ++k) run();
        fused4::check(cudaDeviceSynchronize(), "run");
        muls += 4L * batch;
    }
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%s,%d,%d,%.2f,%ld,%.4f\n", what.c_str(), lg, batch, el, muls, el * 1e3 / muls);
    return 0;
}
