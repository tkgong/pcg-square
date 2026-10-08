// DRAM-traffic probe: run one NTT design for a few multiplies so ncu can sum dram bytes per kernel.
//   ./ntt_bytes_probe --what merge|f4_full|f4_sm|f4_dru|f4g_full|f4g_sm --logN 24 --batch 4 --muls 2
#define main fused4_bench_main
#include "fused4_bench.cu"
#undef main
int main(int argc, char** argv) {
    std::string what = "merge"; int lg = 24, batch = 4, muls = 2;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--what") && i + 1 < argc) what = argv[++i];
        else if (!strcmp(argv[i], "--logN") && i + 1 < argc) lg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--muls") && i + 1 < argc) muls = atoi(argv[++i]);
    }
    const size_t total = static_cast<size_t>(batch) << lg;
    Data64 *x, *y, *tx, *ty;
    fused4::check(cudaMalloc(&x, total * 8), "x"); fused4::check(cudaMalloc(&y, total * 8), "y");
    fused4::check(cudaMalloc(&tx, total * 8), "tx"); fused4::check(cudaMalloc(&ty, total * 8), "ty");
    fill(x, total, 1234 + lg); fill(y, total, 5678 + lg);
    fused4::check(cudaDeviceSynchronize(), "fill");
    std::unique_ptr<fused4::Plan> plan; std::unique_ptr<MergePlan> merge; std::unique_ptr<fused4g::PlanG> plang;
    if (what == "merge") merge.reset(new MergePlan(lg));
    else if (what.rfind("f4g", 0) == 0) plang.reset(new fused4g::PlanG(lg, P));
    else plan.reset(new fused4::Plan(lg, P));
    const int lanes = (what.find("_sm") != std::string::npos) ? 1 : (what.find("_dru") != std::string::npos) ? 2 : 3;
    fused4::check(cudaDeviceSynchronize(), "setup");
    for (int m = 0; m < muls; ++m) {
        if (merge) merge->multiply(x, y, batch);
        else if (plang) plang->multiply(x, y, tx, ty, batch, lanes);
        else plan->multiply(x, y, tx, ty, batch, lanes);
    }
    fused4::check(cudaDeviceSynchronize(), "run");
    printf("done %s logN=%d batch=%d muls=%d\n", what.c_str(), lg, batch, muls);
    return 0;
}
