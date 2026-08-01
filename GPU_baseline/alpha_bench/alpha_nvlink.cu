// Measure GPU<->GPU small-message round-trip latency over NVLink/PCIe P2P.
// One "rung" = ping (GPU0->GPU1) + pong (GPU1->GPU0) of MSG bytes via
// cudaMemcpyPeerAsync, event-timed. Reports median/p10/p90 of RTT in us.
// This is the alpha the network model's NVLink tier should use (message
// sizes match the protocol: 4KB = one level's CW exchange at t=16).
//
// Build: nvcc -O3 -o alpha_nvlink alpha_nvlink.cu
// Run  : ./alpha_nvlink [msg_bytes=4096] [iters=1000]
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

int main(int argc, char **argv) {
    size_t msg = argc > 1 ? atol(argv[1]) : 4096;
    int iters = argc > 2 ? atoi(argv[2]) : 1000;
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { printf("alpha_nvlink,%zu,NA,NA,NA,only_%d_gpu\n", msg, ndev); return 0; }
    int can01 = 0, can10 = 0;
    CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    if (!can01 || !can10) { printf("alpha_nvlink,%zu,NA,NA,NA,no_p2p\n", msg); return 0; }
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
    CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));
    void *b0, *b1;
    CK(cudaSetDevice(0)); CK(cudaMalloc(&b0, msg));
    CK(cudaSetDevice(1)); CK(cudaMalloc(&b1, msg));
    CK(cudaSetDevice(0));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> rtt(iters);
    for (int w = 0; w < 50; w++) {                       // warmup
        CK(cudaMemcpyPeerAsync(b1, 1, b0, 0, msg, s));
        CK(cudaMemcpyPeerAsync(b0, 0, b1, 1, msg, s));
    }
    CK(cudaStreamSynchronize(s));
    for (int i = 0; i < iters; i++) {
        CK(cudaEventRecord(e0, s));
        CK(cudaMemcpyPeerAsync(b1, 1, b0, 0, msg, s));   // ping
        CK(cudaMemcpyPeerAsync(b0, 0, b1, 1, msg, s));   // pong
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        rtt[i] = ms * 1000.0f;                           // us
    }
    std::sort(rtt.begin(), rtt.end());
    printf("alpha_nvlink,%zu,%.2f,%.2f,%.2f,ok\n", msg,
           rtt[iters / 2], rtt[iters / 10], rtt[iters * 9 / 10]);
    return 0;
}
