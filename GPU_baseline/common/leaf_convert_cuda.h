// Device-side DPF output layer for the PCG-OLE protocol (Beaver-corrected
// scheme): consumes the device-resident leaf tensor produced by
// dpf_gpu_batch_full_eval_device and replaces the host output-layer loops in
// pcg_ole_impl.h so the B*D*16-byte leaves never cross PCIe. Only the
// per-instance sums (2B uint64) leave the device for the Beaver opens; the
// reconstructed CWs (B uint64) come back; the folded polynomial g (N uint64)
// is the sole large output.
//
// Semantics mirror the host output layer bit-exactly:
//   h[b][d]   = ChaCha8OutHash(leaf)          (key = leaf ‖ 0^128,
//                                              nonce = OUT_DOMAIN, ctr = 0)
//   C[b][d]   = low64(h) mod P                (NO party sign — host applies
//   tau[b][d] = lsb(leaf)                      signs and beta, exactly where
//   sumC[b]   = sum_d C[b][d]      (mod P)     the CPU path does)
//   sumT[b]   = sum_d tau[b][d]    (plain uint64)
// then, after the two Beaver network rounds on the host:
//   y = C + (tau ? CW : 0); party 1: y = -y
//   pos = (b/t + b%t) * (D/2) + d;  g[pos % N] += (pos < N ? +y : -y) (mod P)
//
// No modular multiply anywhere (the Beaver CW multiply happens on the host,
// once per instance) — so no GPU-NTT dependency in this unit.
//
// Plain C++ header (no CUDA types) so pcg_ole_impl.h can call it directly.

#ifndef COMMON_LEAF_CONVERT_CUDA_H__
#define COMMON_LEAF_CONVERT_CUDA_H__

#include <cstddef>
#include <cstdint>

#include "common/dpf_gpu.h"

namespace pcg_cuda {

// Fixed domain-separation nonce for the output-layer hash. MUST equal
// half_tree_dpf::ChaCha8OutHash::OUT_DOMAIN ("out_hash" in ASCII); the tree
// PRG runs with nonce = 0, so the two keystreams are domain-separated.
constexpr uint64_t kOutHashDomain = 0x6f75745f68617368ULL;

// Per-instance raw output-layer sums (sign-free):
//   sumC_host[b] = sum_d (low64(H'(leaf[b][d])) mod prime)  mod prime
//   sumT_host[b] = sum_d lsb(leaf[b][d])                    (plain add)
void dpf_out_sums(const DpfBlk* d_leaves, int B, size_t D, uint64_t prime,
                  uint64_t* sumC_host, uint64_t* sumT_host);

// After the host reconstructs CW[b] (Beaver opens), scatter-accumulate
// y = ±(C + tau?CW) into the negacyclically-folded polynomial:
// g_out_host = N uint64 residues (overwritten). Recomputes H'(leaf) on the
// fly (cheaper than persisting B*D C/tau arrays). bin_sz = D/2; N = t*bin_sz.
void dpf_out_scatter_g(const DpfBlk* d_leaves, int B, size_t D, int t,
                       int party, uint64_t prime, const uint64_t* CW_host,
                       uint64_t* g_out_host, int N);

void dpf_out_sums_v2(const DpfBlk* d_leaves, int B, size_t D, int t, int party, uint64_t prime, int N, uint64_t* sumC_host, uint64_t* sumT_host);
void dpf_out_scatter_g_v2(int B, size_t D, int t, int party, uint64_t prime, const uint64_t* CW_host, uint64_t* g_out_host, int N);
}  // namespace pcg_cuda

#endif  // COMMON_LEAF_CONVERT_CUDA_H__
