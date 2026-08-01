// Host-facing interface for the GPU full-domain half-tree DPF evaluation
// (device implementation in common/dpf_gpu.cu). Plain C++ (no CUDA) so the
// half_tree_dpf library can dispatch into it.
//
// The DPF tree expansion is local compute EXCEPT for one tiny per-level network
// round (exchange of B*(m-1) correction-word shares). That round is kept on the
// host via the `exchange` callback (which wraps emp::NetIO), so this entry point
// stays free of emp/networking dependencies.

#ifndef COMMON_DPF_GPU_H__
#define COMMON_DPF_GPU_H__

#include <cstddef>
#include <cstdint>
#include <functional>

namespace pcg_cuda {

// 128-bit block as two little-endian halves — matches the in-memory layout of
// emp::block (lo = bytes 0..7, hi = bytes 8..15; lsb of the block = lo & 1).
// 16-byte alignment matches __m128i (sound reinterpret_cast to emp::block on
// the host; vectorized 128-bit loads on the device).
struct alignas(16) DpfBlk {
    uint64_t lo;
    uint64_t hi;
};

// Per-level correction-word exchange, implemented on the host (emp::NetIO).
//   my_cws[0..count)   : this party's CW shares (input).
//   peer_cws[0..count) : filled with the peer's CW shares (output).
using DpfExchangeFn =
    std::function<void(const DpfBlk* my_cws, DpfBlk* peer_cws, size_t count)>;

// Device tree-PRG selection. Must match the CPU TreePRG backend bit-exactly:
//   AES     — CCR hash H_k(x) = AES_k(x) ^ x (AESTreePRG / MKeyPRP).
//   CHACHA8 — H_k(x) = 16-byte chunk k of the ChaCha8 keystream with
//             key = x ‖ 0^128, nonce = 0, counter = k/4 (ChaCha8TreePRG).
enum class DpfGpuPrg { AES, CHACHA8 };

// Batched half-tree full-domain DPF evaluation on the GPU.
//   B          : number of independent DPF instances.
//   n, w       : domain exponent (D = 2^n) and branching width (m = 2^w, nk = m-1).
//   roots      : B blocks; roots[b] = initial frontier (delta_share ^ W) of inst b.
//   dltsft     : B*levels*nk blocks, indexed [b*levels*nk + i*nk + k]
//                = DltSft_share[b][level i][child k].
//   round_keys : AES only — nk*176 bytes; round_keys[k*176 ..] = AES-128 round
//                keys for the k-th tree-PRG sub-key (CCR H_k(x) = AES_k(x)^x).
//                Pass nullptr for CHACHA8 (keyless: the parent is the key).
//   party      : 0 or 1 (only affects send/recv ordering inside `exchange`).
//   exchange   : per-level network round, carrying B*nk blocks.
//   out_leaves : B*D blocks; row b (out_leaves[b*D + d]) = instance b's leaves.
//   prg        : device tree-PRG (must match the CPU oracle's PRGType).
//
// Requires a CUDA device. Throws std::runtime_error on CUDA failure.
void dpf_gpu_batch_full_eval(int B, int n, int w,
                             const DpfBlk* roots,
                             const DpfBlk* dltsft,
                             const uint8_t* round_keys,
                             int party,
                             const DpfExchangeFn& exchange,
                             DpfBlk* out_leaves,
                             DpfGpuPrg prg = DpfGpuPrg::AES);

// Device-resident variant: same protocol, but the leaves are NOT copied back.
// Returns the device pointer to B rows of stride D (aliases the internal
// buffer pool; valid until the next dpf_gpu_* call). Use with the
// leaf-conversion kernels (common/leaf_convert_cuda.h) to keep the B*D*16-byte
// leaf tensor off the PCIe bus entirely.
const DpfBlk* dpf_gpu_batch_full_eval_device(int B, int n, int w,
                                             const DpfBlk* roots,
                                             const DpfBlk* dltsft,
                                             const uint8_t* round_keys,
                                             int party,
                                             const DpfExchangeFn& exchange,
                                             DpfGpuPrg prg = DpfGpuPrg::AES);

}  // namespace pcg_cuda

#endif  // COMMON_DPF_GPU_H__
