// Half-Tree m-ary DPF correlation generator.
// Two parties jointly generate shares of a point function f_{α,Δ}: [0,N) → F_{2^λ}
// with N = 2^n and branching m = 2^w. See Protocol Π_{DPF_{m-ary}}
// (simplified Half-Tree, GYW+22 Fig. 10).

#ifndef HALF_TREE_DPF_DPF_H__
#define HALF_TREE_DPF_DPF_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <emp-tool/io/net_io_channel.h>
#include <emp-tool/utils/block.h>

#include "bool_circuit/circuit.h"
#include "half_tree_dpf/tree_prg.h"

namespace pcg_cuda { struct DpfBlk; }  // common/dpf_gpu.h (plain C++ POD)

namespace half_tree_dpf {

using Block = emp::block;
extern const Block ZERO_BLOCK;

// ---- small block helpers ------------------------------------------------
inline Block block_xor(const Block& a, const Block& b) noexcept { return a ^ b; }
inline bool  get_lsb(const Block& b) noexcept { return emp::getLSB(b); }
inline Block cond_xor(bool cond, const Block& cw) noexcept {
    return cond ? cw : ZERO_BLOCK;
}
bool   block_eq(const Block& a, const Block& b) noexcept;
std::string hex(const Block& b);

// ---- protocol data types -------------------------------------------------
struct PartyInput {
    int   n;
    int   w;
    Block W;
    Block delta_share;
    // DltSft_share[i][k] = ⟨(∏_{c} (α_{iw+c} ⊕ k̄_c))·Δ⟩_b, k ∈ [0, m-2]
    std::vector<std::vector<Block>> DltSft_share;
};

struct DeltaShiftShareResult {
    Block    delta_share;
    std::vector<std::vector<Block>> DltSft_share;
};

struct DPFKey {
    int   b;
    int   n;
    int   w;
    Block root;
    std::vector<std::vector<Block>> cw;  // cw[i][k], k ∈ [0, m-1]
};

// ---- PRG type selection ---------------------------------------------------
enum class PRGType { AES, CHACHA8 };

// ---- DPF full-eval backend ------------------------------------------------
// CPU: the reference tree expansion. GPU: device full-domain evaluation
// (common/dpf_gpu.cu), available when built with PCG_ENABLE_CUDA; both AES
// and ChaCha8 PRGs have bit-exact device implementations. Selectable so the
// CPU path stays available for comparison.
enum class EvalBackend { CPU, GPU };

// ---- class ---------------------------------------------------------------
// Encapsulates per-party state: tree PRG, NetIO channel, and parameters (n, w).
class HalfTreeDPF {
public:
    // `master_key` seeds the PRG.  For AES: derives (m-1) AES sub-keys.
    // For ChaCha8: unused (parent node is the ChaCha key).
    HalfTreeDPF(int party, emp::NetIO* io, int n, int w,
                const Block& master_key, PRGType prg_type = PRGType::CHACHA8);

    HalfTreeDPF(const HalfTreeDPF&)            = delete;
    HalfTreeDPF& operator=(const HalfTreeDPF&) = delete;

    // Secure two-party computation of DltSft shares using Boolean circuits.
    // No information about α or Δ is revealed (except lsb(Δ)=1 which is public).
    DeltaShiftShareResult F_DeltaShiftShare(uint64_t alpha_share, Block delta_share,
                                            bool_circuit::TripleGen& tg);

    // Batch version: compute B independent DltSft results in ONE circuit
    // evaluation.  Same AND depth as the single version, but all B instances
    // share the same communication rounds.
    std::vector<DeltaShiftShareResult> batch_F_DeltaShiftShare(
        const std::vector<uint64_t>& alpha_shares,
        std::vector<Block>& delta_shares,
        bool_circuit::TripleGen& tg);

    PartyInput setup_params(const Block& delta_share,
                            std::vector<std::vector<Block>> DltSft_share) const;

    DPFKey gen(const PartyInput& inp);

    // Combined Gen + full Eval: returns the N = 2^n leaf shares directly.
    // Equivalent to gen() followed by eval(key, j) for all j ∈ [0, N),
    // but avoids constructing the key and re-traversing the tree.
    std::vector<Block> gen_full_eval(const PartyInput& inp);

    // Batch B independent DPF full evaluations. All instances must share
    // the same (n, w). At each tree level, ALL B*(m-1) CW shares are
    // exchanged in a single round trip, reducing communication rounds
    // from B*levels to just levels.
    // Returns: result[b] = N-element leaf vector for instance b.
    std::vector<std::vector<Block>> batch_gen_full_eval(
        const std::vector<PartyInput>& inps);

    // Device-resident batch full eval: identical protocol/PRG semantics to
    // batch_gen_full_eval, but the leaves stay in GPU memory. Returns B rows
    // of stride D = 2^n (aliases the dpf_gpu buffer pool; valid until the
    // next dpf_gpu_* call). Consume with common/leaf_convert_cuda.h. Requires
    // PCG_ENABLE_CUDA + a CUDA device (throws otherwise).
    const pcg_cuda::DpfBlk* batch_gen_full_eval_device(
        const std::vector<PartyInput>& inps);

    Block eval(const DPFKey& key, uint64_t j) const;

    // Select the full-domain evaluation backend used by batch_gen_full_eval.
    // GPU requires PCG_ENABLE_CUDA (AES and ChaCha8 both have device PRGs);
    // otherwise it falls back to CPU.
    void        set_eval_backend(EvalBackend b) noexcept { eval_backend_ = b; }
    EvalBackend eval_backend() const noexcept { return eval_backend_; }

    int         party() const noexcept { return party_; }
    int         n()     const noexcept { return n_; }
    int         w()     const noexcept { return w_; }
    emp::NetIO* io()    const noexcept { return io_; }

private:
    // GPU full-domain evaluation (defined in dpf.cpp under PCG_ENABLE_CUDA).
    std::vector<std::vector<Block>> batch_gen_full_eval_gpu(
        const std::vector<PartyInput>& inps);

    int                       party_;
    int                       n_;
    int                       w_;
    emp::NetIO*               io_;
    Block                     master_key_;   // seeds the AES tree PRG (for GPU path)
    PRGType                   prg_type_;
    EvalBackend               eval_backend_ = EvalBackend::CPU;
    std::unique_ptr<TreePRG>  prg_;  // pluggable tree expansion PRG
};

}  // namespace half_tree_dpf

#endif  // HALF_TREE_DPF_DPF_H__
