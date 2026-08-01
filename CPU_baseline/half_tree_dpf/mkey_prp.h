// (m-1)-key AES CCR hash in pipeline.
// Inspired by emp::TwoKeyPRP in emp-ot/ferret/twokeyprp.h.
//
//   H_k(x) = AES_{keys[k]}(x) ⊕ x,   k ∈ [0, num_keys)
//
// Pipelining: for each parent block, the num_keys encryptions (same input,
// different keys) are interleaved across the 10 AES rounds via
// emp::ParaEnc<num_keys, 1>. The CPU executes aesenc instructions back-to-back
// without waiting for the previous round's output, hiding the AES latency.

#ifndef HALF_TREE_DPF_MKEY_PRP_H__
#define HALF_TREE_DPF_MKEY_PRP_H__

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <emp-tool/utils/aes.h>
#include <emp-tool/utils/aes_opt.h>
#include <emp-tool/utils/block.h>

namespace half_tree_dpf {

class MKeyPRP {
public:
    explicit MKeyPRP(const std::vector<emp::block>& user_keys)
        : num_keys_(static_cast<int>(user_keys.size())) {
        if (num_keys_ < 1)
            throw std::runtime_error("MKeyPRP: num_keys must be ≥ 1");
        aes_keys_.resize(static_cast<size_t>(num_keys_));
        for (int k = 0; k < num_keys_; ++k)
            emp::AES_set_encrypt_key(user_keys[k], &aes_keys_[k]);
    }

    // Compute H_k(parents[j]) for j ∈ [0, n_parents), k ∈ [0, num_keys),
    // writing to out[j*num_keys + k].  Pipelined for common (m-1) ∈ {1,3,7,15}.
    void hash(emp::block* out, const emp::block* parents, size_t n_parents) const {
        switch (num_keys_) {
            case 1:  hash_pipelined<1>(out, parents, n_parents);  break;
            case 3:  hash_pipelined<3>(out, parents, n_parents);  break;
            case 7:  hash_pipelined<7>(out, parents, n_parents);  break;
            case 15: hash_pipelined<15>(out, parents, n_parents); break;
            default: hash_generic(out, parents, n_parents);       break;
        }
    }

    int num_keys() const noexcept { return num_keys_; }

private:
    // Pipeline all num_keys encryptions of one parent across AES rounds.
    template<int NumKeys>
    void hash_pipelined(emp::block* out, const emp::block* parents,
                        size_t n_parents) const {
        emp::block tmp[NumKeys];
        auto* keys = const_cast<emp::AES_KEY*>(aes_keys_.data());
        for (size_t j = 0; j < n_parents; ++j) {
            for (int k = 0; k < NumKeys; ++k) tmp[k] = parents[j];
            emp::ParaEnc<NumKeys, 1>(tmp, keys);
            for (int k = 0; k < NumKeys; ++k)
                out[j * NumKeys + k] = tmp[k] ^ parents[j];
        }
    }

    // Fallback for non-standard widths — no pipelining.
    void hash_generic(emp::block* out, const emp::block* parents,
                      size_t n_parents) const {
        for (size_t j = 0; j < n_parents; ++j) {
            for (int k = 0; k < num_keys_; ++k) {
                emp::block c = parents[j];
                emp::AES_ecb_encrypt_blks(&c, 1,
                    const_cast<emp::AES_KEY*>(&aes_keys_[k]));
                out[j * static_cast<size_t>(num_keys_) + k] = c ^ parents[j];
            }
        }
    }

    int                       num_keys_;
    std::vector<emp::AES_KEY> aes_keys_;
};

}  // namespace half_tree_dpf

#endif  // HALF_TREE_DPF_MKEY_PRP_H__
