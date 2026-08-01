// Abstract PRG for DPF tree expansion.
//
// Given a parent node (128-bit block), produce (m-1) pseudorandom children.
// Two backends:
//   AESTreePRG    — AES-based CCR hash via pipelined ParaEnc (MKeyPRP)
//   ChaCha8TreePRG — ChaCha8 with parent as key, split 64-byte blocks
//                    into 128-bit children

#ifndef HALF_TREE_DPF_TREE_PRG_H__
#define HALF_TREE_DPF_TREE_PRG_H__

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <emp-tool/utils/block.h>

#include "common/chacha8.h"
#include "half_tree_dpf/mkey_prp.h"

namespace half_tree_dpf {

// ---- Abstract interface ---------------------------------------------------

class TreePRG {
public:
    virtual ~TreePRG() = default;

    // For each parent[j] (j = 0..n_parents-1), produce num_children()
    // pseudorandom 128-bit blocks.
    // Layout: out[j * num_children() + k] = H_k(parent[j]).
    virtual void hash(emp::block* out, const emp::block* parents,
                      size_t n_parents) const = 0;

    virtual int num_children() const = 0;   // m - 1
};

// ---- AES backend (existing MKeyPRP wrapper) --------------------------------

class AESTreePRG : public TreePRG {
public:
    // Derive (m-1) sub-keys from master_key via MMO.
    AESTreePRG(int num_keys, const emp::block& master_key) {
        emp::AES_KEY master;
        emp::AES_set_encrypt_key(master_key, &master);

        std::vector<emp::block> sub_keys(num_keys);
        for (int k = 0; k < num_keys; ++k) {
            emp::block tweak = emp::makeBlock(0, static_cast<uint64_t>(k));
            emp::block c = tweak;
            emp::AES_ecb_encrypt_blks(&c, 1, &master);
            sub_keys[k] = c ^ tweak;
        }
        prp_.reset(new MKeyPRP(sub_keys));
    }

    void hash(emp::block* out, const emp::block* parents,
              size_t n_parents) const override {
        prp_->hash(out, parents, n_parents);
    }

    int num_children() const override { return prp_->num_keys(); }

private:
    std::unique_ptr<MKeyPRP> prp_;
};

// ---- ChaCha8 backend -------------------------------------------------------
//
// PRG(parent) = ChaCha8_{parent ‖ 0^128}(counter=0,1,...)
//
// Each ChaCha8 block is 64 bytes = 4 × 128-bit children.
//   w=1 (m-1=1): 1 block, use chunk 0.
//   w=2 (m-1=3): 1 block, use chunks 0-2.
//   w=3 (m-1=7): 2 blocks, use chunks 0-6.
//   w=4 (m-1=15): 4 blocks, use chunks 0-14.

class ChaCha8TreePRG : public TreePRG {
public:
    explicit ChaCha8TreePRG(int num_keys)
        : num_keys_(num_keys),
          blocks_per_parent_((num_keys + 3) / 4) {}

    void hash(emp::block* out, const emp::block* parents,
              size_t n_parents) const override {
        for (size_t j = 0; j < n_parents; ++j) {
            // Use parent as ChaCha8 key (128 bits zero-padded to 256 bits).
            uint8_t key[32];
            std::memset(key, 0, 32);
            std::memcpy(key, &parents[j], 16);

            ChaCha8PRG prg(key);

            // Generate enough 64-byte blocks.
            uint8_t buf[256];   // up to 4 blocks = 16 children (covers w≤4)
            prg.fill_blocks(buf, blocks_per_parent_);

            // Extract 128-bit children.
            for (int k = 0; k < num_keys_; ++k)
                std::memcpy(&out[j * num_keys_ + k], buf + k * 16, 16);
        }
    }

    int num_children() const override { return num_keys_; }

private:
    int num_keys_;
    int blocks_per_parent_;
};

// ---- ChaCha8 output-layer hash ---------------------------------------------
//
// H'(x) = first 16 bytes of ChaCha8 keystream with key = x ‖ 0^128,
//         nonce = OUT_DOMAIN, counter = 0.
//
// Same parent-as-key construction (and hence the same correlation-robustness
// assumption) as ChaCha8TreePRG, but DOMAIN-SEPARATED from the tree levels:
// the tree PRG runs with nonce = 0, this hash runs with a fixed nonzero
// nonce, so the two keystreams are independent even for identical inputs.
// Used by the PCG-OLE output layer (leaf -> field conversion): C = low 64
// bits of H'(leaf) reduced mod P; the control bit tau = lsb(leaf) itself.
//
// Interface mirrors MKeyPRP::hash with num_keys = 1 so the output layer can
// swap between the AES CCR hash and this one with a one-line change.

class ChaCha8OutHash {
public:
    // Fixed domain-separation nonce ("out_hash" in ASCII). Any nonzero
    // constant works; it must only differ from the tree PRG's nonce (0).
    static constexpr uint64_t OUT_DOMAIN = 0x6f75745f68617368ULL;

    void hash(emp::block* out, const emp::block* leaves, size_t n) const {
        for (size_t j = 0; j < n; ++j) {
            uint8_t key[32];
            std::memset(key, 0, 32);
            std::memcpy(key, &leaves[j], 16);

            ChaCha8PRG prg(key);
            prg.set_nonce(OUT_DOMAIN);

            uint8_t buf[64];
            prg.fill_blocks(buf, 1);
            std::memcpy(&out[j], buf, 16);
        }
    }
};

}  // namespace half_tree_dpf

#endif  // HALF_TREE_DPF_TREE_PRG_H__
