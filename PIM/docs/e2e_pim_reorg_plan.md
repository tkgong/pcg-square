# e2e branch: new PCG CPU impl + PIM/GPU skeleton + data-reorg FU

Branch `e2e` is cut from `tkgong` (which already carries the PIM simulator,
the trace toolkit, and `endtoend/run.sh` + the GPU baseline_v2 kernels). This
branch swaps the CPU PCG implementation to the **Beaver-corrected output
layer** (no per-leaf modular multiply) and adds a **data-reorg FU** to the PIM
so DPF-expand output is consumed efficiently by the SM.

---

## 1. What changed in the CPU PCG (the load-bearing algorithmic change)

The leaf → field conversion was rewritten. Old vs new **per-leaf** work:

| | OLD (main / origin) | NEW (this branch) |
|---|---|---|
| CW | `CW = β · v⁻¹` (mod-inverse + modmul, per instance) | `CW = A·B` via Beaver triple (per instance) |
| per-leaf scaling | `coeff = v · CW` — **one 62-bit modmul / leaf** | `y = C + (τ ? CW : 0)` — **conditional add, no mul** |
| per-leaf convert | value already in F | `C = Convert(H′(leaf))` — one PRP hash + one reduce |
| control bit | — | `τ = lsb(leaf)` |

Modmul count drops from **O(leaves) = 2N·t·c²** to **O(instances) = t²·c²**
(the Beaver reconstruction `dv·s + ev·r + rs (+ dv·ev)`), a factor of ~2·bin_sz
fewer. The math is the standard BGI "Convert + control-bit" DPF→field trick:
τ ∈ {0,1}, so "scale by CW" is a conditional add, and the single field
multiply per instance computes `CW = (Στ)·(ΣC + β)` once in the clear.

---

## 2. Change points required to build the new CPU impl on this branch

The pasted `pcg_ole_impl.h` pulls three things `tkgong` did not have. Status:

| # | Change point | File | Status on e2e |
|---|---|---|---|
| 1 | **`TripleGen::zp_triple`** (batched Z_p Beaver triples) | `bool_circuit/circuit.{h,cpp}` | **ADDED** (2× `zp_multiply` cross-terms + local product). Flagged for crypto review. |
| 2 | **Profiling layer** `PCG_PROF_TIC/TOC/ZONE` + zones (`PREPROCESSING`, `DPF_EXPAND_LOCAL`, `DPF_EXPAND_COMM`, `NET_WAIT`, `STEP4`, …) | `common/prof.h` | **ADDED as shim** — real timers, replace with the dev-tree prof.h when it lands. |
| 3 | **Include path** `pcg/common/prof.h` → `common/prof.h` | `pcg_ole_2pc/pcg_ole_impl.h` | **RECONCILED** to the flat `common/` layout (dev tree nests under `pcg/`). |
| 4 | `dpf.batch_gen_full_eval` internal probes (`dpf_expand_communication/_local_computation`) referenced in the comment | `half_tree_dpf/dpf.cpp` | **not required to compile** — the impl only relies on the return value. Add the internal PCG_PROF probes if you want the DPF-internal split. |
| 5 | `FFp::raw`, `FFp::zero/one`, `MKeyPRP`, `get_lsb`, `zp_multiply`, `batch_F_DeltaShiftShare`, `setup_params` | ffp.h / mkey_prp.h / dpf | **present already** — no change. |

Build note: `zp_triple`'s local product uses `unsigned __int128` (product of
two <2⁶² operands). No new external deps.

Not yet verified: end-to-end correctness (`z0+z1 = x0·x1`) with the new output
layer on this branch — gated on emp-tool/emp-ot building on the host. This is
the **first verify step** before any timing is quoted.

---

## 3. PIM + GPU skeleton carried over (unchanged from tkgong)

- **PIM sim**: `pim/sim/` (Ramulator2/AiM, GGM_EXTEND/REDUCE, all-bank
  broadcast, MAU decoupled data-bus path — commit b942075). Trace toolkit
  `pim/tools/gen_dpf_tree_trace.py`, collector `finalize_campaign.py`.
- **GPU**: `endtoend/run.sh` (unified single-GPU battery, per-GPU time),
  `endtoend/run_baseline_grid.sh`, `common/dpf_gpu.{cu,h}` device-resident
  leaf path, `common/leaf_convert_cuda.{cu,h}`.

---

## 3.5 All-ChaCha pipeline (GGM tree + out_hash), CPU/GPU aligned — DONE

Both cryptographic primitives now default to ChaCha8; AES stays selectable
(`--prg aes`):

- **Tree PRG (GGM)**: `PRGType` default flipped to `CHACHA8` everywhere
  (PCGParams, HalfTreeDPF ctor, test/bench defaults). The ChaCha8 tree paths
  (CPU `ChaCha8TreePRG`, device `hash_kernel_chacha8`) already existed and
  are bit-exact mirrors: key = parent ‖ 0^128, **nonce = 0**, ctr from 0.
- **out_hash H′**: replaced the AES `MKeyPRP` with `ChaCha8OutHash`
  (half_tree_dpf/tree_prg.h): H′(x) = first 16 B of ChaCha8 keystream,
  key = x ‖ 0^128, **nonce = OUT_DOMAIN = 0x6f75745f68617368 ("out_hash")**,
  ctr = 0. Domain separation from the tree = nonce 0 vs nonzero. Same
  parent-as-key correlation-robustness assumption as every tree level.
- **GPU output layer rewritten** (common/leaf_convert_cuda.{h,cu}) for the
  Beaver scheme: `dpf_out_sums` (per-leaf ChaCha8 H′ + mod-P convert + τ;
  raw ΣC/Στ per instance, sign-free) and `dpf_out_scatter_g`
  (y = C + τ?CW, party sign, negacyclic fold, CAS mod-add). **No modular
  multiply on the device** — the GPU-NTT Barrett dependency is gone from
  this unit. Party signs and +β stay on the host, exactly where the CPU
  path applies them.
- **impl wiring**: `pcg_ole_impl.h` gained the `PCG_ENABLE_CUDA` device
  fast path (device-resident leaves → dpf_out_sums → two Beaver opens →
  dpf_out_scatter_g → folded g), mirroring the old gpu-ntt-pivot wiring.

Verification (L40S host, CUDA 12.6, emp-tool/emp-ot at Dockerfile-pinned
SHAs, patched GPUNTT 95c739c):
- V3 bit-exactness (test/out_hash_cuda_bitexact.cu, no emp needed):
  host ChaCha8OutHash == device kernels (sums + scatter, both parties) PASS.
- V2 CPU protocol: z0+z1 = x0·x1 PASS at N=128/t=4 and N=4096/t=8
  (all-ChaCha defaults); half-tree DPF point-function batch ALL PASS.
- V4 GPU protocol: `--dpf gpu` PASS at N=4096/t=8 and N=16384/t=16.

PU ISA consequence (unchanged from §"hash 需要多支持哪些操作"): the ChaCha
out-hash reuses the existing ARX datapath — zero new functional units; only
the multiplier-free sparse-prime mod-ALU (REDP/ADDP/CADDP/NEGP) is new.

## 4. The new PIM data-reorg FU (the design ask)

### 4.1 Why leaf-convert changing the game changes the FU

With the per-leaf modmul gone, the PIM does **only ARX** for both expand and
convert (`H′` is the same block-cipher core as the tree PRG; `Convert` is one
mod-p reduce; `τ?+CW` is the existing VCXOR). There is no longer any modular
multiplier to justify on the PIM side. The remaining inefficiency is **not
compute — it is layout**: the bank PU writes leaves in PU-local, word-major,
ping-pong-row order, which is exactly the order an SM warp reads *badly*
(strided, uncoalesced, 16 B/leaf when the SM only needs 8 B). The FU to add
is therefore a **data-reorg unit (DRU)**, not an arithmetic unit.

### 4.2 What the DRU does (fixed-function, no multiplier)

Sitting on the store path between the PU's write-back and the DRAM array, per
column group the DRU performs:

1. **Truncate 16 B → 8 B**: keep only the low 64 b the SM consumes (`C` +
   control bit); the 128 b seed stays PU-local for the next level. Halves both
   write traffic and later SM read traffic.
2. **De-interleave (word-major → element-linear)**: the PACK step already
   interleaves left/right children; the DRU instead lays leaves out in the
   **global linear leaf index** order, so leaf `j` lands at the (ch, bank, row,
   col) that the memory controller's channel-striping maps to SM-linear address
   `j`. A warp of 32 consecutive leaves then hits one 128 B cache line.
3. **Partial-sum tap (ARX add only)**: as leaves stream past, accumulate the
   per-instance `Σ τ` and `Σ C` (32-bit limb adds) that the output layer needs
   anyway — one 128 b partial sum per instance leaves the bank instead of the
   whole `2Nt`-leaf stream. This removes the SM's first read pass entirely.
4. **Doorbell emit**: at each channel-round boundary, write a completion word
   to a fixed address so an SM persistent kernel can consume round r−1 while
   the PU writes round r (round-level ping-pong pipeline).

The DRU is fixed-function: a byte-select mux (truncate), an address-remap AGU
(de-interleave), a small adder tree (partial sum), and a completion-word write.
No multiplier, no SRAM cache beyond a column-wide staging latch.

### 4.3 Placement: per-bank vs per-channel

| | **per-bank DRU** (one per PU) | **per-channel DRU** (one per channel) |
|---|---|---|
| Count | 192–2048 (= PU count) | 24–128 (= channels) |
| Sees | one PU's leaves, already in VRF | must re-read leaves off GIO |
| Truncate/partial-sum | free — leaves are in the store epilogue, ALU idle | needs its own read of the just-written row |
| De-interleave scope | intra-PU only (2 banks) → cannot place leaf `j` on another channel's stripe | full channel → can lay out across all banks of the channel = the SM stripe granularity |
| Area | tiny × many (byte mux + adder tree + AGU) | bigger × few, + GIO read port |
| Contends with EXTEND | no (epilogue slack) | yes (shares GIO with the MAU-class data-bus path) |

**Recommendation: split the DRU across both tiers.**
- **Truncate + partial-sum → per-bank**, folded into the store epilogue. These
  are element-local and free (ALU idle during write-back; validated pattern —
  same slack the LSU staging already uses). Costs 1 accumulator register
  (borrow from SRF const half or shrink store queue 16→15).
- **De-interleave (linear-order scatter) + doorbell → per-channel**, on the
  same channel-level data-bus path the (now-retired) MAU used. Only the
  per-channel unit sees enough address space to place leaf `j` on the SM's
  cross-bank stripe; doing it per-bank cannot reach another bank's columns.

This is the minimum hardware that makes PIM output SM-friendly: the arithmetic
(truncate, sum) rides free on the bank PU, and only the **cross-bank
reordering** — which genuinely needs channel scope — costs a channel-level FU.
It reuses the MAU's channel data-bus slot, so it is not new bandwidth, just a
different (cheaper, multiplier-free) consumer of it.

### 4.4 Expected effect (to be measured — three sim points)

- PIM residency ≈ `none` + a few % epilogue (below `fused`'s 1.24×none, and no
  per-bank multiplier) — the DRU adds no modmul.
- SM consume: one coalesced 8 B/leaf streaming pass + O(instances) partial sums
  instead of two 16 B/leaf uncoalesced passes.
- Contention: the per-channel de-interleave stream shares the data bus with the
  PU write-back — this is the `--sm-stream` contention class; with 8 B
  truncation the injected volume halves vs the old mau_sm point.

**Sim points to run** (generator + Ramulator2, same battery as v2):
1. `none + dru_stream` (8 B truncated linear-scatter volume) → PIM slowdown =
   the only non-free term.
2. last-level epilogue `cl+Δ` point → confirm truncate+partial-sum hide in the
   store window.
3. round-level double-buffered trace → SM-lane / PIM-lane overlap efficiency
   (doorbell pipeline).

If those hold, this is a **case D** (no MAU, no per-bank multiplier, DRU-reorg)
that should sit at or below case C on PIM residency while giving the SM the
cheapest possible consume path. Add it as a column to the speedup grid.
