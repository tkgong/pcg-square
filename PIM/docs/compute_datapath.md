# ChaCha8-DPF PU datapath: seed-parallel vs word-parallel (making compute_latency real)

Purpose: the `compute_latency` of `GGM_EXTEND` was previously a hand-derived 274 — this document accounts for it instruction-by-instruction under two SIMD layouts,
using the **bit-exact implementation of PIMSimulator** as ground truth (it is bit-for-bit consistent with the CPU reference), corrects it to **282**,
and gives the word-parallel comparison. PU = 256b = 8 lane × 32b, single-issue (1 instruction/cycle, no pipelining/no overlap).

## 1. Seed-parallel (baseline, = PIMSimulator's implementation)

**Layout**: 8 seeds placed in 8 lanes; ChaCha's 16 state words placed in 16 vector registers (the 8 copies of word j sit in the 8 lanes of register j).
One instruction advances the same word of all 8 seeds simultaneously. Same as PIMSimulator (the lane loop in PIMBlock.cpp:168-197;
GRF_A[0..7]=words 0-7, GRF_B[0..7]=words 8-15).

**Each quarter-round = 8 instructions** (PIMChaCha8DemoTest.cpp:186-196, the fused XORROTL is already its real ISA):
```
ADD32 a,a,b ; XORROTL16 d,d,a ; ADD32 c,c,d ; XORROTL12 b,b,c
ADD32 a,a,b ; XORROTL8  d,d,a ; ADD32 c,c,d ; XORROTL7  b,b,c
```

**Instruction count for one batch (8 seeds)**:
| Segment | Count | Basis |
|---|---|---|
| ChaCha8 core: 32 QR × 8 | **256** | 8 rounds × 4 QR; PIMBlock.cpp:178-187 |
| feed-forward (16 words added back to initial state) | **16** | PIMBlock.cpp:191-195 |
| **PACKLR32 × 8 (cross-lane child interleaving)** | **8** | PIMBlock.cpp:236-246 —— reorders the left/right children of the 8 lanes into the word-major format for writeback. **The old 274 missed this item** |
| DPF: VCXOR × 2 (one CW applied to each of the left/right children) | **2** | the cond_xor in dpf.cpp:311-315 |
| **Total → compute_latency** | **282** | |

**Register budget**: 16×256b (state) + SRF (constant/CW broadcast) = **17 = all registers of the PIM block, zero headroom**
(PIMBlock.h:37-41). The feed-forward's initial state is not stored in a second copy: the const half resides in SRF, and the seed half is re-read from the input buffer
(PIMSimulator's ggmExtend uses exactly this method). I/O: **read 8 columns** (seed half; const half is in SRF) + **write 16 columns**.

## 2. Word-parallel (user slide convention, parsed comparison)

**Layout**: the 16 words of a single seed laid out across columns (2 seeds/bank), 512b state = 2×256b blocks; on round rotation the words must be shifted cross-lane
→ needs **vec_rot 32/64/96 (cross-lane)**. Per "round" as given by the slide:
```
transfer 512b/256b = 2 cycles + odd-round compute 2×4 = 8 cycles + rotate 2×3 = 6 cycles + even-round 2×4 = 8 cycles = 24 cycles
```
**Round count check**: the 24 cycles **already contain both the odd+even rounds** (= 1 double-round). ChaCha8 = 8 rounds = **4
double-rounds** → the correct total should be **24×4 = 96 cycles/2 seeds** (the slide takes ×8 to get 192, multiplying the double-round by the round count again,
**overcounting by 2×**; its derived 0.75/0.375 ns throughput is correspondingly conservative by 2×). Take the corrected value 96 (including transfer, excluding ff/DPF).

## 3. Comparison (single-issue, 1 instruction per cycle)

| | seed-parallel (baseline) | word-parallel (after correction) |
|---|---|---|
| how many seeds per pass | 8 | 2 |
| cycles/pass | 282 (core 256+ff16+PACK8+VCXOR2) | ~96(+ff/DPF ~12) ≈ 108 |
| **cycles/seed** | **35.25** | **~54** (slide's uncorrected convention gives ~96) |
| cross-lane demand | only PACKLR32 (8 instructions/batch, writeback reorder) | **6 vec_rot per double-round** (in the hot loop) |
| registers | 16×256b + SRF (exactly full) | ~8×256b/2seeds (more frugal) |
| column reads/batch | 8 (const in SRF) | 16 (the entire 512b state over the bus) + transfer per round |

**Conclusion**: seed-parallel is ~1.5× cheaper per seed (~2.7× vs the slide's uncorrected convention), and it pushes cross-lane out of the hot loop
(vec_rot sits inside every double-round in word-parallel, whereas seed-parallel only has PACKLR32 at the tail); the cost is filling all 17 registers.
**Chosen seed-parallel, cl=282**. Sensitivity simulation runs cl∈{0,282}.

## 4. ISA revision (relative to the previous report)

| Instruction | lane attribute | cost | frequency | notes |
|---|---|---|---|---|
| **PACKLR32** (newly included) | **cross-lane** | medium (reorder network) | 8/batch | required for GGM child word-major writeback; missed in the previous version |
| VCXOR | per-lane | cheap ARX | 2/batch | DPF CW application (the only DPF-specific addition) |
| vec_rot 32/64/96 | cross-lane | expensive | 0 (seed-parallel does not need it) | only word-parallel needs it —— not adopted |
| the rest (VADD32/VXROL/VXOR/VLD/VST/VBCAST…) | same as previous version | | | |

**Net effect**: cl 274→282 (+2.9%), the single-batch latency/throughput numbers are recomputed with 282 (see e2e report);
memory access is 8 reads + 16 writes per batch (const-fold already included: the const half resides in SRF, so reads are 8 not 16).

## 5. dpf.cpp alignment convention (--seed-bits 128, benchmark default)

The reference implementation's node is a **128b `emp::block`** (not 256b): ChaCha zero-pads the 128b parent into a key
(tree_prg.h:89-91), and with w=1 only chunk0 of the output is used; the second child = h ⊕ parent (half-tree free child,
dpf.cpp:314), **not taken from the PRG output**. After alignment, the accounting per batch (8 seeds):

| Segment | 256b variant | **128b aligned (default)** |
|---|---|---|
| ChaCha8 core + ff | 256 + 16 | 256 + 16 (block-function count unchanged) |
| PACKLR32 | 8 (8 words/child) | **4** (4 words/child) |
| VCXOR | 2 | 2 |
| free child h⊕parent | — (both children taken from output) | **+1 VXOR** |
| **cl total** | **282** | **279** |
| memory access/batch | 8 reads + 16 writes | **4 reads + 8 writes** (halved) |

### 5.1 Final instruction-level correction (cl = 272, replaces 279)

Recounting precisely at register granularity (the output only consumes the first 128b of keystream; CW=128b=4 words; with w=1 the two children share the same CW,
so R = L ⊕ parent does not need a second CW application):
| Segment | 279 version | **272 final version** |
|---|---|---|
| ARX core | 256 | 256 |
| feed-forward | 16 | **4** (only add the consumed w0-3) |
| CW application | 2 | **4 VCXOR** (the 4 word registers of L) |
| free child | 1 | **4 VXOR** (R = L⊕parent, CW already folded in) |
| PACKLR32 | 4 | 4 |
| **cl** | 279 | **272** (+12 VBCAST for initialization, giving 284 if made explicit) |

**Register pressure and re-read scheme**: the state is scrambled in place within the round, and child R needs the original parent — keeping it requires 20 registers (>17, does not fit).
Tradeoff: (B) **re-read 4 columns from the still-open row** (+16 CK, memory access becomes 8 reads/8 writes, register peak 16+SRF ✓);
(C) the PIMSimulator way = GRF plus staging SRAM (ggmInput/ggmLow/ggmHigh, PIMBlock.h:51-54 ——
it adds these buffers precisely because the GRF cannot hold everything). This design takes (B). The full instruction-by-instruction sequence is in the e2e report/session log.

### 5.2 leaf_convert re-modeling (two passes + VXSUM + modmul=32, SIMD-honest)

Two new instructions added (used only by leaf_convert, the ChaCha expansion does not need them):
| Instruction | semantics | lane | cost |
|---|---|---|---|
| **VMUL32** (implicit in the modmul sequence) | 32b multiply, Barrett 62b modmul = **a 32-cycle lane-local sequence** | per-lane | multiplier (the only place in the PU that needs multiplication) |
| **VXSUM Vd,Va** | horizontal reduction of 8 lane → 1 (tree adder) | **cross-lane** | 4 cycles; only 1 time per subtree sum at the tail |

**Two-pass structure** (consumes only the low 64b of each leaf = 2 columns per group of 8 leaves, pcg_ole_impl.h:171-232):
- **Pass1 summation**: read L/4 columns + ADD64M(4 cycles) per group + 1×VXSUM(4) at the tail → S_k;
  [host: swap S, compute CW=β/v, t values 1 round, does not enter sim]
- **Pass2 scale+scatter**: re-read L/4 columns + **modmul 32 cycles per group (8 leaf lanes in parallel!)** + ADD64M + write g(2 columns/group).

**Accounting correction**: modmul is a lane-local 32b instruction **sequence** — the 8 leaves are computed in the 8 lanes simultaneously,
32 cycles/group = **4 cycles/leaf effective**; the earlier "per leaf × modmul_cl" accounting missed the SIMD, overestimating by 8×.
Measured results in results/tree_sweeps_leafconv32.txt.

**Chained correction (important)**: leaf_convert's modmul is **once per 128b leaf block**
(`block_to_uint64→Z_p`, pcg_ole_impl.h:179/224), not once per 32b word — the earlier fused REDUCE
counted each col-word as 8 modmuls, **overcounting by 8×** (the "REDUCE=expand×3.3" conclusion for n=16 under the 256b convention
is thereby invalidated). After alignment, each w-column chunk = w·8/4 leaves → cl = 2w·modmul. The corrected measured results are in
`results/tree_sweeps_128b.txt` and §④ of the e2e report.
