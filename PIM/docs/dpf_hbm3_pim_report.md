# Design + Evaluation: Full DPF (GGM/ChaCha8) latency/throughput on HBM3-PIM

## ▶ Current iteration: 64-channel PIM configuration (1 rank/ch, 2 banks/rank sharing 1 PU, instruction-level parallelism)

### Context
Scale the validated single-channel DPF simulation to the target architecture: **64 channel × 1 rank × 2 bank, every 2 banks sharing 1 PU = 64 PU**,
GGM tree batch count > NCh (each channel gets multiple batches), verify whether instruction-level parallel dispatch holds, find the system-level bottleneck.
Already verified: the dispatcher itself supports dispatching up to `#controllers` ISRs in parallel per cycle (aim_DRAM_system.cpp:230-231),
GGM ops use direct channel indexing (:295, upper bound MAX_CHANNEL_COUNT=128) → instruction-level parallelism **needs no code change**, only config + trace.

### Changes (in /home/tkgong/PCG-acceleration/pim/sim)
1. **`src/dram/impl/HBM2.cpp`** — org preset `PIM64_2Banks = {2<<10, 128, {64, 1, 1, 2, 1<<17, 1<<6}}`
   (Ch=64, Pch=1, Bg=1, Ba=2, Ro=2^17, Co=64; per-channel density 2048 Mb → density_id=0 valid;
   `pim_unit_of`: flat_bank=bk, unit=bk/2 → **exactly 1 PU per channel**). **This preset is already written to the file, pending compilation.**
2. **`test/hbm3_dpf_64ch.yaml`** (new) — org preset=PIM64_2Banks + timing preset=HBM3_6400Mbps,
   Controller uses `fpu_gate_issue=true` (single-issue serial PU, one instruction at a time, no overlap — the correct model).
3. **trace generation** (script / one-off python) — B batches (B=1024 > 64), round-robin `ch = batch % 64`,
   bank=0, write_bank=1 (double buffering), **a different row per batch within a channel** (avoids the previously discovered address-merging artifact),
   `GGM_EXTEND opsize=8 cl=274`, SYNC+EOC at the end. Saved to `test/dpf_64ch_1024.trace`.

### Experiments (what to run, what to measure)
- **A scalability**: B=1024, 64ch vs 1ch (same B) → speedup; ideal = per channel 16×274 ≈ 4384 cyc total duration.
- **B compute sensitivity**: cl ∈ {0, 274, 1000} @64ch → determine whether the system level is compute-bound (total duration ≈ B/64×cl)
  or dispatch/frontend-bound (insensitive to cl / total duration ≈ B×constant).
- **C compute fraction**: per-batch = compute + memory access/ACT, measure each fraction.
- Record per-batch aggregate cycle, per-channel active/idle, CH0 vs CH63 balance.

### Deliverables (report after runs)
1. **Where the code changed**: list per-file/per-line (including the previous round's HBM3_6400Mbps preset and the fpu_gate_issue gating).
2. **Bottleneck determination**: let the A/B/C data speak (candidates: per-PU ChaCha compute 274/batch; frontend 1 op/cycle dispatch rate;
   DMA head-of-line (request_queue ordering, a single full channel stalls); per-channel command bus).
   Prediction: at cl=274 the consumption rate = 64/274 ≈ 0.23 op/cycle < 1 op/cycle dispatch rate → still **compute-bound**,
   64× nearly linear; at cl=0 the memory floor ≈ consumption 64/193 ≈ 0.33 op/cycle, may start to hit frontend/dispatch.
3. **How to optimize**: order the levers by the measured bottleneck (raise PU frequency/lower cl; more PUs per channel (bank↑);
   if dispatch-bound → batched ISR/wider host channel; pack multiple batches per row to save ACT, etc.).

### Verification
- Build with `cmake --build build -j`; first run 1-batch @64ch sanity (should be ≈335 CK), then run the A/B/C sweeps;
  compare results against the already-validated 1ch numbers (274.6/batch, floor 193), write back into this plan file.

### ✅ 64ch results (already run, B=1024 batches, 16 per channel)
**Model: single-issue serial PU — one PU executes only one instruction at a time; for one batch, read 16 columns → compute 274 cycles → write 16 columns
are all serial, the next batch must wait for the previous to finish. No pipeline, no overlap, no decoupled LSU.** (in sim via `fpu_gate_issue=true`)

| Experiment | Result | Reading |
|---|---|---|
| sanity 1 batch @64ch | **337 CK** | single-batch serial latency (read+compute+write+ACT) |
| A scalability (cl=274) | 1ch 428916 → 64ch **6536** = **65.6×** | **linear scaling**, 64-channel instruction-level parallelism holds |
| channel balance | CH0=CH31=CH63: GGM 2057 cyc / 33 ACT / idle 65 | **fully balanced**, dispatcher has no hotspot |
| **per-batch (cl=274)** | 6536/16 = **408 CK/batch** | compute 274 + memory access/ACT ~134, **serially added** |
| pure memory access (cl=0) | 3836/16 = **240 CK/batch** | read+write+ACT (no compute) |
| cl sensitivity | cl 0/274/1000 → 3836/6536/17426 | total duration tracks cl, compute is the largest single item |
| in-row packing (cl=274) | 5552/16 = **347 CK/batch** | 8 batch/read-row + 4 batch/write-row, saves ACT → 408→347 |
| dispatch utilization | 1024 op / 6536 cyc = 0.16 op/cyc ≪ 1 | frontend/DMA is **not** the bottleneck |

**Bottleneck = the 408 CK for one PU to serially execute one batch**, of which compute 274 (67%) + memory access/ACT ~134 (33%),
**both are on the critical path** (no overlap, memory access is not hidden). After linear 64-channel scale-up the system level is still this per-PU serial time.
**Optimization ordering** (under serial execution both memory access and compute count directly into per-batch, both worth shrinking):
① **in-row packed read/write + const-fold (16→8 reads)**: save ACT + halve column reads → per-batch 408→347→lower (directly shortens total duration);
② **shrink compute 274**: raise PU frequency, reduce instructions (VXROL fusion already used; fold const into SRF) — compute is 67%, the biggest lever;
③ **prefer adding channels to scale PUs** (measured linear 65.6×); adding banks within a channel is limited by the command bus (16-bank only 4.3×), poor cost-efficiency.

---

## ✅ Implementation + simulation validation in Ramulator2 (completed)
Landed and ran in `/home/tkgong/PCG-acceleration/pim/sim` (Ramulator2/AiM, with GGM_EXTEND/REDUCE, buildable in conda env pim):
- **Code changes**: (1) `src/dram/impl/HBM2.cpp` adds `HBM3_6400Mbps` timing preset (CK values from upstream hbm3.py, rate=3200→tCK=625ps);
  (2) `src/dram_controller/impl/AiM_dram_controller.cpp` adds `fpu_gate_issue` parameter + `ggm_fpu_ready()` gating
  (**single-issue serial PU: one instruction at a time, no overlap** — the correct model, enabled when running the simulation). Config `test/hbm3_dpf.yaml`,
  DPF batch = `GGM_EXTEND opsize=8 compute_latency=274`.
- **Validation results** (tCK=625ps, 1 PU, single-issue serial gated):
  | Quantity | Simulation | Reading |
  |---|---|---|
  | single-batch latency | **337 CK** | read+compute+write+ACT serial |
  | steady-state per-batch (cl=274) | **~418 CK** | compute 274 + memory access/ACT ~144, serially added |
  | pure memory access per-batch (cl=0) | **240 CK** | read+write+ACT, no compute |
- **New findings (correct/refine the plan)**:
  1. The early "memory-limited 66/batch" was a **column-address-reuse → read-merging trace artifact**; it disappeared after switching to distinct addresses.
  2. **Under single-issue serial, memory access is not hidden**: per-batch = compute(274) + memory access/ACT(~134-144) serially added ≈ 408-418, **both are on the critical path**.
  3. **Multiple banks within a single channel is not linear** (16-bank only 4.3×), limited by the command bus + shared FPU → **system throughput scales by adding channels**.
- **Not modeled inside the single-node sim** (still per §7.6 analytically): 16-channel aggregation, Gen two-party cross-die CW exchange (analytically as 31×RTT).

## Context
Extend the previous version's "GGM expansion only" PIM scheme into a **full 2-party DPF** (point function f_{α,β}, PRG=ChaCha8),
completing the ISA, and using **Ramulator2 HBM3** as the sole ground truth for latency/throughput. The DPF construction follows the local implementation
`/home/tkgong/PCG-acceleration/half_tree_dpf/dpf.cpp` (m-ary Half-Tree, GYW+22 + BGI).

**Core conclusion (stated first)**: EvalAll of the full DPF is still **compute-bound and almost as cheap as pure GGM**.
Reason (verified against dpf.cpp): the control bit = the **LSB** of the seed (dpf.h:27), CW is **one broadcast constant per level**,
applying CW = `cond_xor(t, CW[i][k])` is a **predicated XOR** (dpf.cpp:311-315), β/Δ is **folded into each level's CW** via `DltSft_share`
(dpf.cpp:354, dpf.h:40), leaves have no separate correction. **Eval has no cross-node/cross-lane reduction at all**
(agent verified: "Cross-lane reduction: NONE", dpf.cpp:307-316). ⟹ The full DPF only needs to add **one** cheap
lane-local instruction **VCXOR (predicated XOR)**; the cross-lane reduction / bit-field extract the prompt worried about are **both unnecessary**
(they only appear when embedding the DPF into PCG for leaf aggregation).

---

## 1. Parameter table (Ramulator2 HBM3, preset = HBM3_8Gb_8hi + HBM3_6400Mbps; same as the repo tests)
| Item | Value | Source |
|---|---|---|
| hierarchy | Channel→PseudoChannel→**Sid**→BankGroup→Bank→Row→Column | hbm3.py:13-21; HBM3.cpp:28 |
| PC/sid/BG/bank | 2 / 2(8hi) / 4 / 4 → **banks/ch = 2·2·4·4 = 64** (16hi sid=4 → 128) | hbm3.py:214,220 |
| row/column/dq | 16384 / 256 / 32 | hbm3.py:214 |
| prefetch | BL8 | hbm3.py:8 |
| **column granularity** | 8·32 = **256 bit = 32 B = 1 column = PU width** | — |
| **page/row** | 256÷BL8 = **32 columns × 256b = 1 KiB/bank** | hbm3.py:214 comment |
| channel count | 1 controller=1 channel; multi-channel at system level → **assume 16 ch/stack** | generic_dram_system.cpp:31; spec.py:20 |
| **tCK** | **625 ps → CK 1.6 GHz**; tick_mult=2 | hbm3.py:9,234 |
| timing (CK) | nBL2 nCL20 nRCDRD31 nRP26 nRAS45 **nRC72** nWR33 nRTP9 nCWL10 **nCCDS2 nCCDL4** nCCDR3 nRRDS4 nRRDL5 **nFAW24** | hbm3.py:225-236 |
| read_latency | nCL+nBL = **22 CK = 13.75 ns** | spec.py:81; controller_base.cpp:230 |

> Compared with this scheme's placeholder numbers: column granularity 256b, page 1KiB **match** (no recomputation needed). Frequency uses the 6400Mbps preset's tCK=625ps.

---

## 2. Full DPF algorithm (pseudocode + instruction mapping + execution layer)

Notation: node seed = 256b (= ChaCha key/8 words), control bit `t = lsb(seed)`; binary w=1, m=2, depth n=31.
PRG: ChaCha8(state 512b)→512b output = **[s^L(256b) | s^R(256b)]**, control being each one's LSB (half-tree, no independent control field).

### Gen(α,β) → (k0,k1)  —— top-down, generating CW along the levels (dpf.cpp:330-389)
```
s_b ← root_b;  t_b ← b                                  # each party's own root seed
for i in 0..n-1:                                        # each level (serial along one path of α)
  (sL_b, sR_b) ← ChaCha8(s_b)                           # PRG expansion   [256 VADD/VXROL + 16 ff]
  my_cw ← (⊕_j H(parent_j)) ⊕ DltSft_share[i]          # this party's CW share (dpf.cpp:349-355) [VXOR; β/Δ enters here]
  exchange my_cw ↔ their_cw                             # ★2-party communication: send/recv m-1 blocks (dpf.cpp:358-369)
  CW[i] ← my_cw ⊕ their_cw                              # reconstruct CW (1 constant per level) [VXOR]
  s_b ← child_{α_i}(sL_b,sR_b) ⊕ cond_xor(t_b, CW[i])  # descend along α + correction (dpf.cpp:382-388) [VCXOR]
  t_b ← lsb(s_b)
k_b = (root_b, {CW[i]}_{i<n})                           # CW public, root private
```
- **CW formula**: `CW[i][k] = (⊕_j H_k(parent_j) ⊕ DltSft[i][k])₀ ⊕ (…)₁`, k∈[0,m-2]; `CW[i][m-1]=⊕_{k<m-1}CW[i][k]` (free m-th child, dpf.cpp:371-373).
- **control-bit rule**: `t = lsb(seed)`; CW is applied only when `t=1` (`cond_xor`).
- **β/Δ**: `DltSft_share[i][k]=⟨(∏_c(α_{iw+c}⊕k̄_c))·Δ⟩_b` (dpf.h:40) → folded into each level's CW, **leaves have no separate correction**.
- Execution layer: PRG/CW application is on the **PU**; CW reconstruction requires **cross-party communication** (over host/network, 1 round per level).

### EvalAll(k_b) → 2^n shares  —— throughput-limited, per-node independent, lane-local (dpf.cpp:293-318)
```
for each node at level i (all parallel, SIMD 8 seed/batch):
  (sL, sR) ← ChaCha8(seed)                              # [256 + 16 ff]
  t ← lsb(seed)
  child0 ← sL ⊕ cond_xor(t, CW[i][0])                  # [VCXOR]
  child1 ← sR ⊕ cond_xor(t, CW[i][1])                  # [VCXOR]
  emit child0,child1 → next level                       # [16 VST]
# share0 ⊕ share1 = β @α, 0 elsewhere; no cross-node reduction
```
- Extra op per node for DPF = **2× VCXOR** (once for each of the left/right children), CW[i] is a per-level broadcast constant (VLD once per level, amortizes to ≈0).
- **No cross-lane / no bit-field extract / no β leaf op** (all absorbed by LSB-control + per-level broadcast CW).
- Leaf output = **a single 128b block, XOR-shared** (β a single word, **not Z_p**, no modmul) (agent verified dpf.cpp:404-462).

### Memory & feasible upper bound on n
- EvalAll output = 2^n leaves. leaf = 256b seed(32B) → n=31 output = 2³¹·32B = **64 GB** (16B/leaf → 32 GB),
  **does not fit in a single stack** (HBM3 8hi≈8-16 GB / 16hi≤32 GB).
- Single-stack materialization upper bound ≈ **n≈29 (32B/leaf) ~ 30 (16B/leaf)**; to reach n=31 must **span multiple stacks or stream the output** (EvalAll only needs to retain the current-level frontier, but the final 2^31 leaves still have to land). This evaluation reports the **steady-state throughput** at n=31 (independent of whether it is materialized all at once).

---

## 3. Final ISA table (baseline + additions)
| Instruction | Semantics | lane nature | Cost | Frequency in DPF |
|---|---|---|---|---|
| VADD32 / VADD32I | per-lane 32b mod 2³² add (+imm) | per-lane | ARX cheap | 128/batch in core |
| VXOR | 256b bitwise XOR | per-lane | cheap | ff/CW reconstruction |
| **VXROL** Vd,Vs,#k | Vd=rotl32(Vd^Vs,k) fused | per-lane | cheap (key) | 128/batch in core |
| VROL32 | standalone rotl32 | per-lane | cheap | redundant/spare |
| VLD/VST col | already-open row column ↔ register | — | DRAM column access | 16+16/batch |
| VMOV / VBCAST #imm | register copy / fill constant across all lanes | per-lane | cheap | CW/constant load |
| NOP / REP | control / hardware loop | — | — | — |
| **★VCXOR Vd,Vs,Vctrl** | **predicated XOR: per lane `Vd^=Vs` when lsb(Vctrl)=1** (=cond_xor) | **per-lane** | **ARX cheap** (=VMASKLSB+VAND+VXOR synthesizable) | **2/node** (Eval applies CW; Gen descent) |
| (optional) VMASKLSB Vd,Vt | Vd=lane lsb expanded into an all-1/all-0 mask | per-lane | cheap | if not doing VCXOR fusion |

**Completeness**: Gen = PRG(VADD/VXROL/ff) + CW compute(VXOR) + communication + descent(VCXOR); EvalAll = PRG + 2×VCXOR + VST.
The full flow runs end-to-end. **Redundancy**: VROL32 (covered by VXROL, kept as spare); VMASKLSB can be dropped if VCXOR exists.
**Only one addition, VCXOR, is required, and it is lane-local ARX cheap** → does not break the "pure ARX cheap" advantage.
The cross-lane reduction / bit-field extract / standalone β-op the prompt expected are **all unnecessary** (see §2/verification).

---

## 4. Performance report (formula + substitution)

### A. Instructions/cycles per batch (8 seed)
| Segment | Instructions |
|---|---|
| ChaCha8 core = 32 QR×(4 VADD+4 VXROL) | 256 |
| feed-forward | 16 |
| **DPF CW (2 VCXOR)** | **2** |
| compute subtotal | **274** |
| column read VLD (const-fold → 8) | 16 |
| write-back VST (2 children×8 words) | 16 |
**Single-issue serial PU (one instruction at a time, no overlap)**: one batch = read 16 columns → compute 274 → write 16 columns, three serial segments + ACT.
- single-batch latency = 274 + 32 I/O + first-column tRCD(31) ≈ **337 CK** (measured 337).
- steady-state per-batch (measured gated, 64ch) = **408 CK** (compute 274 + memory access/ACT ~134, serially added); in-row packing = **347 CK**.
- register file needs ≥ **24–32×256b** (16 working + 8~16 init).

### B. Single-PU throughput (formula: 8 / (per-batch×tCK), per-batch = 408 CK measured serial)
| Frequency | per-batch | ns/batch | cycle/seed | ns/seed | nodes/s·PU |
|---|---|---|---|---|---|
| **(a) CK 1.6 GHz** (command clock upper bound) | 408 CK | 408×0.625=**255** | 51 | 31.9 | 8/255ns=**3.14×10⁷** |
| **(b) PIM FPU 1.0 GHz** (self-set, note: HBM-PIM core 0.3–1 GHz) | 408 cyc | **408** | 51 | 51 | **1.96×10⁷** |

### C. System throughput = N_PU × single-PU (current architecture: **64 channel × 1 PU/ch = 64 PU**)
| Frequency | N_PU | system nodes/s | output bandwidth ×64B (internal generation rate) |
|---|---|---|---|
| (a) 1.6 GHz | 64 | 64×3.14e7 = **2.01×10⁹** | **128 GB/s** |
| (b) 1.0 GHz | 64 | 64×1.96e7 = **1.25×10⁹** | 80 GB/s |
- More PUs can only come from **adding channels** (measured linear 65.6×); adding banks within a channel is limited by the command bus.
- In-row packing (347/batch) raises the above table by 408/347 ≈ **1.18×**.

### D. Latency
- **EvalAll single-batch latency** = 337 CK = **211 ns** ((b)=337 ns).
- **Whole-tree EvalAll** = 2³¹ / system throughput: (a) 2.147e9/2.01e9 ≈ **1.07 s**; (b) ≈ 1.72 s (64 PU).
  (Adding channels shortens linearly: 256 ch → ÷4 ≈ 0.27 s.)
- **Gen critical path** = serial along the 31 levels of α: each level = compute ~274 CK(171 ns) + **1 two-party CW exchange** (`levels`=31 round-trips,
  m-1=1 block per round; batch version B·(m-1) block but still 31 rounds, dpf.cpp:359/519).
  `Gen ≈ 31×(171 ns + RTT)`. RTT=0 compute lower bound = **5.3 µs**; RTT=2 µs (same data center) → **~67 µs**, **communication latency dominates**.

### E. DRAM timing check (under single-issue serial, memory access counts directly into per-batch)
- **16 column reads**: same BG limited by nCCDL=4 → 16×4=**64 CK**; 16 column writes similarly ~64 CK; together with ACT/tRCD totaling ~134 CK/batch,
  **serially added to the 274 compute** (not hidden) → per-batch 408.
- **Cross-level ACT**: nFAW(4 ACT/24CK), nRC(72) **far from triggered** (only 1 ACT per batch) → ACT is not the bottleneck, but its tRCD counts into per-batch.
- ⟹ **No timing violation**; memory access (~33%) and compute (~67%) are both on the critical path.

---

## 5. Bottleneck determination
**Single-issue serial PU: per-batch = 408 CK entirely on the critical path — compute 274 (67%) + read/write/ACT ~134 (33%), serially added, no hiding.**
Basis: sim measured gated (single-issue) 64ch = 6536/16 = 408/batch; cl=0 (pure memory access) = 240/batch, cl sensitivity confirms compute is the largest single item.
ACT is only 1 per batch (nFAW/nRC far from triggered), frontend/DMA utilization 0.16 op/cyc ≪ 1, neither is the bottleneck.

## 6. Plausibility conclusion
- **The datapath + completed ISA can run the full DPF** stands up; under single-issue serial, per-batch 408 CK, both compute and memory access on the critical path.
- **The added cross-lane/bit-field operations do not break "pure ARX cheap"**: EvalAll actually only adds 1 lane-local VCXOR (×2/node),
  control=LSB, CW=per-level broadcast constant, β folded into CW — **the true cross-lane XOR reduction + 62-bit modmul + modular inverse only appear
  when embedding DPF into PCG**: leaf summation `S=Σleaves` (pcg_ole_impl.h:179-181), CW-scale `β/v` (modular inverse+multiply, :199-206, ffp.h:37/48-59),
  scatter into g (:218-229), inner product `<a⊗a,g>` (:245-256) — **all in PCG, not in DPF eval** (agent verified).
- **Optimization lever ordering** (under serial execution both compute and memory access count directly into per-batch, both worth shrinking):
  ① **shrink compute 274 (67%, biggest lever)**: raise PU frequency; reduce instructions (VXROL fusion already used; fold ChaCha constants into SRF);
  ② **shrink memory access ~134 (33%)**: in-row packing saves ACT (408→347) + const-fold column reads 16→8;
  ③ **scale PUs by adding channels** (measured linear 65.6×), adding banks within a channel is limited by the command bus (16-bank only 4.3×).
  Keep CW compute **at the PU level** (one broadcast constant per level, trivial), **do not push it down**.

## 7. Ramulator2 change list (to really simulate this scheme)
1. **PIM opcode**: add a macro command `GGM_DPF_EXPAND` (load opsize columns→`compute_latency`=274 cyc→store 2·opsize columns),
   or count VCXOR into compute_latency; reference the local `aim_simulator`'s `ISR_GGM_EXTEND` (request.h opcode + fields).
2. **Frontend**: AiM-style trace parsing (`aim_simulator/src/frontend/impl/memory_trace/aim_trace.cpp` template).
3. **PU timing model**: inject load→compute→store into the HBM34 controller, **single-issue serial** (read+compute+write do not overlap, serial on the same PIM-unit FPU)
   (reference `aim_simulator/src/dram_controller/impl/AiM_dram_controller.cpp`'s `m_fpu_busy_until`/`pim_unit_of`).
4. **Placement level & PU count**: parameterize banks_per_pu (bank=1 / BG=4), N_PU=ch·banks/ch÷ba_per_pu; channel configured as 16 in memory_system; 8/16-die switch via sid=2/4 preset.
5. **Inter-level read/write + CW traffic**: model each level as a "read layer-i seeds → write layer-(i+1) children" GGM op stream; CW is a per-level broadcast constant (traffic negligible, 1 block/level).
6. **Gen two-party CW exchange**: Ramulator2 single-node → model each level's CW exchange as an **inter-level fixed-RTT event** (over host/cross-stack, 31 serial injections), or move Gen communication out of the sim and report separately.
> Functional correctness (ChaCha bit-exact, share0⊕share1 check) goes into **PIMSimulator** (already has ggmExtend/ChaCha8 demo); Ramulator2 only runs cycles.

## Verification
- Baseline: `PYTHONPATH=python python -m pytest tests/smoke/testcases/hbm3.py` (confirms HBM3+HBM34 are simulatable; note this machine needs g++-12/clang-15, current g++-11/clang-14 has limited compilation).
- After extension: construct a single-batch DPF trace (16 RD same row + compute_latency=274 + 16 WR), hand-check against §4A's 274/337 CK;
  sweep channel/PU to reproduce §4C system throughput and §4E's ACT/tCCD no-violation; functional correctness compared against dpf.cpp's EvalAll leaves in PIMSimulator.
