# End-to-End Benchmark: Ring-LPN PCG→OLE with ChaCha8-DPF on HBM3-PIM (Methodology ①–⑧)

**Measurement target**: Ring-LPN PCG (output OLE: z0+z1=x0·x1 over Z_p, p=4611686018326724609 (62-bit), c=2),
reference implementation `~/PCG-acceleration` (half-tree DPF + GMW circuit + NTT). PRG=**ChaCha8** (no AES).
**PIM architecture**: 64ch × 1 rank × 2 bank, 2 banks share 1 PU (=64 PU), **single-issue serial PU** (1 instruction/cycle, no overlap),
HBM3_6400Mbps timing. **Two things newly made concrete this round**: (a) **real tree dataflow** (children written at level i = parents read at level i+1,
inter-level SYNC, measured on Ramulator2/AiM sim); (b) **compute datapath made concrete**: cl=**282**
(=256 ChaCha core + 16 ff + **8 PACKLR32** + 2 VCXOR; see `compute_datapath.md`, PIMSimulator bit-accurate implementation of the same).

> Absolute times are annotated with platform/frequency; comparable quantities are presented as **share / ratio / scaling** (methodology ⑧).
> Two PU frequency tiers: **(a) 1.6 GHz** (=DRAM command clock tCK=625ps upper bound); **(b) 1.0 GHz** (realistic PIM FPU tier, self-assumed).

---

## 1. Ramulator2 HBM3 parameter table (single source of truth, value + citation)

| Item | Value | Source (upstream CMU-SAFARI ramulator2) |
|---|---|---|
| Hierarchy | Channel→PseudoChannel→Sid→BankGroup→Bank→Row→Column | python/ramulator/dram/hbm3.py:13-21 |
| org (HBM3_8Gb_8hi) | pch2 / sid2 / bg4 / ba4; row 2^13; col 256; dq 32 | hbm3.py:214 |
| prefetch / column granularity | BL8; 8×32 = **256 b = 32 B/column** | hbm3.py:8 |
| row(page) size | 256 col-addr ÷ BL8 = 32 columns × 256b = **1 KiB/bank** | hbm3.py:214 comment |
| tCK | **625 ps** (6400 Mbps preset) | hbm3.py:234 |
| timing (CK) | nBL2 nCL20 nRCDRD31 nRCDWR15 nRP26 nRAS45 nRC72 nWR33 nRTP9 nCWL10 **nCCDS2 nCCDL4** nRRDS4 nRRDL5 **nFAW24** | hbm3.py:225-236 |
| read_latency | nCL+nBL=22 CK=13.75 ns | spec.py:81 |
| Simulation vehicle | Above timings injected into local AiM sim (`sim/src/dram/impl/HBM2.cpp`'s `HBM3_6400Mbps` preset + `PIM64_2Banks` org: 64ch×1pch×1bg×2ba, 2^17 row, 64 col) | this repo |

---

## 2. Complete DPF: Gen/EvalAll instruction mapping + final ISA

Construction = half-tree (dpf.cpp): control bit = seed LSB; CW = per-level broadcast constant; β/Δ folded into each level's CW (DltSft).
**EvalAll per node**: ChaCha8 → 2×VCXOR (apply CW) → PACKLR32×8 (word-major store reordering) → write 2 children.
**Gen**: along α, 1 ChaCha per level + this party's CW share (VXOR reduction to host) + **two-party exchange, 1 round** + VCXOR downstream.

| Instruction | lane | Cost | Frequency | Notes |
|---|---|---|---|---|
| VADD32/VADD32I | per-lane | ARX cheap | 128+16/batch | core + ff |
| VXROL (fused) | per-lane | cheap | 128/batch | key fusion |
| VXOR / VMOV / VBCAST | per-lane | cheap | few | CW/constant |
| **VCXOR** (new, required) | per-lane | cheap | 2/batch | cond_xor applies CW |
| **PACKLR32** (newly included, required) | **cross-lane** | medium | 8/batch | child-node word-major interleave (undercounted in previous version) |
| VLD / VST | — | DRAM column access | 8 read + 16 write/batch | const half resides in SRF → 8 reads |
| VROL32 / vec_rot | per/cross | — | 0 | redundant / only needed for word-parallel (not adopted) |
| NOP/REP; GGM_REDUCE macro (modmul) | — | modmul ~20 cycles/leaf (assumption) | leaf_convert | 62-bit Barrett on PU (multi-precision 32b sequence) |

**Completeness**: full Gen+EvalAll flow is runnable (§2 mapping); cross-lane only in PACKLR32 (wrap-up), hot loop is pure ARX.
**cl = 256+16+8+2 = 282**; registers = 16×256b+SRF = 17 total (exactly full, same as PIMBlock.h).

---

## 3. Methodology ①–⑧

### ① Correctness gate
Upstream test assertions: DPF point function (share0⊕share1=Δ@α, 0 elsewhere; test/half_tree_dpf.cpp:57-89),
OLE z0+z1=x0·x1 (test/pcg_ole_2pc.cpp:171-207, spot-check min(N,256) points), triple c=a·b (test/triple.cpp:80-91).
**Local status**: existing binary is macOS ARM64; Linux rebuild requires emp-tool/emp-ot, which was blocked by the permission policy in this session (external code build).
⟹ **Correctness cites upstream tests (README/report declares all PASS, including ChaCha8 backend), not re-run locally**—if a local gate is needed,
after approving the emp build, `./run_2pc build/bin/test_pcg_ole_2pc N t w` re-verifies in one shot. Coverage: upstream triple test sweeps w∈{1,2,3}×{AES,ChaCha8}.
PIM-side bit-accurate correctness is carried by the PIMSimulator GGM demo (leaves bit-identical to CPU reference).

### ② End-to-end phase decomposition vs N (host baseline = profiling_report.md measured, c=2,t=4,w=1; Apple M ARM64)
| Phase \ N | 2¹⁸ | 2²⁰ | 2²² | 2²⁴ |
|---|---|---|---|---|
| total wall (s) | 1.27 | 5.00 | 19.89 | 139.2 |
| modmul(__udivmodti4) 【software artifact】 | 84% | 75% | 71% | 44% |
| NTT core (step4) | 10% | 12% | 13% | 11% |
| **dpf_expand** | 2% | 9% | 9% | **22%** |
| mem ops | — | 3% | 3% | 10% |
| net_wait (share of wall) | 14% | 17% | 7% | 28% |
| deltashift/OT (FerretCOT) | ≲2% (§4.4: invisible at large N) ||||
**crossover (dpf+step4 > deltashift)**: at t=4, holds for N≲2¹⁸ (FerretCOT grows only with log);
the t=32 measured table (below) already shows 87% vs 14% at **N=2¹⁶**.

### ③ (t,w) sensitivity
Measured anchor (profiling t=32, w=1):
| Phase \ N | 2¹⁴ | 2¹⁶ | 2¹⁸ | 2²⁰ |
|---|---|---|---|---|
| FerretCOT+circuit (≈pos_add+payload_ot+deltashift) | 83% | 14% | 32% | 11% |
| dpf_expand | 8% | 37% | 38% | **54%** |
| NTT+modular | 2% | 50% | 27% | 26% |
Pattern (formula): dpf ops ∝ c²t²·(2·2N/t/8)=**c²·t·N/2** (linear in t); deltashift AND gates ∝ t²·w·log₂(N/t)
(linear in w, super-linear in t²); NTT ∝ N logN independent of t → **t↑ pushes the share from NTT toward dpf** (t=4→32: 9%→54% @2²⁰ ✓),
**w↑ only lifts deltashift** (pronounced at small N / large t). Security calibration stress point: small N large t (N=2¹⁴,t=32: OT 83% dominates).

### ④ standalone DPF expansion (measured: real tree dataflow @64PU, gated serial, tCK=625ps)
**Primary configuration = dpf.cpp-aligned (128b seed, 4 reads/8 writes, cl=279; corrected modmul = 1 per 128b leaf block)**:
| domain n | cycles | ACT | RD/WR | total time (µs) | **ns/leaf** | **ChaCha8/s** |
|---|---|---|---|---|---|---|
| 12 | 5814 | 128 | 2.6K/5.2K | 3.63 | 0.887 | 1.13×10⁹ |
| 16 | 43317 | 1408 | 33K/67K | 27.1 | 0.413 | 2.42×10⁹ |
| 18 | 166179 | 6144 | 132K/263K | 103.9 | **0.396** | **2.52×10⁹** |
| 16 **+fused leaf_convert** | 64059 | 1920 | 66K/71K | 40.0 | 0.611 | — |
| 18 **+fused leaf_convert** | 248441 | 8194 | 263K/280K | 155.3 | **0.592** | — |
| pure memory access cl=0 (n=16/18) | 10266/35672 | — | — | — | 0.098/0.085 | — |
| (256b variant n=16/18, for comparison) | 49148/189678 | — | — | — | 0.469/0.452 | — |
- Formulas: ns/leaf = cycles×0.625/2ⁿ; ChaCha8/s = (2ⁿ−1)/total time. Asymptotic (deep two levels dominate, 128b)
  ≈ 2×279/(16×64)×0.625 = **0.34 ns/leaf** ((a) tier); (b) 1.0 GHz tier ×1.6.
- **PCG anchor measured** (instances mode, N=2¹²,t=4: 64 trees of n=11 + fused convert) = **122615 CK = 76.6 µs**
  → 0.585 ns/leaf ✓ consistent with subtree mode 0.592—PCG phase time = I·D·0.59ns linear extrapolation holds.
- **Cost of real inter-level read-back** (vs previous round's independent batch): compute share n=16: 36660/49148=**75%**,
  n=18: 144948/189678=**76%**—the remaining ~24% = per-level SYNC drain + shallow-level under-parallelism (L0–5 on a single channel) +
  memory-access tail. Tree dataflow is not free; the independent-batch model cannot see this part.
- **★Complete DPF (expand + fused leaf_convert)**: after alignment correction (modmul = 1 per 128b leaf block,
  previously counted per 32b word, overcounting by 8×, "REDUCE=expand×3.3" is void)—n=16: 64059 CK, REDUCE=20742
  = 8chunk×2560 ✓ matches hand calculation ⟹ **leaf_convert ≈ 48% of expand**, complete DPF still **ChaCha-dominated (68%)**.
  The true cycle count of the 62-bit modmul's 32b instruction sequence (currently assumed 20 cycles/leaf) is still the largest sensitivity term on the kernel side.

### ⑤ Memory scan (leaf materialization = I·D·32B, I=c²t², D=2N/t; seed=32B)
| (N,t) security calibration | I | D | leaf materialization | 8-die(≈16GB)/16-die(≈24-32GB) |
|---|---|---|---|---|
| (2¹⁶,16) | 1024 | 2¹³ | **256 MB** | ✓/✓ |
| (2²⁰,8) | 256 | 2¹⁸ | **2 GB** | ✓/✓ |
| (2²⁴,8) | 256 | 2²² | **32 GB** | ✗/marginal |
| (2²⁴,4) | 64 | 2²³ | 16 GB | marginal/✓ |
**fused REDUCE (no leaf materialization)** working set = in-flight instance × 3D×32B ≈ 64×3×2²³×32 ≈ 48 GB → level-by-level ping-pong
actual = 2-level frontier ≈ **2×D×32/64 per channel**—N=2²⁴ is also only ~512 MB/stack ⟹ **the feasible N upper bound is determined by leaf materialization:
8-die N≤2²²(t=8); the fused mode lifts this upper bound**.

### ⑥ cold vs steady
Cold = one-time FerretCOT/COT setup + key/root seed generation: measured anchor at N=128, OT accounts for ~90% (profiling §4.4),
at t=32, 2¹⁴ accounts for 83% → drops to 11% at 2²⁰. setup ∝ t²·w·log(N/t) (circuit), steady ∝ N logN
⟹ **cold/steady ∝ log/N·logN → hyperbolic decay with N**: 2¹⁴≈5:1 → 2¹⁶≈1:6 → 2²⁰≈1:9 (t=32 measured, converted).

### ⑦ Communication
- rounds: DPF CW 1 round per level (shared by all instances after batching) → **rounds = dpf_n = log₂(2N/t) ∝ logN** ✓
  (+2 rounds per pair: leaf-sum, CW-scale; +constant rounds for FerretCOT setup).
- bytes = c²·[dpf_n·B·16B (tree CW) + 2·B·8B]; N=2²⁰,t=8: 4×[18·64·16+1024] = **78 KB** (protocol body is tiny;
  the bulk is OT setup ~MB scale).
- **PIM does not change communication volume**: net_wait (host-measured 14–28% wall) **rises in share** after PIM compresses compute, becoming
  an **N-independent floor** (RTT×logN + bandwidth term); the final ceiling on end-to-end speedup comes from here (see Amdahl).

### ⑧ Portable quantities
All of the above are given as share/ratio/scaling; absolute ms are annotated (host=Apple-M ARM64 measured; PIM=this sim tCK=625ps, two frequency tiers).

---

## 4. Timing check + bottleneck

**Inside dpf_expand** (measured, tree dataflow):
- Under single-issue serial, per-batch = 282 compute + 8RD/16WR + ACT amortized → steady state ~420 CK/batch (cl282 + memory-access serial);
  the leading 8 consecutive reads 8×nCCDL=32 CK ≪ 282 (could be fully hidden if overlap were allowed, but **this model does not overlap, it adds serially directly**).
- ACT: reads and writes each page-flip along the 64-col row, one ACT per 8 (read)/4 (write) batches → far below the nFAW(4/24)/nRC(72) limits; measured n=16
  only 2944 ACT/66.6K RD ✓ no timing violation.
- **Verdict (split into two levels, after 128b alignment)**: **expand = compute-bound 86%** (n=18: 143406/166179; memory access already
  squeezed by 128b + row packing + ping-pong down to ~69 CK/op); **complete DPF = ChaCha-dominated 68% + leaf_convert 32%**
  (after modmul accounting correction, no longer modmul-bound). Secondary ~14% = inter-level SYNC + shallow-level under-parallelism.
**End-to-end**: on host, modmul(software, Barrett-fixable)>NTT>dpf; after fixing Barrett, dpf/NTT are on par, net_wait is the floor.

## 5. Latency / Throughput summary table

**kernel level (dpf @64PU, 128b-aligned measured, (a)1.6GHz/(b)1.0GHz)**:
| Quantity | (a) | (b) |
|---|---|---|
| ns/leaf expand (asymptotic/measured n=18) | 0.34 / **0.396** | 0.54 / 0.63 |
| ns/leaf complete DPF (+leaf_convert) | **0.592** measured | 0.95 |
| ChaCha8/s (system) | **2.52×10⁹** measured | 1.58×10⁹ |
| single-batch latency (279+I/O+ACT) | ~330 CK = 206 ns | 330 ns |
| Gen critical path (n levels serial, N=2²⁰t=8:n=18) | 18×(174ns+RTT): RTT=0→3.1µs; RTT=2µs→**39µs** (communication-dominated) ||

**End-to-end (N∈{2¹⁶,2²⁰,2²⁴}, t calibrated {16,8,8}, host anchor + PIM complete DPF(0.59ns/leaf) substituted)**:
| N | host total | dpf share | PIM dpf time(a) | k_dpf | end-to-end (see Amdahl) |
|---|---|---|---|---|---|
| 2¹⁶(t=16) | ~1.9 s (t=32 table approx) | ~37% | 2²³×0.59ns≈**4.9 ms** | ~143 | ≤1.58 |
| 2²⁰(t=8) | 5.0 s (t=4 table) | 9%(t=4)/54%(t=32) | 2²⁶×0.59ns≈**40 ms** | ~11(t4)/~190(t32) | ≤1.10(t4)/≤2.1(t32) |
| 2²⁴(t=8) | 139.2 s | 22% | 2³⁰×0.59ns≈**0.63 s** | ~49(t4:2²⁹→0.32s,k=97) | ≤1.28 |

## 6. End-to-end Amdahl upper bound (speedup ≤ 1/(1−s+s/k))
| Case | s | k | Upper bound |
|---|---|---|---|
| Accelerate dpf only (modmul unfixed, N=2²⁴,t=4) | 0.22 | 71 | **1.28×** |
| dpf+NTT offloaded together (+GPU) | 0.33 | ≫1 | **1.49×** |
| **after modmul Barrett fix** (host cpu reorder, 2²⁴: mod÷6) | dpf share→~0.21′ (total time drops to ~103s) | 71 | 1.27×(dpf) / **~2.6×**(dpf+NTT+mem all offloaded) |
| large t (t=32,N=2²⁰,dpf 54%) | 0.54 | ~270 | **2.16×** (+NTT 26%→**~4.6×**) |
**Conclusion**: end-to-end gains from touching only dpf_expand are capped at 1.1–1.3× by modmul(software)/NTT/net_wait;
**realizing PIM value requires the combination**: Barrett (front-loaded) + NTT→GPU + dpf→PIM (i.e. the division of labor in pim_gpu_dataflow.md),
with the large-t regime (DPF-dominated) yielding the biggest gains (≥2–4.6×). net_wait is the final N-independent floor (28%@2²⁴ → ceiling ~3.6×).
**kernel-configuration sensitivity**: the k in the table above is expand-only; if leaf_convert is also moved into PIM (20 cycles/leaf modmul),
PIM kernel time ×4.3 (§④★) → k divided by ~4 (still ≥16-60), but the host-side leaf_convert (included in the modmul share)
is offloaded along with it, so s increases—net effect on the Amdahl upper bound is **favorable** (moving away 44-84% of the modmul phase),
the exact value depending on the true cycle count of the 62-bit modmul on the PU (the largest sensitivity term).

## 7. Plausibility + Ramulator2 change list
- Datapath+ISA runs the complete DPF ✓; real tree dataflow reveals 25% level-sync/shallow-level overhead (invisible to the independent-batch model);
  cross-lane only in PACKLR32 (wrap-up), hot loop pure ARX ✓; registers exactly full 17×256b (zero slack is a real constraint).
- Change list (implemented ✔/to do): ✔GGM_EXTEND/REDUCE macro ops + fields; ✔single-issue FPU gating (`fpu_gate_issue`);
  ✔PIM64_2Banks placement/PU count; ✔tree trace of inter-level read/write + SYNC (`gen_dpf_tree_trace.py`);
  to do: CW/control-bit broadcast traffic (1 block per level, negligible, currently folded into SYNC); Gen two-party CW exchange = inter-level RTT event injection
  (currently analytical: n×RTT); the precise cycle count of decomposing REDUCE's 62-bit modmul into a 32b instruction sequence (currently a 20 cycles/leaf assumption).

---
### Reproduction
```
python3 tools/gen_dpf_tree_trace.py -n 16 -C 64 --cl 282 -o t.trace      # real tree dataflow
sim/build/ramulator2 -f sim/test/hbm3_dpf_64ch.yaml -t t.trace -p MemorySystem.Controller.fpu_gate_issue=true
bash tools/run_tree_sweeps.sh                                            # full suite → results/tree_sweeps.txt
```
Data files: results/tree_sweeps.txt; compute configuration: docs/compute_datapath.md; host anchor: PCG-acceleration/profiling_report.md.


## Security-calibrated E2E campaign (BCG+20 Table 1; Beaver algorithm; naive/4step world)

Full tables: pim/results/e2e_secparams.txt. Benchmark caliber = the 30
security rows (lambda x N x (c,t)); leaves = c^2*2Nt, NTT size = N,
batch = c^2. Algorithm: Beaver output layer -- NO modular multiply anywhere
in DPF+convert (PIM = dual-ALU ARX cl155 + fused ChaCha CONVERT cl465 +
multiplier-free mod-ALU). NTT entirely on the SM via 4-step (GPU baselines:
naive and 4step kernels, both L40S silicon, BITEXACT-gated).

MEASURED ablations:
- **DPF +-LSU** (sim, corrected ggm_compute_overlap semantics: no-LSU holds
  the FPU through the serial mem+compute window, no II): chacha465
  692,720 -> 761,808 cycles = LSU gain **1.10x**, cross-checked against the
  memory floor (65,796/692,720 = 9.5% -- the ledger closes).
- **4step +-DRU**: transpose share 26-32% (L40S) / 21.7-24.5% (B200,
  measured); DRU hides it in shadow bandwidth on HBM (GDDR recorded 0 --
  no shadow). NTT-phase stream contention on the PIM lane measured
  **f_dru = 0.00%** (chacha465+stream = 692,720, bit-identical). e2e DRU
  gain appears where the NTT lane approaches the DPF lane: H200/B200
  1.04-1.08x at N>=2^22; 1.00x elsewhere (DPF-bound).

Headline (lam=128, c=4, t=16; x vs the GPU-4step baseline, PIM +LSU+DRU):
| gen | dev | 2^20 | 2^24 |
|---|---|---|---|
| GDDR6 | L40S (all-measured) | 2.5x | 2.6x |
| GDDR6 | RTX5000Ada (est) | 2.2x | 2.3x |
| GDDR7 | RTXPRO6000 (est) | 3.1x | 3.2x |
| HBM2e | A100 (est) | 4.3x | 4.5x |
| HBM3 | H200 (est) | 7.6x | 7.9x |
| HBM3e | B200 (4step measured) | 11.0x | 11.2x |
Vs the naive-NTT baseline the same cells read 3.2-15.8x. All security rows
are DPF-bound on every device (the 4step NTT lane never dominates after
the DPF's chacha465 residency) -- the memory-generation ladder is therefore
almost purely the PIM bandwidth story: 2.3x (GDDR6) -> 11.2x (HBM3e).

## L40S three-baseline campaign (measured CPU / GPU / PIM+GPU)

Full report: `pim/results/l40s_cpu_gpu_pim.txt`; raw CPU rows:
`pim/results/cpu_secparams_l40s.csv`.

- **CPU baseline (first on-host measurement)**: `bench_pcg_ole_2pc`
  real 2PC on localhost (Xeon Gold 6442Y, chacha8, verify=PASS gated),
  all 25 unique (N,c,t) of the BCG+20 table, gen_and_expand wall.
  88.6 s (2^20,c2,t64) to 3866.6 s (2^24,c8,t8).
- **Speedups (30 security rows)**: PIM+GPU vs CPU 1613-2302x; vs
  GPU-naive 2.3-5.1x; vs GPU-4step 2.3-3.0x. Column order
  CPU > GPU-naive > GPU-4step holds on every row.
- **Ablation B (issue width x LSU, all measured)**: dual+LSU 0.0978
  ns/leaf (baseline); single-issue 1.75x slower (1210060 vs 692720
  cycles); LSU gain 1.100x (dual) vs 1.057x (single) — smaller with
  single issue because the compute share grows, as predicted.
- **Ablation B2 (data layout, same hardware)**: node-major baseline =
  per-word strided access, 4x rd+wr column traffic (measured 807580
  cycles = 1.17x, memory becomes the critical path); word-major
  design = 1.00x — the storage layout performs the transpose at VLD.
- **Ablation C (NTT +-DRU on L40S)**: transpose is 22-32% of the
  4step NTT lane (measured stage split); e2e DRU gain on GDDR
  recorded 1.00x by the honest no-shadow-bandwidth rule.
