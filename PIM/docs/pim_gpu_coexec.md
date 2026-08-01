# PIM ‖ GPU Co-Execution Model for Ring-LPN PCG-OLE

**Question.** If the DPF expansion (+ `leaf_convert`) runs on **near-bank PIM** and the
Ring-LPN NTT runs on the **CUDA SMs**, where is the bottleneck, how much does overlap
buy, and how many PU are needed so the PIM lane does not stall the GPU?

**Architecture (the key correction).** The PIM is **not** an external PCIe accelerator.
It is near-bank compute **inside the GPU's own HBM3 stacks** (HBM-PIM). The CUDA SMs are
on the same package and share the **same HBM**. Therefore:

* There is **no PCIe hand-off.** The vector `g` (the DPF/`leaf_convert` output) is written by
  the near-bank PIM directly into HBM cells and read back by the SMs from the same HBM.
* The two execution lanes use **different bandwidth domains of the same HBM**:
  * **near-bank PIM lane** — DPF tree expand + `leaf_convert`. All `2·c²·N·t` leaf blocks are
    produced and consumed **inside the banks**; they never cross the HBM→SM bus.
  * **SM-NTT lane** — forward NTT + pointwise-MAC + one inverse NTT. Uses the HBM→SM
    external bus + SM compute.
* The only shared-bus traffic is the small `g` materialization (N words) plus the NTT's own
  HBM round-trips — orders of magnitude below the DPF leaf volume.

Model, sweep data and this report:

* `pim/tools/coexec_model.py` — the analytical two-lane model (pure Python, no deps).
* `pim/results/coexec_sweep.txt` — full sweep, H100- and H200-class shared HBM.

---

## 1. The two-lane model

Work is decomposed into `c²` polynomial pairs `(i,j)` (here `c=2` ⇒ 4 pairs), each with
`B=t²` DPF instances of domain `D=2N/t`. Per pair the near-bank PIM must expand and convert

```
leaves/pair = B · D = t² · (2N/t) = 2·N·t          (leaf blocks, 128-bit each)
```

**Near-bank PIM lane (bank-internal BW):**

```
T_PIM(pair) = 2·N·t · (0.45 ns/leaf) / (PU / 64)
```

`0.45 ns/leaf` is the **measured** 64-PU full-DPF+`leaf_convert` rate (HBM3, tCK=625 ps,
ChaCha8, `pim/results/tree_sweeps_leafconv32.txt`, n=18 near-asymptote). PU scales linearly
across channels (measured 65.6× for 64 channels); packing extra banks **within** a channel
is capped (~4.3×/8 measured) — flagged in the model, not hidden.

**SM-NTT lane (HBM→SM bus + SM):**

```
T_GPU(pair) = 2 · fwdNTT(N)        (Move2: a·a spectra precomputed and resident;
                                     each pair needs the forward transforms only)
+ one INTT per instance, amortized.
```

`fwdNTT(N)` is derived from the **measured** merge-NTT full poly-mul table
(RTX 5000 Ada, `docs/merge_vs_4step.md:145-155`), taking `fwd ≈ full/3` (a full negacyclic
poly-mul = 2 fwd + 1 pointwise + 1 inv ≈ 3 transforms). A `--sm-scale` factor re-points the
table to a target HBM3 GPU (H100 ≈ 3×, H200 ≈ 4× the RTX 5000 Ada SM throughput).

**Shared-HBM contention (the check that matters):**

```
T_bus(pair) = (g write N·8B  +  NTT a·a·g reads ≈ 3·N·8B) / BW_HBM
```

**Steady state** (pipelined over pairs and many OLE instances):

```
T_pair = max( T_PIM , T_GPU , T_bus )
serial baseline A→B = T_PIM + T_GPU        (per pair)
speedup = serial / overlap
```

Bottleneck = `argmax` lane. Balance point: solve `T_PIM ≤ T_GPU` for PU.

**Sanity checks (built in):**
* PU→∞ ⇒ `T_PIM→0` ⇒ collapses to the pure-GPU time. ✓
* Instances-anchor: model `T_PIM(N=2¹², t=4, 4 pairs) = 59 µs` vs measured **76.6 µs**
  (the model uses the asymptotic 0.45 ns/leaf, so it slightly under-counts small-n shallow-layer
  overhead — right ballpark, correct direction). ✓

---

## 2. Results — bottleneck phase map

Two structural facts drive everything (both **confirmed** by the sweep):

1. **`T_GPU` is independent of `t`.** The NTT sees a degree-`N` polynomial regardless of noise
   weight. `T_PIM ∝ t`. So increasing the LPN noise weight `t` pushes work **onto the PIM lane
   only**.
2. **The shared HBM bus is never the bottleneck.** `T_bus` is 0.0004–0.16 ms across the whole
   sweep — 1–2 orders of magnitude below both compute lanes. This is the thesis payoff:
   because the DPF leaves stay bank-internal, co-locating PIM in the GPU's HBM does **not**
   create a bus-contention bottleneck. Overlap is essentially free.

Phase map (which lane is the bottleneck), H100-class, `c=2`:

| N \ (t, PU)      | small t / few PU | large t / few PU | any t / many PU |
|------------------|:----------------:|:----------------:|:---------------:|
| 2¹⁶ (65 K)       | PIM near-balanced | **PIM**          | GPU-NTT         |
| 2²⁰ (1 M)        | PIM               | **PIM**          | GPU-NTT (t≤8)   |
| 2²⁴ (16 M)       | GPU-NTT (PU≥256)  | **PIM**          | GPU-NTT         |

**One-line rule:** the PIM lane dominates whenever `t` is large or PU is small; the GPU-NTT lane
dominates once you add PU (or raise `t` cheaply). The crossover is entirely captured by the
balance curve below. The HBM bus never dominates.

Representative rows (H100-class, `sm_scale=3`):

```
      N   t    PU |  T_PIM ms  T_GPU ms  T_bus ms |     lane speedup PU@balance
  65536   4    64 |    0.2359    0.0330    0.0006 |      PIM    1.14        457
  65536   4   512 |    0.0295    0.0330    0.0006 |  GPU-NTT    1.79        457
1048576   8    64 |    7.5497    1.2124    0.0100 |      PIM    1.16        399
1048576   8   512 |    0.9437    1.2124    0.0100 |  GPU-NTT    1.69        399
16777216  4   256 |   15.0995   19.3991    0.1603 |  GPU-NTT    1.69        199
16777216 32    64 |  483.1838   19.3991    0.1603 |      PIM    1.04       1594
```

Peak overlap speedup is **~1.7–1.8×** (approached when the two lanes are balanced); it degrades
toward 1.0× when one lane dwarfs the other (nothing to overlap with).

---

## 3. Balance curve — "how many PU so PIM does not stall the GPU?"

Setting `T_PIM ≤ T_GPU` and solving:

```
PU* = 64 · T_PIM(64) / T_GPU
    = 64 · [ 2·N·t · 0.45ns ] / [ 2 · fwdNTT(N) ]
    ⟹  PU*  ∝  N · t / fwdNTT(N)  ∝  N·t / (N log N)  =  t / log N
```

**The PU threshold grows linearly with the noise weight `t` and only weakly (inversely with
`log N`) with the ring degree.** Concretely (H100-class):

| N        | t=4 | t=8 | t=16 | t=32 |
|----------|----:|----:|-----:|-----:|
| 2¹⁶      | 457 | 915 | 1829 | 3658 |
| 2²⁰      | 199 | 399 |  797 | 1594 |
| 2²⁴      | 199 | 399 |  797 | 1594 |

Read this as: **provision ≈ `t · 114` PU per HBM stack** (H100-class NTT) and the near-bank PIM
lane keeps up with the SMs. At the standard `t=4` LPN setting, **256 PU** (4 channels' worth in
the 64-PU-per-64-channel model, i.e. modest bank packing) already tips every N≥2²⁰ into
GPU-NTT-bound — the desired regime, where the expensive NTT is the pole and the DPF is hidden
underneath it. High-weight settings (`t=32`) need ~1.6 K PU to stay balanced; below that the
PIM lane is the pole and adding PU is the highest-value lever.

On a faster NTT (H200-class, `sm_scale=4`) `T_GPU` shrinks, so the same balance shifts **up**
by ~4/3× — a faster GPU demands more PU to keep it fed (`PU* ∝ 1/fwdNTT`). See the second block
of `pim/results/coexec_sweep.txt`.

---

## 4. NTT stage-1 offload to PIM (4-step)

Since the PIM PU already has the modmul (32 CK, Barrett — needed by `leaf_convert`), it has
every primitive an NTT butterfly needs (`modmul + modadd + modsub`). The constraint is
**dataflow, not compute**: near-bank PUs have no inter-bank interconnect, so only butterflies
whose span stays inside a bank are PIM-eligible. That is exactly the 4-step decomposition:

```
N = N1 x N2  (N1 <= 2^14, bank-local; N2 >= 64 columns for PU parallelism)
PIM : N2 column NTTs of size N1 + twiddle correction  = stage-1
      (fraction f1 = log2(N1)/log2(N) of all butterflies: 70% at N=2^20)
SM  : transpose read + N1 row NTTs of size N2 + pointwise = stage-2
Bonus: the negacyclic psi-twist (elementwise modmul by psi^i) fuses into
       leaf_convert's final modmul for free — PIM writes g pre-twisted.
```

Cost model: butterfly = modmul(32 CK / 8-lane batch, ~4 CK effective) + add/sub + a per-batch
twiddle VLD (twiddles are **not** broadcast constants — stage `s` has `2^s` distinct values,
so each bank holds a resident table) ⇒ **6 CK/butterfly/PU** at 1.6 GHz. The model picks the
offload fraction `x*∈[0,1]` minimizing `max(T_PIM_dpf + x·W1_pim, (1−x)·W1_gpu + W2_gpu, T_bus)`.

**Result — the offload is a one-sided lever, and the model confirms it:**

* **PIM-bound cells: `x* = 0`, gain 1.00×.** The PUs have no spare cycles; offloading would
  make the pole worse. The optimizer correctly refuses.
* **GPU-NTT-bound cells: `x*` rises toward 1, gain up to 2.3–3.3×.** E.g. N=2²⁰, t=4,
  PU=1024: `x*=1.0`, per-pair 1.21 ms → 0.36 ms (**3.3×**); N=2²⁴, t=4, PU=512: `x*=0.91`,
  19.4 ms → 9.1 ms (**2.1×**).
* **Large N hits a stage-2 floor.** With N1 capped at 2¹⁴ (bank-local), the stage-1 fraction
  shrinks as N grows (`f1 = 14/24` at N=2²⁴), so even full offload leaves the GPU
  `t_gpu·(1−f1)` of transpose + row-NTT work — the 2.4× ceiling at N=2²⁴, PU=1024.

**Revised one-line rule:** provision `~114·t` PU to hide the DPF; every PU beyond that buys
NTT stage-1 throughput at ~267 M butterflies/s/PU (64 PU ≈ one H100-class fwdNTT), until the
stage-2 floor `t_gpu·(1−f1)` is reached.

Caveats specific to the offload: (a) `leaf_convert`'s scatter layout must place each
contiguous g-chunk in the bank of the PU that will column-NTT it — a joint layout design, not
free; (b) the 6 CK/butterfly figure inherits the modmul=32 CK assumption and adds the twiddle
VLD, unvalidated in the trace simulator; (c) scheduling is dynamic — `x` must adapt per
(N, t, PU) operating point, as the phase map shows `x*` swinging between 0 and 1.

## 5. Speedup vs. an all-GPU baseline

Baseline: the **same GPU** runs everything itself — DPF expand + `leaf_convert` on the
SMs (serially before the NTT; same SM pool, nothing to overlap with), then the NTT.

Two baseline modes (`--baseline`):

* **`simgrounded` (primary, iso-device)**: both the baseline and the design's GPU lane use
  numbers measured by replaying L40S SASS traces on **one simulated machine** (Accel-Sim,
  SM80_A100 config): DPF+`leaf_convert` = 0.912 ns/leaf (ChaCha8, kernel-only, fastest of a
  three-shape grid — most favorable to the baseline), poly-mul(65536) = 56.1 µs, measured
  fwd fraction 30.4%. See `endtoend/results.md`. The only difference between baseline and
  design is *where the DPF executes* — the comparison isolates the architecture.
* **`bwscale` (legacy)**: the measured L40S asymptote (2.9 ns/leaf incl. D2H) scaled by
  memory-bandwidth ratio to the target GPU (memory-bound; AES vs ChaCha <3%).

```
T_allGPU(pair) = 2·N·t · 0.75ns  +  T_GPU_NTT(N)
speedup        = T_allGPU / T_coexec(offload)
```

Result: the speedup is set almost entirely by the **PU count**, because the all-GPU
baseline is dominated by its DPF term (93% of the baseline at N=65536, t=4 on the simulated
A100) while the co-exec design moves that term into bank-internal bandwidth:

| PU   | vs all-GPU, simgrounded (iso-device) | vs all-GPU, bwscale (H100-class) |
|------|--------------------------------------|----------------------------------|
| 64   | **2.0–2.2×**                         | 1.7–2.0×                         |
| 256  | **8.2–8.7×**                         | 6.7–7.5×                         |
| 512  | **16–17×**                           | 13–14×                           |
| 1024 | **25–31×**                           | 21–27×                           |

The iso-device numbers are slightly higher than the legacy BW-scaled ones because the
simulated A100's SMs are somewhat slower at the DPF than the BW-scaling estimate assumed —
the baseline got *more* honest and the case got *stronger*. Headline datum: **64 near-bank
PU ≈ 2× the entire A100 SM array on DPF+leaf_convert** (0.45 vs 0.912 ns/leaf).

≈ **0.027× per PU** — near-linear, because in the PIM-bound region both numerator (GPU DPF)
and denominator (PIM DPF) scale with `N·t`, leaving the bank-internal-vs-bus bandwidth ratio
times PU. The dips below the trend (e.g. 20.6× at N=2²⁰, t=4, PU=1024) are cells where the
co-exec side is GPU-NTT-bound even after offload — there, extra PU stops paying.

The honest floor: at the **same silicon budget of 64 PU** the co-exec design is only ~2×
faster than the all-GPU baseline. The 10–27× numbers are what the *architecture enables* —
PU count scales in the DRAM dies (measured linear across channels) without touching the SM
die, which is precisely the resource an all-GPU implementation cannot add.

## 6. Takeaways

1. **No new bottleneck from sharing HBM.** Because DPF/`leaf_convert` traffic is bank-internal,
   the shared HBM→SM bus stays 1–2 orders under both compute lanes across the entire sweep.
   Co-locating PIM in the GPU's HBM is contention-safe; overlap is near-free.
2. **The bottleneck is a `t`-vs-PU tug-of-war, not the bus.** `T_GPU` is `t`-independent, `T_PIM ∝ t/PU`.
   Small `t` or many PU ⇒ NTT-bound (good — the NTT is the fundamental cost). Large `t` or few PU
   ⇒ PIM-bound.
3. **Balance rule:** `PU* ∝ t / log N`, ≈ `114·t` PU/stack for H100-class NTT. Provision to this
   and the DPF hides entirely under the NTT; the co-execution speedup peaks at ~1.7–1.8×.
4. **Highest-value lever when PIM-bound:** add PU by adding **channels** (measured linear), not by
   packing banks within a channel (measured ~4.3×/8, command-bus limited).
5. **NTT stage-1 offload (§4) converts spare PU into GPU-lane speedup** — up to 3.3× in the
   GPU-bound region, exactly zero in the PIM-bound region. It changes the ceiling, not the rule:
   first hide the DPF (`~114·t` PU), then surplus PUs eat the NTT's bank-local stages.

## 7. Two-tier PIM: the channel-level MAU (validated)

Design: the multiplier leaves the DRAM die entirely. Near-bank PUs keep only the
ARX set (VADD32/VXOR/VXROL/VCXOR/PACKLR32 + loads/stores) and run pure DPF
expansion; one **MAU** (modular arithmetic unit) per channel on the **base logic
die** runs a fused-butterfly pipeline — 62×62 multiply (Booth-Wallace, ~3-4
stages) → **sparse-prime reduction** (P = 2⁶²−3·2²⁵+1 ⇒ `X_lo + (X_hi≪25) +
(X_hi≪26) − X_hi`, two fold rounds, no Barrett multiplies) → mod-add/sub pair,
**1 op/CK @1.6 GHz**, with a 128 KB twiddle SRAM and a 32–64 KB accumulator
tile. Four macro-ISRs (MAU_MAC / MAU_BFLY / MAU_TWIST / MAU_FLUSH) + CFR config
writes; the bank-PU ISA drops VMUL32/ADD64M/VXSUM.

**Validation** (`pim/results/tree_sweeps_mau.txt`; trace-level model: cl=0
GGM_REDUCE = pure memory traffic, since the MAU pipeline at 1 op/CK never binds
against the 32B/4CK CCDL column stream):

| config | ns/leaf |
|---|---|
| expansion-only floor (PU, no leaf_convert) | 0.388 |
| **two-tier: PU expand + MAU leaf_convert (overlapped)** | **0.400** |
| single-tier baseline: PU does both | 0.450 |

The MAU stream overlaps the next instance's expansion (double-buffered leaf
region; issue-order matters — prepend + round-robin across channels, see the
placement iterations in the results file). The residual +5% over the floor is
**physical bank contention** (ACT count 2.4×): with 2 banks/channel the MAU's
leaf reads and g writes share both banks with the deepest expansion level. A
third bank per channel (dedicated leaf region) is the org lever to recover it;
0.39 was the zero-contention estimate, **0.40 is the honest 2-bank number** —
an 11% DPF throughput gain over single-tier, plus the (bigger) implementability
win of removing all multipliers from the DRAM die. `coexec_model.py --mau`
uses the measured 0.401.

## 8. Shared-HBM contention, cycle-level (Level-1 measurement)

The one term of this model that had remained purely analytic — contention
between the PIM lanes and the SM's NTT traffic inside the shared HBM — now has
a cycle-level measurement (`pim/results/tree_sweeps_smstream.txt`). The SM's
per-pair HBM traffic (read g, read a-hat, write z) was injected as a third
traffic class into the same Ramulator2 trace, arbitrating in the same bank
queues as the expansion and MAU streams:

| config (n=12, I=256) | ns/leaf | delta |
|---|---|---|
| expansion + MAU | 0.4010 | --- |
| + SM stream (all three classes) | 0.4424 | **+10.3%** |
| SM stream alone (drain reference) | 0.0372 equiv | --- |

Two results:

1. **The external-bus thesis stands untouched.** Every observed effect is
   bank/issue-side inside the stack; the HBM-to-SM interface itself never
   saturates (SM demand is 2-3% of per-channel bandwidth).
2. **+10% is an upper bound, and the gap to the ~3% lower bound is a
   simulator-structure artifact, identified precisely:** this frontend is a
   single in-order FIFO with per-channel backpressure, so the SM stream is
   forced through the PIM host-command path and serializes with it (the
   write-conservation check and the (b)-(a) = (c) identity prove it adds its
   full drain time). A real GPU's SM requests enter the memory controllers
   out-of-band and compete only for bank slots (~2-3% share). The model takes
   `--contention` in [0.03, 0.10]; collapsing the interval to a point value is
   exactly the phase-2 co-simulation task (Accel-Sim SMs driving Ramulator2
   controllers directly).

Bank-partitioning note: four row regions (ping-pong, leaf, g/a-hat, z) across
only 2 banks/channel is the structural pressure behind both this and the MAU's
+5%; the 4-bank org variant remains the single highest-value lever.

### Caveats / provenance
* The GPU NTT table is measured on **RTX 5000 Ada**; the DPF PIM number is a 64-PU HBM3 **model**
  calibrated to Ramulator2. The target here is a *hypothetical HBM3-PIM GPU*; the GPU lane is kept
  as a **separate device axis** with an explicit `--sm-scale` re-pointing factor rather than
  pretending the RTX/L40S numbers are H100 silicon.
* `fwd ≈ full/3` is an estimate; it can be replaced with a pure-forward measurement via the 4-step
  stage timer (`PCG_4STEP_STAGE_MS=1`) without changing the model structure.
* Per-pair two-party CW/seed network round-trips are a hideable term (overlapped with the previous
  pair) and are reported separately, not folded into either compute lane.

## 9. Superiority map v1 (full (N,t,PU) plane, Expand phase)

`pim/tools/superiority_map.py` → `pim/results/superiority_map_v1.txt`: design
(two-tier PIM + SM NTT + stage-1 offload, contention=0 under the bank-partition
placement premise of §8) vs the strongest L40S all-GPU baseline (kernel-only),
N∈{2¹⁶..2²⁴} × t∈{4..32} × PU∈{64,256,1024}. Headline: **1.9–2.3× @64PU,
6.6–8.3× @256PU, 9.1–30.6× @1024PU** across the whole plane; the 1024-PU dip at
small t is the NTT term (offload-limited by f₁ shrinking at large N). Two
provenance flags carried per cell: `p` = NTT point awaiting L40S silicon (the
`endtoend/run_largeN_bench.sh` watcher lands them automatically when a GPU
frees), `W` = capacity wall (double-buffered leaf region ≥ one bank) where the
block-streaming schedule is required and its overhead is not yet measured —
those speedups are optimistic until step-4 lands.


## Addendum: the five software-scheduler optimizations (new-algorithm PIM, measured)

Coupling architecture -- three calibration layers composed by a discrete-event
timeline (pim/tools/sched_timeline.py); **Accel-Sim is NOT used** (the merge
backend's fwd/pointwise/INTT are separable kernel launches -> L40S silicon
stage-split beats replay; Accel-Sim/Ramulator2 cannot co-simulate, and the
shared-DRAM contention is already modeled on the correct (bank) side by the
Ramulator2 stream injection):

| lane | source |
|---|---|
| PIM  | Ramulator2 chacha campaign (per-round = total/rounds; round op-mix uniform) |
| SM   | L40S silicon merge stage split (`PCG_MERGE_STAGE_MS=1`, batch=16): fwd share 0.28 of full mul (old heuristic said 1/3) |
| NET  | RTT parameter {0, 50us, 2ms}; separate lane per this doc's convention |

Schedules S0..S5 (cumulative): serial -> block pipeline -> +Move2 streamed NTT
(a-hat precomputed; per block ONE fwd(g_ij)+pointwise-MAC, single INTT at end)
-> +per-round Beaver opens -> +doorbell round consumption -> +parity discipline.

Measured findings (pim/results/sched_timeline.txt, newalgo_speedup_streamed.txt):
1. **Move2 (opt-2) is the big lever on NTT-bound cells**: B200 small-N cells go
   4.2-5.0x -> 14.5-38.6x; every NTT-bound cell flips to DPF-bound. DPF-bound
   cells (all GDDR, large t) unchanged, as predicted.
2. **Doorbell C-consume is FREE on the PIM side**: f_opp = f_same = 0.00%
   (L40S sim, trickled stream; solo drain = 10% of the compute window, fully
   hidden). The earlier +5% contention number was the much larger NTT g/a/z
   traffic, not the 8B C stream. Parity discipline (opt-5) is therefore a
   correctness requirement of the double buffer, not a cycle optimization.
3. **Per-round Beaver (opt-3) matters exactly when RTT is large**: at 2ms WAN
   RTT it recovers most of the block-pipeline loss (e.g. L40S 2^18 t=4:
   7.30 -> 4.02 ms); at <=50us it is a no-op (network already hidden).
4. **Trickle discipline is load-bearing**: dumping the C-consume stream as a
   block behind the fused level serialized the in-order frontend queue
   (measured 2.57x blowup); interleaving it between the level's ops restored
   full hiding. Same lesson as the MAU drain, now re-verified on the new path.


### AGU-layout claim verified on L40S silicon (agu_layout_bench)

Claim: if the channel-MC AGU emits g in the 4-step transform's native
(transposed, tile-column-major) layout, the SM forward path drops its input
GPU_Transpose pass. A/B on L40S, same production kernels, outputs BITEXACT
at every size (the NTT kernel cannot tell whether its transposed input came
from a transpose kernel or was born in that layout):

| N     | armA transpose+NTT | armC AGU-layout NTT | AGU saves |
|-------|-----|-----|-----|
| 2^14  | .0157 ms | .0128 ms | 18.3% |
| 2^16  | .0400 | .0341 | 14.8% |
| 2^18  | .214  | .150  | 29.9% |
| 2^20  | 1.657 | 1.233 | 25.6% |
| 2^22  | 6.90  | 4.98  | 27.8% |
| 2^24  | 31.7  | 23.5  | 25.8% |

Cross-check: PCG_4STEP_STAGE_MS ladder puts ALL transposes at 26-32% of the
full 4-step mul -- consistent (E2 isolates the forward input-side one).
Hardware cost side (already argued): the pi_tile bit-field swap lives in the
channel MC's command sequencer (tens of gates + 2 CFR fields), DRAM die
unchanged; tile-granular swizzle keeps writes full-column and ACT-count
neutral. This is M4c of the indirect-PIM-helps-NTT argument, now silicon-
grounded end to end.
