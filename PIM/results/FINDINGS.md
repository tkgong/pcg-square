# GGM/DPF-on-HBM2-PIM — Phase 6 findings

Config: N=256, t=4, c=2 → D=128, dpf_n=7 levels, B=16 instances, 4 pairs, 1152 GGM ops
(9216 read + 18432 write column-words). HBM2_4Gb, compute_latency=128 (ChaCha8 on a 16b FPU).
Timing-only model; bank-spread; per-PIM-unit FPU serialization (2 banks/FPU).

## 1. Per-level breakdown (`per_level.csv`)
Cycles scale with tree depth — the deep levels dominate (level 6 alone = 55k of the work, the
3 smallest levels are flat at the per-instance floor). Adding ChaCha compute (128 cyc/op) moves
total cycles by only **0.2–2.7%** → at this scale the workload is **store/memory-bound, compute
is hidden**.

## 2. Store-long-pole (`store_long_pole.csv`)
Distributing child writes across banks matters:

| bank mode | cycles | ACTs |
|---|---|---|
| concentrate (1 bank/instance) | 190703 | 5470 |
| spread (round-robin 16 banks) | 123760 | 4156 |

→ **spread is 1.54× faster** — bank-level parallelism on the stores is real, validating the
1:2 store-leaning thesis.

## 3. Channel-count crossover (`channel_sweep.csv`)
More HBM channels barely help (1→16 ch: 123595→109642, ~1.13×) and it stays memory-bound.
So the bottleneck is **not** channel bandwidth.

## Root cause: row-conflict bound
The baseline issues each op as **load(row_in) → store(row_out)** on the same bank, two different
rows → **3.6 ACTs/op** (4156 ACTs / 1152 ops). Row-cycle (tRC) management dominates; both the
ChaCha compute and extra channel bandwidth hide under it.

## Implication / next step
The dominant cost is the per-op load/store **row thrash**, not compute and not channel BW.

## UPDATE — double-buffer refactor (solves the row thrash)
Added `write_bank` to the op + a `double_buffer` bank mode: each instance is pinned to one PIM
unit (2 banks A,B); per level the read/write banks **ping-pong** (read A→write B, next level
read B→write A), so load and store never share a row buffer. The generator also **interleaves
instances within a level** so consecutive ops hit different units (full bank parallelism).

Result (N=256, t=4, c=2; memory-only):

| mode | cycles | ACTs |
|---|---|---|
| concentrate | 172252 | 3898 |
| spread | 123595 | 4219 |
| **double_buffer** | **120696** | **2854** |

- **ACTs −32%** vs spread (4219→2854) — the row conflict is largely removed.
- **Channel scaling restored**: 1→16 channels now gives **10.9×** (120696→11053) vs only ~1.13×
  before (the row conflict had serialized everything).
- **Compute regime now reachable**: at 16 ch (mem floor 11053), sweeping compute_latency shows
  memory-bound at 0–128, then **compute-bound** at ≥512 (37k→590k). So whether DPF-on-PIM is
  compute- or memory-bound is now correctly a function of (compute_latency, channels, FPU sharing),
  not an artifact of row thrash.

Bottleneck progression: **row-conflict (baseline) → single-channel command bus → near-linear
channel scaling**, with the ChaCha FPU as the ceiling once memory is parallel enough.

## UPDATE 2 — aliasing fix (independent channel/unit hashing)
The generator originally derived both channel and PIM-unit from one instance hash, so
`unit == ch % NUM_UNITS` when channels is a multiple of NUM_UNITS → only 1 of 8 FPUs per channel
used (7/8 idle). Fixed by hashing channel from `gid % channels` and the unit from `gid // channels`
(independent). Result — channel scaling went from ~linear-ish to **near-perfect linear**:

| channels | before fix | after fix | speedup vs 1ch |
|---|---|---|---|
| 1 | 120696 | 120696 | 1.0× |
| 2 | 60234 | 58533 | 2.06× |
| 4 | 34061 | 29942 | 4.03× |
| 8 | 23414 | 16067 | 7.51× |
| 16 | 11053 | **7968** | **15.15×** |

Memory floor at 16ch dropped 11053→7968 (more banks/FPUs utilized); compute crossover unchanged
(compute=128 hidden at 8147, compute≥512 → compute-bound). This is the model approaching the ideal
**internal-bandwidth-bound** regime (all banks streaming in parallel — the "good" memory-bound that
justifies PIM), with the ChaCha FPU as the ceiling only once compute/op is large.

## UPDATE 3 — complete DPF (leaf reduction added)
Implemented the OTHER half of the DPF per the reference (`PCG-acceleration/pcg_ole_2pc/pcg_ole_impl.h`,
lines 169-257): a new `ISR_GGM_REDUCE` op (read leaves + per-leaf 62-bit modmul + write g, with a
CONFIGURABLE read:write ratio, unlike EXTEND's fixed 1:2). The generator's `--with-reduction` appends,
per (i,j) pair: (1) **sum pass** (read all leaves), (2) **scale+scatter** (read leaves again + modmul
each + write g), (3) **2N→N fold**. Leaves are read TWICE (sum, then scale-after-network).

Result (N=256, t=4, c=2, 8 channels, double_buffer):

| stage | cycles |
|---|---|
| expansion only | 16168 |
| + reduction, modmul=0 (memory floor) | 43166 |
| + reduction, modmul=20/leaf | 47248 |
| + reduction, modmul=50/leaf | 54928 |

- The reduction **~2.7×'s** the runtime even at modmul=0 → dominated by **reading the leaves twice**
  (sum + scale). The leaf memory is the bottleneck, as predicted.
- The 62-bit modmul is a real secondary cost (+9% at 20/leaf, +27% at 50/leaf) and — unlike ChaCha —
  does NOT SIMD-amortize well, so it becomes compute-bound at high per-leaf cost / tight channels.
  This is the one place microarch compute optimization (Barrett/Montgomery) actually pays.

## UPDATE 4 — fuse the reduction (and its coupling to the modmul)
`--fuse`: never materialize leaves. Skip the leaf-writing last expansion level; the reduction
re-expands the (smaller) level-(L-1) parents on the fly — pass1 (expand+sum), [CW], pass2
(re-expand + per-leaf modmul + scatter). Leaves are never written/read; cost is 2× last-level ChaCha.

Result (N=256, t=4, c=2, 8ch, double_buffer), sweeping the 62-bit modmul cost:

| modmul/leaf | materialize | fused |
|---|---|---|
| 0   | 43166 | 44025 |
| 20  | 47248 | 44186 |
| 50  | 54928 | 45146 |
| 100 | 67728 | 46746 |

The headline is the **coupling**, not a flat speedup:
- **Materialize** is very modmul-sensitive (+57% from modmul 0→100): the modmul *stacks on top of* the
  leaf-read memory.
- **Fused** is nearly modmul-insensitive (+6%): the modmul **hides under the re-expand ChaCha** (the
  FPU is already busy, so the modmul rides along free). At modmul=0 it's a wash (fuse just swaps leaf
  memory for re-expand compute); at modmul=100 fuse is **31% faster**.

→ **Whether to fuse depends on whether the modmul is co-designed.** Cheap modmul (Barrett/Montgomery)
→ materialize is fine and simpler. Expensive modmul (naive 62-bit on a 16b FPU) → fuse wins by
absorbing it. (A third option — store leaves once + fuse only the sum, no re-expand — would lower the
memory floor without the re-expand compute; not yet built.)

Next: the store-once fuse variant, then GPU-NTT contention overlay + an energy model (the paper).
