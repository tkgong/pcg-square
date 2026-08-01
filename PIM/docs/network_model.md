# Network model of the PCG-OLE 2PC protocol (code-extracted)

Symbolic communication model derived by code review of the CPU reference
implementation, with the instrumentation that calibrates its constants.
Every claim below carries a `file:line`. Companion artifacts:

- instrumentation: `common/comm_trace.h` (+ traced sites listed in §6)
- calibration data: `pim/results/comm_calib_*.txt`
- model/figures: `pim/tools/net_model.py`

## 1. One-sentence summary

> For each tree level, every batch of `B = t²` DPF instances locally expands
> its frontier, exchanges `B·(m−1)·16` bytes of correction material in one
> **party-serialized** round trip, and applies the received information before
> the next level can begin. Communication inside a batch is therefore
> latency-bound and strictly on the critical path; it can only be hidden by
> overlapping independent batches.

Every quantity in that sentence is traceable: `B` (`pcg_ole_impl.h:115`),
`m−1` (`dpf.cpp:521`), `16` = `sizeof(emp::block)`, party-serialization
(`dpf.cpp:535-543`), blocking dependency (§4).

## 2. Call graph

```
gen_and_expand                                   pcg_ole_impl.h:113
└── for ii,jj in [0,c)²                          pcg_ole_impl.h:113-114
    ├── 3a Kogge-Stone position add              pcg_ole_impl.h:145  → circuit.cpp:260
    ├── 3b Gilboa payload mult + zp_triple       pcg_ole_impl.h:169,174
    ├── 3c batch_F_DeltaShiftShare               pcg_ole_impl.h:187  → dpf.cpp:257
    │      setup_params (local, no comm)         dpf.cpp:288-295
    ├── 3c batch_gen_full_eval                   pcg_ole_impl.h:309
    │      for i in [0, levels):                 dpf.cpp:503
    │        expand → local CW share → EXCHANGE → derive CW → apply
    │                                            dpf.cpp:511,523,534,548,555
    └── 3d output layer: 2 opens (d,e then CW)   pcg_ole_impl.h:362,391
```

**Full-domain evaluation has zero communication.** `HalfTreeDPF::eval`
(`dpf.cpp:301-326`) contains no `io_` reference; inside
`batch_gen_full_eval` the only IO in the level body is the CW exchange
(`dpf.cpp:534-543`). All expansion work is local.

## 3. What is sent (per phase)

`B = t²`, `m = 2^w`, `levels = dpf_n / w`, `dpf_n = log₂(2N/t)`,
`off_bits = log₂(N/t)`, `A_ks`/`A_ds` = AND-gate counts of the Kogge-Stone
adder (`adder.h:27-50`) and of the delta-shift circuit (`dpf.cpp:218-240`).

| Phase | bytes per direction | rounds | scaling | site |
|---|---|---|---|---|
| Kogge-Stone triples | `64·B·A_ks + 16` | 1 | `c²t²·log(N/t)` | `circuit.cpp:268` |
| KS input masks | `B·off_bits` | 1 | `c²t²·log(N/t)` | `circuit.cpp:296-304` |
| KS AND levels | `2·B·A_ks` total | `S+1`, `S=⌈log₂ off_bits⌉` | rounds ∝ log log N | `circuit.cpp:333-346` |
| Gilboa payload + triple | `3·(32·62·B + 16)` | 3 | `c²t²` | `circuit.cpp:97-132` |
| **Delta-shift triples** | **`64·B·A_ds + 16`** | 1 | `c²t²·log(N/t)·(2^w−1)(127+w)/w` | `circuit.cpp:268` |
| DltSft masks + AND levels | `B(dpf_n+128)`, `2·B·A_ds` | `⌈log₂w⌉+1` | | `circuit.cpp:280-346` |
| **DPF-Gen CW** | **`16·B·(m−1)` per level** | **`levels`** | rounds ∝ log N | `dpf.cpp:534-543` |
| Beaver `d,e` open | `16·B` | 1 | `c²t²` | `pcg_ole_impl.h:362` |
| Beaver CW open | `8·B` | 1 | `c²t²` | `pcg_ole_impl.h:391` |
| FerretCOT setup | ~0.3–0.6 MB | ~10–15 | one-time (cached in `data/`) | `ferret_cot.hpp:100-140` |

Rounds per `(ii,jj)` pair:
`levels + ⌈log₂⌈log₂(N/t)⌉⌉ + ⌈log₂ w⌉ + 11`, times `c²`.

**Message-size model: constant, not frontier-dependent (Model A).**
`total_cws = B·(m−1)` (`dpf.cpp:521`) does not depend on the frontier size
`sz = m^i` (`dpf.cpp:504`): the frontier is collapsed by the XOR reduction at
`dpf.cpp:526-527` *before* the send. Hence `M_l` is identical on every level
and DPF-Gen is latency-sensitive, not bandwidth-sensitive.

**Byte- vs latency-dominance split.** Delta-shift AND-triple OTs carry ~93 %
of the bytes (they scale with `t²·log(N/t)`), while DPF-Gen contributes ~0.2 %
of the bytes but `levels` of the round trips. The two bottlenecks live in
different phases — a bandwidth-only or latency-only model gets the wrong
answer.

**`w` is the pivot knob:** raising `w` divides DPF-Gen rounds by `w` but
multiplies delta-shift bytes by `(2^w−1)/w`.

## 4. Blocking dependency (why the network is on the critical path)

Within one batch, level `l+1` cannot start before level `l`'s exchange
completes:

```
E(l) ─▶ P(l) ─▶ N(l) ─▶ C(l) ─▶ E(l+1)
expand   pack   exchange  apply-CW
```

Evidence chain: `their_cws` is first read at `dpf.cpp:550` → feeds `CW[k]`
→ used to build `next` at `dpf.cpp:560-563` → stored into `trees[b]`
(`dpf.cpp:565`) → read by the level-`l+1` hash at `dpf.cpp:511`. Formally
`S(E_{l+1}) ≥ F(N_l) + T_corr,l`.

No overlap exists in the reference implementation: no `std::thread`,
`std::async` or futures anywhere in `half_tree_dpf/`, `pcg_ole_2pc/`,
`bool_circuit/`; a single `my_cws`/`their_cws` pair is allocated per level
(`dpf.cpp:522,534`), so at most one message is outstanding. The GPU path is
the same shape — synchronous D2H → exchange → H2D → kernel
(`common/dpf_gpu.cu:450-461`).

## 5. Transport mechanics (grounding α and β)

| Fact | Site |
|---|---|
| `send_block(p,n)` → `send_data(p, n·16)` | `emp-tool/io/io_channel.h:24-26` |
| `send_data` writes into a **1 MiB stdio buffer** (`_IOFBF`) | `net_io_channel.h:82-84,121-131`, `constants.h:7` |
| Small messages therefore go on the wire only at the explicit `flush()` | `dpf.cpp:537,542` |
| `recv_data` auto-flushes if anything was sent | `net_io_channel.h:134-136` |
| `TCP_NODELAY` set (Nagle off) | `net_io_channel.h:80,107-110` |
| Blocking sockets, one connection shared by OT + circuits + DPF + opens | `net_io_channel.h:46-78`; `circuit.cpp:22-23` |
| **Party-serialized exchange** costs ≈ `2(α + M/β)`, not `1×` | `dpf.cpp:535-543` |
| `IOChannel::counter` counts **sent** bytes only (asymmetric: ALICE is the OT sender) | `io_channel.h:11-13` |

The GPU build does not change any of this: only per-instance sums (`2B`
uint64) and reconstructed CWs (`B` uint64) cross the wire, from **host**
buffers, identical in size to the CPU path (`pcg_ole_impl.h:239-288`,
`common/leaf_convert_cuda.h:4-16`).

## 6. Instrumentation

`common/comm_trace.h`, gated by `-DPCG_ENABLE_PROFILING` (expands to the
original block otherwise). One CSV row per exchange:
`party,context,phase,level,batch,send_bytes,recv_bytes,t_begin_ns,t_mid_ns,t_end_ns,io_counter_delta`.

Traced sites: `dpf_gen` (`dpf.cpp:534`), `dpf_gen_gpu` (the two exchange
lambdas), `circuit_and` (`circuit.cpp:333`), `beaver_de` / `beaver_cw`
(`pcg_ole_impl.h:362,391` and the device-path twins). Phase zones
`PRE_KS / PRE_GILBOA / PRE_DLTSFT / DPF_GEN` added in `common/prof.h`;
`PCG_PROF_REPORT()` is now actually called (`bench_pcg_ole_2pc.cpp`), and the
bench prints per-iteration `sent_bytes` (protocol only, verification
excluded).

Historical note: before this change the `DPF_EXPAND_COMM` / `NET_WAIT` zones
covered **only** the two Beaver opens — the `levels` DPF-Gen round trips were
never counted, and the comment claiming otherwise (`pcg_ole_impl.h:307`) was
stale.

## 7. Reconciliation (smoke, N=1024, c=2, t=4, w=1 → B=16, levels=9)

| check | expected | measured |
|---|---|---|
| `dpf_gen` rows | `c²·levels = 36` | 36 |
| `dpf_gen` bytes/row | `B(m−1)·16 = 256` | 256 (identical on all 9 levels) |
| `beaver_de` / `beaver_cw` rows | `c² = 4` each | 4 / 4 |
| `beaver_de` bytes | `2B·8 = 256` | 256 |
| `beaver_cw` bytes | `B·8 = 128` | 128 |
| `io.counter` delta | = `send_bytes` per row | equal |

Party asymmetry measured on the same run: ALICE sends 5,997,712 B vs BOB
193,456 B (31×) — ALICE is the OT sender in every triple batch
(`circuit.cpp:38,99`). Any half-duplex link model must use the *asymmetric*
per-direction volumes.
