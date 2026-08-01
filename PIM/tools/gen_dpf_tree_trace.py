#!/usr/bin/env python3
"""
REAL GGM/DPF tree-dataflow trace generator (inter-layer read-back).

Unlike gen_dpf_trace.py (independent batches) and the PCG-acceleration toolkit
(synthetic per-level regions, "children need not chain"), this generator emits a
full n-level tree where LEVEL i's CHILD WRITES ARE EXACTLY LEVEL i+1's PARENT READS:

  layout (word-major / seed-parallel): column j = word j of 8 seeds
    -> a frontier of P seeds occupies P column-words; op = 8 seeds:
       read 8 cols (parents) -> ChaCha8+DPF (compute_latency) -> write 16 cols (children)
  chaining: per (channel, buffer-pair) two banks ping-pong:
    level i reads bank (i%2) at linear col-word offsets [0, P_i),
    writes 2*P_i child col-words to bank ((i+1)%2) at offsets [0, 2*P_i);
    level i+1 op k reads offsets [8k, 8k+8) of that bank -- the exact rows/cols written.
    offset o -> row = o // 64, col = o % 64  (64 cols/row; 8-col reads and 16-col
    writes are aligned so no op crosses a row boundary).
  levels are separated by SYNC (level-synchronous dataflow: level i+1 cannot start
  before level i's children exist; also prevents the timing-only sim from reordering
  the read-after-write across levels).

Modes:
  --mode subtree    one tree of depth n on C channels: levels 0..log2(C)-1 expand on
                    ch0 (under-parallel, counted honestly); at level log2(C) each
                    channel takes one node as subtree root and expands depth n-log2(C)
                    independently (root hand-off via host noted, ~C col-words, negligible).
  --mode instances  I independent trees of depth n (PCG: I = c^2*t^2, n = log2(2N/t));
                    instance m -> channel m % C, sequential per channel, all channels parallel.

Leaf reduction (--reduce fused): after the last level, one pass re-reads all 2^n leaf
col-words in row-chunks (<=64 cols/op) with per-leaf modmul charged (GGM_REDUCE),
writing g (leaves/8 col-words) -- the store-once/fused leaf_convert model.

Trace line:  AiM GGM_EXTEND <opsize> <ch> <bank> <row_in> <row_out> <cl> <col_in> <col_out> <wbank> <nbanks>
             AiM GGM_REDUCE <opsize> <ch> <bank> <leaf_row> <g_row> <cl> <col_in> <col_out> <wbank> <wrsize> <nbanks>
"""
import argparse
import sys

COLS_PER_ROW = 64          # PIM64_2Banks: Co = 1<<6
SEEDS_PER_OP = 8           # 8 lanes (seed-parallel)
# Set by --seed-bits in main(): dpf.cpp reference uses 128b emp::block seeds (4 words),
# the 256b variant matches the earlier ChaCha-key-width model. ChaCha block-function
# count per node is identical; only memory traffic and PACKLR32 count change.
RD_COLS = 4                # words per seed (= parent cols per 8-seed op)
WR_COLS = 8                # 2 children x words-per-seed


REREAD = False   # set by --reread: model child-R parent re-read -> 2*RD reads, WR writes

def emit_extend(f, ch, rbank, row_in, col_in, wbank, row_out, col_out, cl, nbanks=1):
    if REREAD:
        # register-exact datapath: child R = parent ^ L needs the ORIGINAL seed, but the
        # 16 state regs are scrambled by the rounds -> re-read the 4 seed cols from the
        # still-open row. Emit as GGM_REDUCE (decouples reads=opsize from writes=wrsize):
        # 2*RD_COLS reads (4 seed + 4 re-read) + WR_COLS writes + compute on last read.
        f.write("AiM GGM_REDUCE %d %d %d %d %d %d %d %d %d %d %d\n"
                % (2 * RD_COLS, ch, rbank, row_in, row_out, cl, col_in, col_out, wbank,
                   WR_COLS, nbanks))
    else:
        f.write("AiM GGM_EXTEND %d %d %d %d %d %d %d %d %d %d\n"
                % (RD_COLS, ch, rbank, row_in, row_out, cl, col_in, col_out, wbank, nbanks))


def emit_reduce(f, ch, bank, leaf_row, col_in, ncols, g_row, g_col, wrsize, cl, nbanks=1):
    f.write("AiM GGM_REDUCE %d %d %d %d %d %d %d %d %d %d %d\n"
            % (ncols, ch, bank, leaf_row, g_row, cl, col_in, g_col, bank ^ 1, wrsize, nbanks))


def level_ops(parents):
    """ops to expand a frontier of `parents` seeds (= parents column-words)."""
    return max(1, (parents + SEEDS_PER_OP - 1) // SEEDS_PER_OP)


def expand_level(f, ch, level, parents, cl, stats):
    """Emit one level's EXTEND ops for a frontier of `parents` seeds on channel ch.
    Reads bank level%2 offsets [0,parents), writes bank (level+1)%2 offsets [0,2*parents)."""
    rbank, wbank = level % 2, (level + 1) % 2
    n_ops = level_ops(parents)
    for k in range(n_ops):
        off_in = k * RD_COLS
        off_out = k * WR_COLS
        emit_extend(f, ch, rbank,
                    off_in // COLS_PER_ROW, off_in % COLS_PER_ROW,
                    wbank, off_out // COLS_PER_ROW, off_out % COLS_PER_ROW, cl)
    stats[level] = stats.get(level, 0) + n_ops
    return n_ops


ADD64M_CL = 4    # 64b mod-P accumulate per 8-leaf group (lane-local: add lo/hi + cond-sub)
VXSUM_CL = 4     # cross-lane 8->1 reduction at end of sum pass (new instruction, tree adder)


def reduce_leaves(f, ch, n_leaf_cols, leaf_bank, modmul_cl, stats, mau=False,
                  row_base=0, nbanks=1):
    """Two-pass leaf_convert (pcg_ole_impl.h:171-232), SIMD-honest accounting.
    Only the LOW 64b of each 128b leaf is consumed (block_to_uint64) -> both passes
    read 2 cols per 8-leaf group (w0,w1), i.e. n_leaf_cols/2 columns each.
      Pass 1 (sum S_k):   read cols + lane-local 64b mod-add (ADD64M_CL per group)
                          + ONE VXSUM (8->1 cross-lane) per subtree at the end.
      [host: exchange S, CW_k = beta/v -- t values, 1 round, not in sim]
      Pass 2 (scale+scatter): re-read cols + per-GROUP Barrett modmul: the 32b
                          instruction sequence is lane-parallel, so modmul_cl(=32)
                          covers 8 leaves at once (NOT per leaf -- earlier accounting
                          missed the SIMD) + ADD64M accumulate + write g (2 cols/group).
    mau=True: the channel-level MAU (base-die, fused 1-op/CK pipeline, sparse-prime
    reduction) consumes the SAME column streams; its compute never binds (pipeline
    1/CK >> CCDL-paced 32B/4CK column stream), so cl=0 on every op -- the bank PU
    is untouched (cl=0 GGM_REDUCE never charges the FPU, AiM_dram_controller:157,
    322-324) and only the memory traffic remains, exactly the MAU's cost."""
    used_cols = max(1, n_leaf_cols // 2)             # only w0,w1 of each group
    n_ops = 0
    # ---- pass 1: sum ----
    done = 0
    while done < used_cols:
        w = min(COLS_PER_ROW, used_cols - done)
        groups = max(1, w // 2)
        cl = groups * ADD64M_CL
        if done + w >= used_cols:
            cl += VXSUM_CL                           # final 8->1 cross-lane reduce
        emit_reduce(f, ch, leaf_bank, row_base + done // COLS_PER_ROW,
                    done % COLS_PER_ROW, w,
                    g_row=row_base, g_col=0, wrsize=1, cl=0 if mau else cl,
                    nbanks=nbanks)
        done += w
        n_ops += 1
    # ---- pass 2: scale + scatter ----
    done = 0
    while done < used_cols:
        w = min(COLS_PER_ROW, used_cols - done)
        groups = max(1, w // 2)
        emit_reduce(f, ch, leaf_bank, row_base + done // COLS_PER_ROW,
                    done % COLS_PER_ROW, w,
                    g_row=row_base, g_col=0, wrsize=w,  # g out: 2 cols/group = w cols
                    cl=0 if mau else groups * (modmul_cl + ADD64M_CL),
                    nbanks=nbanks)
        done += w
        n_ops += 1
    stats["reduce"] = stats.get("reduce", 0) + n_ops
    return n_ops


def gen_subtree(f, n, channels, cl, modmul_cl, reduce_mode):
    """One tree of depth n split over `channels` at level s=log2(channels)."""
    s = max(0, (channels - 1).bit_length())          # log2(C) for power-of-2 C
    stats = {}
    total = 0
    # shallow levels 0..s-1 on ch0 (frontier 1..2^(s-1)), chained
    for lvl in range(min(s, n)):
        total += expand_level(f, 0, lvl, 1 << lvl, cl, stats)
        f.write("AiM SYNC\n")
    # NOTE: level-s roots hand off to per-channel banks via host (C col-words, negligible).
    # deep levels: each channel expands its own subtree of depth n-s, all in lockstep
    for d in range(n - s):                            # per-channel subtree level d
        lvl = s + d
        parents = 1 << d                              # per-channel frontier
        per_ch = level_ops(parents)
        for k in range(per_ch):                       # interleave channels per op index
            off_in, off_out = k * RD_COLS, k * WR_COLS
            for ch in range(channels):
                emit_extend(f, ch, d % 2,
                            off_in // COLS_PER_ROW, off_in % COLS_PER_ROW,
                            (d + 1) % 2, off_out // COLS_PER_ROW, off_out % COLS_PER_ROW, cl)
        stats[lvl] = stats.get(lvl, 0) + per_ch * channels
        total += per_ch * channels
        f.write("AiM SYNC\n")
    if reduce_mode in ("fused", "mau") and n >= s:
        # subtree mode has no adjacent instance to hide behind: mau here is a
        # SERIAL tail (cl=0 memory stream after expansion) -- an upper bound.
        leaf_cols = (1 << (n - s)) * RD_COLS // SEEDS_PER_OP   # per channel
        leaf_bank = (n - s) % 2
        for ch in range(channels):
            total += reduce_leaves(f, ch, leaf_cols, leaf_bank, modmul_cl, stats,
                                   mau=(reduce_mode == "mau"))
        f.write("AiM SYNC\n")
    return total, stats


AA_ROW_BASE = 1 << 11    # a-hat spectra region (SM reads; resident, static rows)
Z_ROW_BASE = (1 << 11) + 512   # z accumulation region (SM writes)
NTT_ROW_BASE = (1 << 11) + 1024  # MAU NTT stage-1 working region (case-A)


def mau_ntt_lines(n, ch, bank_base, parity, factor, stats):
    """case-A: the MAU also runs NTT stage-1 column NTTs -- its butterfly
    column traffic (read a/b blocks + write back, equal volume) shares the
    channel with the DPF streams. Same modeling device as the MAU leaf stream
    (cl=0 REDUCE, no PU charge). Volume = factor x the leaf stream's column
    count (2 passes x used_cols); factor sweeps {0.5, 1, 2} bound the ratio
    of stage-1 work to leaf-convert work per round."""
    import io
    used = max(1, ((1 << n) * RD_COLS // SEEDS_PER_OP) // 2)
    vol = max(1, int(2 * used * factor))       # columns read (= columns written)
    base = NTT_ROW_BASE + parity * 256
    b = bank_base + (n % 2)
    buf = io.StringIO()
    n_ops = 0
    done = 0
    while done < vol:
        w = min(COLS_PER_ROW, vol - done)
        emit_reduce(buf, ch, b, base + done // COLS_PER_ROW, done % COLS_PER_ROW,
                    w, g_row=base + 128 + done // COLS_PER_ROW, g_col=0,
                    wrsize=w, cl=0)
        done += w
        n_ops += 1
    stats["mau_ntt"] = stats.get("mau_ntt", 0) + n_ops
    return buf.getvalue().splitlines(keepends=True)


def c_consume_lines(n, ch, parity, stats, bank_base=0, scale=1, cl=0):
    """SM doorbell consumption of round-`parity`'s C region (the chacha fused
    CONVERT output: 8B/leaf mod-P residues). Reads bank n%2 (where the fused
    op writes -- NOT the MAU g bank (n%2)^1), double-buffered rows
    (1<<10) + parity*256. cl=0 REDUCE reads (no PU charge), write artifact
    parked in Z. Volume = 2^n/4 cols/instance (= the 8B/leaf stream).
    Models opt-4 (doorbell round-level consume); parity chooses opt-5's
    good case (opposite: SM reads round r-1 while PIM writes round r) vs
    bad case (same: colliding with the round being produced)."""
    import io
    used = max(1, ((1 << n) * RD_COLS // SEEDS_PER_OP) // 2)
    cb = bank_base + (n % 2)               # bank the fused CONVERT writes C into
    c_base = (1 << 10) + parity * 256      # C region double buffer
    buf = io.StringIO()
    n_ops = 0
    # scale = injected NTT pass count (1 = legacy single pass over the C region)
    for rep in range(max(1, scale)):
        done = 0
        while done < used:
            w = min(COLS_PER_ROW, used - done)
            emit_reduce(buf, ch, cb, c_base + done // COLS_PER_ROW, done % COLS_PER_ROW,
                        w, g_row=Z_ROW_BASE + parity * 256, g_col=0, wrsize=1, cl=cl)
            done += w
            n_ops += 1
    stats["smc"] = stats.get("smc", 0) + n_ops
    return buf.getvalue().splitlines(keepends=True)


XP_SRC_ROW = (1 << 11) + 1024   # DRU xpose source-tile region (row = tile id)
XP_DST_ROW = (1 << 11) + 1536   # DRU xpose destination-tile region


def xpose_lines(n, ch, parity, stats, pus, pattern=True, cl=0, scale=1):
    """True DRU XPOSE access pattern (pattern=True) or its linear-matched
    control (pattern=False), as cl=0 REDUCE streams at ONE COLUMN PER
    REQUEST -- the DRU's real request granularity (one 32B gather / write
    per bus beat; the earlier C-consume point batched 64 cols per ISR).
      pattern: per circle 4 reads of the SAME col from 4 CONSECUTIVE banks
               ((4K+J) mod 16) of the tile's source row, then 1 column write
               to the rolling destination bank (U mod 16) in the tile's
               destination row. Row switches every TILE_CIRC circles =
               per-tile ACT cadence (1 rd + 1 wr row per bank per tile).
      linear : identical volume, op count and granularity, single-bank
               sequential cols (isolates addressing pattern from granularity).
    Volume matched to the measured f=0 C-consume point: pus*used read cols
    per channel-round."""
    import io
    used = max(1, ((1 << n) * RD_COLS // SEEDS_PER_OP) // 2) * pus
    circles = max(1, used // 4) * scale
    TILE_CIRC = 64
    src = XP_SRC_ROW + parity * 128
    dst = XP_DST_ROW + parity * 128
    lb = n % 2
    buf = io.StringIO()
    for c in range(circles):
        t, cc = divmod(c, TILE_CIRC)
        col = cc % COLS_PER_ROW
        if pattern:
            banks = [(4 * (cc % 16) + j) & 15 for j in range(4)]
            cols = [col] * 4
            wb = cc & 15
        else:
            banks = [lb] * 4
            cols = [(4 * cc + j) % COLS_PER_ROW for j in range(4)]
            wb = lb ^ 1
        for j in range(4):
            buf.write("AiM GGM_REDUCE 1 %d %d %d %d %d %d %d %d %d 1\n"
                      % (ch, banks[j], src + t, dst + t, cl, cols[j],
                         cc % COLS_PER_ROW, wb, 1 if j == 3 else 0))
    stats["xp"] = stats.get("xp", 0) + 4 * circles
    return buf.getvalue().splitlines(keepends=True)


def sm_stream_lines(n, ch, parity, stats, bank_base=0, scale=1):
    """One instance's SM (GPU NTT) HBM traffic, as cl=0 REDUCE column streams
    (same modeling device as the MAU: no PU charge, no gating, exact column
    stepping -- plain R MEM lines have no column field and would coalesce).
      read g   : used_cols from the g region the MAU wrote (bank gb, parity row)
      read a-hat + write z : used_cols read from AA region, used_cols written to
                             Z region (one op: opsize reads + wrsize writes)
    Physically these columns leave the stack over TSV/PHY to the SM; the sim
    only needs their bank/channel-side occupancy, which is exactly what these
    ops cost. cl=0 => never touches the PU."""
    import io
    used = max(1, ((1 << n) * RD_COLS // SEEDS_PER_OP) // 2)
    gb = bank_base + ((n % 2) ^ 1)         # bank the MAU wrote g into
    g_base = (1 << 10) + parity * 256      # matches LEAF_ROW_BASE parity buffers
    buf = io.StringIO()
    n_ops = 0
    # scale = number of NTT passes to inject (1 = legacy single pass over g).
    for rep in range(max(1, scale)):
        done = 0
        while done < used:
            w = min(COLS_PER_ROW, used - done)
            # g read (write artifact wrsize=1 pointed at Z region, off the hot rows)
            emit_reduce(buf, ch, gb, g_base + done // COLS_PER_ROW, done % COLS_PER_ROW,
                        w, g_row=Z_ROW_BASE + parity * 256, g_col=0, wrsize=1, cl=0)
            # a-hat read + z write
            emit_reduce(buf, ch, gb, AA_ROW_BASE + done // COLS_PER_ROW, done % COLS_PER_ROW,
                        w, g_row=Z_ROW_BASE + parity * 256 + done // COLS_PER_ROW,
                        g_col=0, wrsize=w, cl=0)
            done += w
            n_ops += 2
    stats["sm"] = stats.get("sm", 0) + n_ops
    return buf.getvalue().splitlines(keepends=True)


def interleave(a, b):
    """merge two line lists alternately (both are already channel-round-robin)."""
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


BLOCK_ROW_BASE = 512   # block-streaming ping-pong region (double-buffered by block parity)


def gen_instances_blocked(f, n, instances, channels, cl, modmul_cl, block_lg):
    """Block-streaming schedule for the capacity-wall regime (leaf buffer does
    not fit the bank): expand levels 0..s-1 once (roots persist in the phase-A
    region, rows 0..), then expand each of the 2^s subtrees to its 2^b leaves
    in a small double-buffered block region; the MAU drains block j-1's leaves
    inside block j's deepest window (finer-grained overlap than the flat
    schedule). Leaf footprint = 2 x 2^b x 16B instead of 2 x 2^n x 16B.
    Overheads vs flat: underfilled ops in each block's shallow levels
    (frontier < 8) and per-block-level SYNCs -- this generator exists to
    MEASURE them. reduce mode is always mau here."""
    import io
    stats = {}
    total = 0
    s = n - block_lg
    assert s >= 1, "block must be smaller than the tree"
    rounds = (instances + channels - 1) // channels
    pending = []
    for r in range(rounds):
        active = [ch for ch in range(channels) if r * channels + ch < instances]
        # phase A: levels 0..s-1, flat pattern in rows 0.. (output = 2^s roots)
        for lvl in range(s):
            per = level_ops(1 << lvl)
            for k in range(per):
                off_in, off_out = k * RD_COLS, k * WR_COLS
                for ch in active:
                    emit_extend(f, ch, lvl % 2,
                                off_in // COLS_PER_ROW, off_in % COLS_PER_ROW,
                                (lvl + 1) % 2, off_out // COLS_PER_ROW,
                                off_out % COLS_PER_ROW, cl)
            stats[lvl] = stats.get(lvl, 0) + per * len(active)
            total += per * len(active)
            f.write("AiM SYNC\n")
        # blocks: subtree j from root j (phase-A region) down b levels
        for j in range(1 << s):
            base = BLOCK_ROW_BASE + (j % 2) * 128
            for d in range(block_lg):
                if d == block_lg - 1 and pending:     # MAU drain of block j-1
                    for line in pending:
                        f.write(line)
                    pending = []
                per = level_ops(1 << d)
                rbank, wbank = (s + d) % 2, (s + d + 1) % 2
                for k in range(per):
                    if d == 0:   # read root j from the phase-A output region
                        off_in = j * RD_COLS
                        row_in, col_in = off_in // COLS_PER_ROW, off_in % COLS_PER_ROW
                    else:
                        off_in = k * RD_COLS
                        row_in = base + off_in // COLS_PER_ROW
                        col_in = off_in % COLS_PER_ROW
                    off_out = k * WR_COLS
                    for ch in active:
                        emit_extend(f, ch, rbank, row_in, col_in, wbank,
                                    base + off_out // COLS_PER_ROW,
                                    off_out % COLS_PER_ROW, cl)
                stats[s + d] = stats.get(s + d, 0) + per * len(active)
                total += per * len(active)
                f.write("AiM SYNC\n")
            # MAU ops for block j's leaves (in the block region, bank (s+b)%2)
            per_ch = []
            for ch in active:
                buf = io.StringIO()
                total += reduce_leaves(buf, ch, (1 << block_lg) * RD_COLS // SEEDS_PER_OP,
                                       (s + block_lg) % 2, modmul_cl, stats,
                                       mau=True, row_base=base)
                per_ch.append(buf.getvalue().splitlines(keepends=True))
            pending = [ln for group in zip(*per_ch) for ln in group]
    if pending:
        for line in pending:
            f.write(line)
        f.write("AiM SYNC\n")
    return total, stats


def gen_instances(f, n, instances, channels, cl, modmul_cl, reduce_mode,
                  sm_mode="off", pus_per_ch=1, broadcast=False, mau_ntt=0.0,
                  convert_cl=465, sm_parity="opp", xpose_mode="off",
                  xpose_timing="tap", sm_timing="tap", sm_passes=1, xpose_passes=1,
                  net_mode="off", net_alpha=0,
                  net_beaver=True, net_gang=1, net_beta=0,
                  pass2_placement="block"):
    """I independent depth-n trees, instance m -> channel m%C, sequential per channel.

    pus_per_ch=P > 1 (multi-PU channels, e.g. GDDR6_L40S 16 banks/ch -> P=8 at
    2 banks/PU): a round runs C*P instances at once, one per PU. Instance
    m -> unit (ch = m%C, pu = (m//C)%P), round = m//(C*P). Each PU owns bank
    pair (2*pu, 2*pu+1); all row/offset math is per-bank so the validated
    single-PU layout carries over unchanged within each pair.

    broadcast=True (all-bank mode, AiM ACT16/ABRD/ABWR): the P instances of a
    channel are LOCKSTEP BY CONSTRUCTION (identical tree shape, level, op
    index, row layout -- only the bank pair differs), so each (level, op) is
    ONE all-bank command (nbanks=2P) instead of P per-PU command streams:
    command/bus traffic /P, one FPU charge on the channel key (all PUs compute
    in parallel). Approximation vs a masked half-bank broadcast: the all-bank
    read/write hits both ping-pong sides; command count and tCCD pacing are
    identical, only the extra row-opens differ (second-order). MAU streams
    (reduce=mau) stay per-bank nbanks=1: the die-level MAU is a single
    consumer walking each bank's leaf columns -- not broadcastable.

    reduce=mau: round r's leaf consumption (cl=0 MAU column streams) is emitted
    INSIDE round r+1's DEEPEST level block (largest SYNC window: level n-1 has
    2^(n-1)/8 gated EXTENDs per channel, ~17k CK at n=10 -- ample slack for the
    ~2k CK CCDL-paced MAU stream) -- modeling the channel MAU draining instance
    r's leaves while the bank PUs expand instance r+1. Leaves live in a
    double-buffered region at distinct high rows (row_base alternates by round
    parity) so MAU reads never alias the ping-pong rows (avoids the simulator's
    same-address read-coalescing artifact). The last round's MAU block is exposed
    at the tail (amortized 1/rounds)."""
    import io
    stats = {}
    total = 0
    P = max(1, pus_per_ch)
    per_round = channels * P
    rounds = (instances + per_round - 1) // per_round  # instances per PU
    LEAF_ROW_BASE = 1 << 10                           # leaf region: high rows, 2 buffers

    def units(r):
        """Active (ch, bank_base) units of round r, ordered pu-major so that
        consecutive entries cycle channels (round-robin issue discipline)."""
        out = []
        for pu in range(P):
            for ch in range(channels):
                if r * per_round + pu * channels + ch < instances:
                    out.append((ch, 2 * pu))
        return out

    if sm_mode == "only" or xpose_mode == "only":
        # ISOLATION references (no EXTEND/CONVERT at all): drain time of the
        # SM-consume stream alone, the DRU XPOSE stream alone, or both together.
        # Subtracting these from the combined runs separates each engine's own
        # latency from the bus contention between them.
        want_sm = (sm_mode in ("only", "on"))
        want_xp = (xpose_mode in ("only", "pattern", "linear"))
        xp_pat = (xpose_mode != "linear")
        lines_fn = c_consume_lines if reduce_mode == "chacha" else sm_stream_lines
        for r in range(rounds):
            per_u = []
            for ch, bb in units(r):
                acc = []
                if want_xp:
                    acc.extend(xpose_lines(n, ch, r % 2, stats, 1,
                                           pattern=xp_pat,
                                           cl=(-2 if xpose_timing == "full" else 0),
                                           scale=xpose_passes))
                if want_sm:
                    acc.extend(lines_fn(n, ch, r % 2, stats, bank_base=bb,
                                        scale=sm_passes,
                                        **({"cl": -2} if sm_timing == "full"
                                           and lines_fn is c_consume_lines else {})))
                per_u.append(acc)
            for group in zip(*per_u):
                for ln in group:
                    f.write(ln)
            f.write("AiM SYNC\n")
        return stats.get("sm", 0) + stats.get("smc", 0) + stats.get("xp", 0), stats
    pass2_pending = ""                    # sched-mode deferred scatter
    for r in range(rounds):
        active = units(r)
        # ---- network timeline (ISR_NET_DELAY / ISR_NET_WAIT, frontend-only) ----
        # One rung = one party-serialized CW-exchange round trip (net_alpha CK).
        # serial: per-level rung+wait emitted after each level's SYNC below.
        # sched : round r's ladder was queued during round r-1 (level-skewed
        #         block pipeline); WAIT here = max(r-1 compute, r ladder).
        #         Round 0's ladder has no prior window -> exposed ramp.
        #         Beaver = 2 serial rungs (open d,e -> open CW) on the same lane.
        # gang (net_gang=G): G rounds level-lockstep share ONE exchange per
        # level (coalesced message, blocks independent => protocol-legal).
        # rung cost = alpha + beta*G (coalesced-message serialization term).
        # G=1 degenerates byte-exactly to the per-round schedule.
        rung = net_alpha + net_beta * net_gang
        if net_mode == "sched" and r % net_gang == 0:
            gi = r // net_gang
            if gi == 0:
                f.write("AiM NET_DELAY %d %d\n" % (n, rung))
            f.write("AiM NET_WAIT\n")
            if pass2_pending and pass2_placement == "block":
                f.write(pass2_pending)              # CW now guaranteed arrived
                pass2_pending = ""
            if (gi + 1) * net_gang < rounds:
                f.write("AiM NET_DELAY %d %d\n" % (n, rung))
            if net_beaver:
                f.write("AiM NET_DELAY 2 %d\n" % rung)
        if net_mode == "serial" and net_gang > 1 and r % net_gang == 0:
            # lockstep-serial gang: one blocking ladder (+ coalesced Beaver)
            # per gang instead of per level -- ablation arm.
            f.write("AiM NET_DELAY %d %d\n"
                    % (n + (2 if net_beaver else 0), rung))
            f.write("AiM NET_WAIT\n")
        for lvl in range(n):                          # all active units in level lockstep
            parents = 1 << lvl
            per = level_ops(parents)
            if broadcast:
                # Broadcast mode: the drain is TRICKLED between the deepest
                # window's EXTEND ops instead of dumped as one block. Same
                # window, same round-robin order, identical request counts --
                # but the per-channel queue depth stays O(chunk) instead of
                # O(whole round's MAU stream), which (a) matches the hardware
                # (the MAU consumes continuously, it does not burst-buffer a
                # whole round) and (b) avoids the simulator's quadratic
                # wall-clock blowup on deep FRFCFS/pending scans (measured:
                # 1-round probe 73 s, 2-round >9 min with the block dump).
                # one all-bank command per (level, op) per channel feeds all P PUs.
                # Read stride: with --reread each op issues 2*RD_COLS reads; the
                # physical re-read hits the same seed cols, but all-bank requests
                # (bank=-1 = wildcard in compare_addr_vec) alias so aggressively
                # that overlapping cols trigger the simulator's same-address
                # read-coalescing artifact (validated: 4112 ABRD collapsed to
                # 253). Emit DISTINCT cols (stride = reads/op) -- identical
                # command count, byte count and pacing, no aliasing.
                rd_stride = (2 * RD_COLS) if REREAD else RD_COLS
                chans = sorted({ch for ch, _ in active})
                # chacha (new Beaver algo): the LAST level fuses expand + per-child
                # ChaCha8 out-hash H' + sparse mod-P reduce + partial-sum, and writes
                # only 8B/leaf (half the columns). No separate re-read reduce tail.
                fuse = (reduce_mode == "chacha" and lvl == n - 1)
                # opt-4/5 contention: SM's C-consume stream rides the same SYNC
                # window as the fused CONVERT level, TRICKLED between the
                # level's ops (the MAU lesson: a block dump serializes the
                # in-order frontend queue behind the stream -- measured 2.6x
                # blowup; trickling restores concurrency). Parity 'opp' = SM
                # reads round r-1's buffer (doorbell-good); 'same' = the buffer
                # being written (the opt-5 bad case). Per-PU streams.
                p2_lines = {}
                if fuse and pass2_pending and pass2_placement == "trickle":
                    # the pending block's trailing SYNC is dropped: the fuse
                    # level's own SYNC provides the barrier.
                    for ln in pass2_pending.splitlines(keepends=True):
                        if ln.startswith("AiM GGM_REDUCE"):
                            p2_lines.setdefault(int(ln.split()[3]), []).append(ln)
                    pass2_pending = ""
                c_lines = {}
                if fuse and (sm_mode == "on" or xpose_mode != "off"):
                    cpar = (r % 2) ^ (1 if sm_parity == "opp" else 0)
                    for ch in chans:
                        acc = []
                        if xpose_mode != "off":
                            # DRU XPOSE stream rides the trickle slot.
                            # timing 'tap' (cl=0): DRU-FIFO bus-slot model;
                            # 'full' (cl=-2): normal rd/wr buffers, full DRAM
                            # timing incl. ACT/row state, still no FPU charge.
                            acc.extend(xpose_lines(
                                n, ch, cpar, stats, P,
                                pattern=(xpose_mode == "pattern"),
                                cl=(-2 if xpose_timing == "full" else 0),
                                scale=xpose_passes))
                        if sm_mode == "on" or xpose_mode == "off":
                            # SM C-consume stream; combined with xpose above =
                            # the three-way (expand + DRU + SM) contention run.
                            for pu in range(P):
                                acc.extend(c_consume_lines(n, ch, cpar, stats,
                                                           bank_base=2 * pu,
                                                           scale=sm_passes,
                                                           cl=(-2 if sm_timing == "full" else 0)))
                        c_lines[ch] = acc
                for k in range(per):
                    off_in, off_out = k * rd_stride, k * WR_COLS
                    for ch in chans:
                        if fuse:
                            # reads = full leaf cols (2*RD under reread = 8), writes
                            # = WR_COLS//2 (8B truncation), cl = convert_cl on the PU.
                            # C region is DOUBLE-BUFFERED by round parity (rows
                            # (1<<10)+ (r%2)*256) so the SM can consume round r-1
                            # while round r is produced (doorbell pipeline).
                            emit_reduce(f, ch, lvl % 2,
                                        off_in // COLS_PER_ROW, off_in % COLS_PER_ROW,
                                        rd_stride,
                                        (1 << 10) + (r % 2) * 256 +
                                        off_out // COLS_PER_ROW,
                                        off_out % COLS_PER_ROW, WR_COLS // 2,
                                        convert_cl, nbanks=2 * P)
                        else:
                            emit_extend(f, ch, lvl % 2,
                                        off_in // COLS_PER_ROW, off_in % COLS_PER_ROW,
                                        (lvl + 1) % 2, off_out // COLS_PER_ROW,
                                        off_out % COLS_PER_ROW, cl,
                                        nbanks=2 * P)
                    if p2_lines:
                        for ch in chans:
                            p2_ch = p2_lines.get(ch, [])
                            lo = k * len(p2_ch) // per
                            hi = (k + 1) * len(p2_ch) // per
                            f.writelines(p2_ch[lo:hi])
                    if c_lines:
                        for ch in chans:
                            cl_ch = c_lines[ch]
                            lo = k * len(cl_ch) // per
                            hi = (k + 1) * len(cl_ch) // per
                            f.writelines(cl_ch[lo:hi])
                stats[lvl] = stats.get(lvl, 0) + per * len(chans)
                total += per * len(chans)

            else:
                for k in range(per):
                    off_in, off_out = k * RD_COLS, k * WR_COLS
                    for ch, bb in active:
                        emit_extend(f, ch, bb + lvl % 2,
                                    off_in // COLS_PER_ROW, off_in % COLS_PER_ROW,
                                    bb + (lvl + 1) % 2, off_out // COLS_PER_ROW,
                                    off_out % COLS_PER_ROW, cl)
                stats[lvl] = stats.get(lvl, 0) + per * len(active)
                total += per * len(active)
            f.write("AiM SYNC\n")
            if net_mode == "serial" and net_gang == 1:
                # unscheduled: this level's CW exchange blocks the next level
                f.write("AiM NET_DELAY 1 %d\n" % net_alpha)
                f.write("AiM NET_WAIT\n")
                if net_beaver and lvl == n - 1:
                    f.write("AiM NET_DELAY 2 %d\n" % net_alpha)
                    f.write("AiM NET_WAIT\n")
        if reduce_mode == "fused":
            if broadcast:
                # lockstep fused lc: every PU reduces its own leaves, same
                # column schedule -> all-bank REDUCE, cl charged once per op
                # on the channel key (all PUs compute in parallel).
                chans = sorted({ch for ch, _ in active})
                for ch in chans:
                    total += reduce_leaves(f, ch, (1 << n) * RD_COLS // SEEDS_PER_OP,
                                           n % 2, modmul_cl, stats, nbanks=2 * P)
            else:
                for ch, bb in active:
                    total += reduce_leaves(f, ch, (1 << n) * RD_COLS // SEEDS_PER_OP,
                                           bb + n % 2, modmul_cl, stats)
            f.write("AiM SYNC\n")
        elif reduce_mode == "mau":
            # INLINE emission (same trace position as fused): round r's MAU
            # drain is emitted right here, marked cl=0. The controller routes
            # cl=0 GGM_REDUCE to a dedicated per-channel DRU FIFO (is_dru,
            # AiM_dram_controller) that drains O(1) head-of-line CONCURRENTLY
            # with the next round's EXTEND stream (the aim_buffer). That is the
            # physical overlap -- the channel-level MAU consumes leaves while
            # the bank PUs expand the next round -- expressed without a giant
            # cross-round pending block (which serialized in the single
            # frontend request_queue and blew up the wall clock).
            base = LEAF_ROW_BASE + (r % 2) * 256      # double-buffered leaf region
            per_u = []
            for ch, bb in active:
                buf = io.StringIO()
                total += reduce_leaves(buf, ch, (1 << n) * RD_COLS // SEEDS_PER_OP,
                                       bb + n % 2, modmul_cl, stats, mau=True,
                                       row_base=base)
                if mau_ntt > 0:      # case-A: stage-1 butterfly column stream
                    buf.write("".join(mau_ntt_lines(n, ch, bb, r % 2, mau_ntt, stats)))
                if sm_mode == "on":  # case-B contention: SM reads this round's g
                    buf.write("".join(sm_stream_lines(n, ch, r % 2, stats, bank_base=bb,
                                                     scale=sm_passes)))
                per_u.append(buf.getvalue().splitlines(keepends=True))
            # round-robin across channels so consecutive lines hit different
            # channels (independent FIFOs, no head-of-line block).
            for group in zip(*per_u):
                for ln in group:
                    f.write(ln)
            f.write("AiM SYNC\n")
        elif reduce_mode == "chacha":
            # ---- PASS 2 (scatter): the OTHER half of the output layer -------
            # The fused CONVERT above is pass 1 only: expand + H' + mod-reduce
            # + partial SUM, materialising C (8B/leaf) in the double-buffered
            # C region. The protocol then OPENS the sums (one network round,
            # the Beaver rungs emitted above) to reconstruct CW -- so a second
            # full pass over C is unavoidable: y = C + tau?CW, negacyclic
            # scatter-accumulate into g (pcg_ole_impl.h:171-232, mirrored by
            # dpf_out_scatter_g on the GPU). Omitting it (as this generator
            # did before 2026-07-30) undercounts the DPF lane.
            #   PIM advantage, kept honest: C never crossed the channel bus,
            # so pass 2 REREADS C from the bank (no ChaCha8 recompute). The
            # GPU recomputes H' instead, because persisting C costs it
            # 16B/leaf of extra bus traffic. Same algorithm, different optimum.
            #   Cost per op: <=64 C cols in, groups=cols/2 (8 leaves = 2 cols),
            # predicated mod-add ADD64M_CL per group on the SPU, g writes =
            # cols/2 (the D/2-slot fold halves the volume). Reads parity r%2
            # (the region the CONVERT just wrote, post-SYNC = safe); serial-net
            # mode already emitted the Beaver rungs before this point, so the
            # CW dependency is in the timeline.
            c_cols = (1 << n) // 4                  # 2^n leaves x 8B / 32B col
            c_base = (1 << 10) + (r % 2) * 256      # CONVERT's C region parity
            cbank = n % 2                           # CONVERT wrote bank^1 of (n-1)%2
            chans = sorted({ch for ch, _ in active})
            buf2 = io.StringIO()
            for ch in chans:
                done = 0
                while done < c_cols:
                    w = min(COLS_PER_ROW, c_cols - done)
                    emit_reduce(buf2, ch, cbank,
                                c_base + done // COLS_PER_ROW,
                                done % COLS_PER_ROW, w,
                                g_row=(1 << 10) + 512 + done // (2 * COLS_PER_ROW),
                                g_col=(done // 2) % COLS_PER_ROW,
                                wrsize=max(1, w // 2),
                                cl=max(1, w // 2) * ADD64M_CL,
                                nbanks=2 * P)
                    done += w
                    stats["pass2"] = stats.get("pass2", 0) + 1
                    total += 1
            buf2.write("AiM SYNC\n")
            if net_mode == "sched":
                # CO-SCHEDULED pass 2: block k's CW arrives on the pipelined
                # NIC timeline; its scatter is DEFERRED into round k+1's
                # window (after that round's NET_WAIT), so the SPU fills the
                # 2-RTT gap with round k+1's expansion. Double-buffered C
                # parity makes the deferral safe. Last round drains at EOC.
                pass2_pending = buf2.getvalue()
            else:
                # serial (or no net): the Beaver rungs just blocked above, so
                # CW is in hand -- emit in place.
                f.write(buf2.getvalue())
    if net_mode == "sched":
        f.write("AiM NET_WAIT\n")     # drain the last round's Beaver rungs
        if pass2_pending:
            f.write(pass2_pending)     # last round's scatter, exposed (1/R)
            pass2_pending = ""
    return total, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, required=True, help="tree depth (2^n leaves)")
    ap.add_argument("-C", "--channels", type=int, default=64)
    ap.add_argument("--mode", choices=["subtree", "instances"], default="subtree")
    ap.add_argument("-I", "--instances", type=int, default=64,
                    help="instances mode: number of independent trees (PCG: c^2*t^2)")
    ap.add_argument("-P", "--pus-per-ch", type=int, default=1,
                    help="PUs per channel (instances mode). PU p owns bank pair "
                         "(2p, 2p+1); a round runs C*P instances. GDDR6_L40S: "
                         "16 banks/ch -> P=8 (192 PUs on 24 channels)")
    ap.add_argument("--broadcast", action="store_true",
                    help="all-bank mode (AiM ACT16/ABRD/ABWR): one command per "
                         "(level, op) per channel feeds all P PUs (nbanks=2P); "
                         "requires the lockstep instance discipline (holds by "
                         "construction). EXTEND + fused-reduce broadcast; MAU "
                         "streams stay per-bank")
    ap.add_argument("--mau-ntt", type=float, default=0.0, metavar="FACTOR",
                    help="case-A: inject MAU NTT stage-1 butterfly column "
                         "traffic (cl=0 streams, volume = FACTOR x the leaf "
                         "stream). Requires --reduce mau. 0 = off")
    ap.add_argument("--seed-bits", type=int, choices=[128, 256], default=128,
                    help="seed width. 128 = dpf.cpp-aligned (emp::block, 4 words/seed, "
                         "4 rd + 8 wr cols/op); 256 = ChaCha-key-width variant (8 rd + 16 wr)")
    ap.add_argument("--cl", type=int, default=None,
                    help="EXTEND compute latency; default 279 @128b (256 ChaCha + 16 ff + "
                         "4 PACKLR32 + 2 VCXOR + 1 VXOR free-child), 282 @256b")
    ap.add_argument("--modmul-cl", type=int, default=32, help="Barrett 62b modmul cycles per 8-leaf GROUP (lane-parallel 32b sequence)")
    ap.add_argument("--convert-cl", type=int, default=465,
                    help="chacha mode: fused CONVERT compute latency for the LAST "
                         "level (expand + per-child ChaCha8 out-hash H' + sparse "
                         "mod-P reduce REDP + partial-sum ADDP). Default 465 (dual-"
                         "ALU ~3x the 155 EXTEND: 1x expand + 2x child-hash). Single-"
                         "ALU ~846; sensitivity 310/620.")
    ap.add_argument("--reduce", choices=["none", "fused", "mau", "chacha"], default="none",
                    help="none=expansion only; fused=leaf_convert on the bank PU "
                         "(modmul-cl charged); mau=leaf_convert on the channel-level "
                         "MAU (cl=0, memory traffic only; instances mode overlaps it "
                         "with the next round's expansion); chacha=NEW Beaver algorithm "
                         "(no modmul): the LAST level is a fused CONVERT op "
                         "(expand+H'+REDP+ADDP, convert-cl, 8B write = half cols), no "
                         "re-read tail. Pure ARX + multiplier-free mod-ALU.")
    ap.add_argument("--reread", action="store_true",
                    help="register-exact: re-read seed for child R -> 8 read + 8 write/batch")
    ap.add_argument("--block", type=int, default=0, metavar="B_LOG",
                    help="block-streaming schedule: expand in subtree blocks of "
                         "2^B_LOG leaves (capacity-wall regime; implies "
                         "--reduce mau, instances mode). 0 = flat schedule")
    ap.add_argument("--sm-parity", choices=["opp", "same"], default="opp",
                    help="chacha C-consume stream parity: opp = SM reads round "
                         "r-1's C buffer while PIM writes round r (doorbell "
                         "good case); same = colliding with the buffer being "
                         "produced (opt-5 bad case)")
    ap.add_argument("--sm-stream", choices=["off", "on", "only"], default="off",
                    help="inject the SM (GPU NTT) HBM traffic as cl=0 column "
                         "streams: read g (MAU output), read a-hat, write z. "
                         "'on' = alongside expansion+MAU (requires --reduce mau, "
                         "instances mode); 'only' = the SM blocks alone (drain "
                         "reference)")
    ap.add_argument("--xpose-stream", choices=["off", "pattern", "linear", "only"],
                    default="off",
                    help="inject the DRU XPOSE stream in the fused level's "
                         "trickle slot (requires --reduce chacha --broadcast): "
                         "'pattern' = true 4-bank gather + rolling write bank "
                         "+ per-tile row cadence, 1 col/request; 'linear' = "
                         "volume/op-count/granularity-matched single-bank "
                         "control")
    ap.add_argument("--net", choices=["off", "serial", "sched"], default="off",
                    help="native network timeline (ISR_NET_DELAY/WAIT): 'serial'"
                         " = per-level exchange blocks the next level (no"
                         " co-schedule); 'sched' = level-skewed block pipeline"
                         " (round r+1's ladder hidden under round r, block-0"
                         " ladder exposed as ramp). Requires chacha+broadcast.")
    ap.add_argument("--net-alpha-ck", type=int, default=0,
                    help="one round-trip in CK (alpha / tCK of the target org)")
    ap.add_argument("--net-beaver", choices=["on", "off"], default="on",
                    help="add the output layer's 2 serial Beaver opens "
                         "(open d,e -> open CW) to the network timeline")
    ap.add_argument("--net-gang", type=int, default=1,
                    help="gang size G: G rounds level-lockstep share one "
                         "coalesced exchange per level (amortizes alpha by G; "
                         "message grows xG -> see --net-beta-ck-per-block). "
                         "G=1 = per-round schedule (byte-exact degeneration)")
    ap.add_argument("--pass2-placement", choices=["block", "trickle"],
                    default="block",
                    help="sched-mode placement of the deferred scatter pass: "
                         "'block' = one contiguous block at the next round's "
                         "head (conservative); 'trickle' = interleaved between "
                         "the next round's CONVERT ops so pass 2's memory "
                         "phase rides the compute-bound level's column-bus "
                         "slack (the validated MAU-trickle mechanism)")
    ap.add_argument("--net-beta-ck-per-block", type=int, default=0,
                    help="CK added to each rung per coalesced block "
                         "(the g*M/beta serialization term); 0 = latency-only")
    ap.add_argument("--sm-timing", choices=["tap", "full"], default="tap",
                    help="which arbitration path the SM stream uses: 'tap' "
                         "(cl=0, lowest priority, yields to normal traffic) or "
                         "'full' (cl=-2, normal rd/wr buffers WITH priority -- "
                         "the correct model for SM/external traffic; the DRU is "
                         "the engine that must yield)")
    ap.add_argument("--sm-passes", type=int, default=1,
                    help="scale the injected SM-NTT stream volume by this factor "
                         "(1 = legacy ~1 pass over g; the true 4-step rate is "
                         "27 passes/mul on the SM after the transposes leave)")
    ap.add_argument("--xpose-passes", type=int, default=1,
                    help="scale the injected DRU XPOSE stream volume (true rate "
                         "= 12 passes/mul for the transposes the DRU absorbs)")
    ap.add_argument("--xpose-timing", choices=["tap", "full"], default="tap",
                    help="xpose stream timing model: 'tap' = DRU FIFO bus-slot "
                         "(cl=0, addressing-blind); 'full' = cl=-2 sentinel -> "
                         "normal rd/wr buffers with full DRAM timing (ACT/row "
                         "state), no FPU charge -- bank-conflict-sensitive")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    global RD_COLS, WR_COLS, REREAD
    RD_COLS = a.seed_bits // 32
    WR_COLS = 2 * RD_COLS
    REREAD = a.reread
    if a.cl is None:
        a.cl = 279 if a.seed_bits == 128 else 282

    with open(a.out, "w") as f:
        f.write("# GGM tree dataflow: n=%d mode=%s C=%d seed=%db cl=%d reduce=%s\n"
                % (a.n, a.mode, a.channels, a.seed_bits, a.cl, a.reduce))
        if a.sm_stream == "on" and (a.mode != "instances" or
                                    a.reduce not in ("mau", "chacha")):
            sys.exit("--sm-stream on requires --mode instances --reduce mau|chacha")
        if a.net != "off":
            if a.mode != "instances" or a.reduce != "chacha" or not a.broadcast:
                sys.exit("--net requires --mode instances --reduce chacha "
                         "--broadcast")
            if a.net_alpha_ck <= 0:
                sys.exit("--net requires --net-alpha-ck > 0")
        if a.xpose_stream != "off":
            if a.mode != "instances" or a.reduce != "chacha" or not a.broadcast:
                sys.exit("--xpose-stream requires --mode instances "
                         "--reduce chacha --broadcast")
            # --sm-stream on may be combined: both streams share the trickle
            # slot (concatenated per channel) = the three-way contention run.
        if a.pus_per_ch > 1 and (a.block > 0 or a.mode != "instances"):
            sys.exit("--pus-per-ch requires --mode instances (flat schedule)")
        if a.block > 0:
            if a.mode != "instances":
                sys.exit("--block requires --mode instances")
            total, stats = gen_instances_blocked(f, a.n, a.instances, a.channels,
                                                 a.cl, a.modmul_cl, a.block)
        elif a.mode == "subtree":
            total, stats = gen_subtree(f, a.n, a.channels, a.cl, a.modmul_cl, a.reduce)
        else:
            if a.mau_ntt > 0 and a.reduce != "mau":
                sys.exit("--mau-ntt requires --reduce mau")
            if a.reduce == "chacha" and not a.broadcast:
                sys.exit("--reduce chacha requires --broadcast (last-level fuse "
                         "is only wired in the all-bank path)")
            total, stats = gen_instances(f, a.n, a.instances, a.channels, a.cl,
                                         a.modmul_cl, a.reduce, sm_mode=a.sm_stream,
                                         pus_per_ch=a.pus_per_ch,
                                         broadcast=a.broadcast, mau_ntt=a.mau_ntt,
                                         convert_cl=a.convert_cl,
                                         sm_parity=a.sm_parity,
                                         xpose_mode=a.xpose_stream,
                                         xpose_timing=a.xpose_timing,
                                         sm_timing=a.sm_timing,
                                         sm_passes=a.sm_passes,
                                         xpose_passes=a.xpose_passes,
                                         net_mode=a.net,
                                         net_alpha=a.net_alpha_ck,
                                         net_beaver=(a.net_beaver == "on"),
                                         net_gang=a.net_gang,
                                         net_beta=a.net_beta_ck_per_block,
                                         pass2_placement=a.pass2_placement)
        f.write("AiM SYNC\nAiM EOC\n")

    lv = {k: v for k, v in stats.items() if isinstance(k, int)}
    print("[tree] n=%d mode=%s C=%d P=%d -> %d EXTEND+REDUCE ops (%d leaves)"
          % (a.n, a.mode, a.channels, a.pus_per_ch, total, 1 << a.n), file=sys.stderr)
    print("[tree] per-level ops: %s%s"
          % (" ".join("L%d:%d" % (k, lv[k]) for k in sorted(lv)),
             "  reduce:%d" % stats["reduce"] if "reduce" in stats else ""), file=sys.stderr)
    if a.reduce == "mau":
        # GDDR die-level MAU consumption check (analytic; the trace itself only
        # carries the DRAM-side column streams). One MAU per DRAM die serves
        # 2 channels = 2*P PUs. Leaf production per die (leaves/CK, command
        # clock) vs MAU capacity: the fused sum+scale pipeline retires one
        # 8-leaf group op per MAU cycle, 2 passes -> leaves/4 ops needed.
        for f_ck_ghz, tag in ((2.25, "GDDR6 CK"),):
            for f_mau_ghz in (1.0, 0.5):
                leaves_per_s = 2 * a.pus_per_ch * 2 * SEEDS_PER_OP / a.cl * f_ck_ghz * 1e9
                need_gops = leaves_per_s / 4 / 1e9
                util = need_gops / f_mau_ghz
                print("[mau-die] P=%d cl=%d %s=%.2fGHz MAU@%.1fGHz: "
                      "%.2f Gleaf/s/die -> %.2f Gop/s needed = %.0f%% of one MAU%s"
                      % (a.pus_per_ch, a.cl, tag, f_ck_ghz, f_mau_ghz,
                         leaves_per_s / 1e9, need_gops, util * 100,
                         "  ** BINDS: need 2 MAUs/die (one per channel) **"
                         if util > 1.0 else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
