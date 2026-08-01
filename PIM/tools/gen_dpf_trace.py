#!/usr/bin/env python3
"""
DPF/GGM trace generator for the AiM sim (sim/, Ramulator2-based).

One DPF batch (8 seeds) = one GGM_EXTEND:
    load 8 columns (seeds) -> ChaCha8+DPF compute (compute_latency FPU cycles)
    -> store 16 columns (2 children x 8 words).
compute_latency=274 = 256 ChaCha core (32 QR x 8 ARX) + 16 feed-forward + 2 VCXOR (CW apply).

Batches are issued round-robin over channels (instruction-level parallel dispatch:
the AiM DMA issues up to #channels ISRs per cycle). Within a channel each batch
uses a DISTINCT read row by default -- reusing addresses lets the controller
coalesce reads and silently deflates the memory traffic (measured artifact).

--packed: 8 batches share one read row (col_in=(k%8)*8) and 4 batches share one
write row (col_out=(k%4)*16), amortizing the per-batch ACT (floor 240 -> 141 CK/batch).

Trace line (10 fields):
    AiM GGM_EXTEND <opsize> <ch_idx> <bank> <row_in> <row_out> <compute_latency>
                   <col_in> <col_out> <write_bank> <nbanks>
ch_idx is a DIRECT channel index (not a bitmask) in this sim.
"""
import argparse

def gen(path, batches, channels, cl, packed=False, opsize=8):
    with open(path, "w") as f:
        f.write("# B=%d batches round-robin over %d channels, cl=%d%s\n"
                % (batches, channels, cl, ", row-packed" if packed else ""))
        f.write("# fields: opsize ch_idx bank row_in row_out compute_latency"
                " col_in col_out write_bank nbanks\n")
        for i in range(batches):
            ch = i % channels
            k = i // channels                       # k-th batch on this channel
            if packed:
                row_in, col_in = k // 8, (k % 8) * opsize
                row_out, col_out = 60000 + k // 4, (k % 4) * 2 * opsize
            else:
                row_in, col_in = k, 0
                row_out, col_out = 60000 + k, 0
            f.write("AiM GGM_EXTEND %d %d 0 %d %d %d %d %d 1 1\n"
                    % (opsize, ch, row_in, row_out, cl, col_in, col_out))
        f.write("AiM SYNC\nAiM EOC\n")
    print("wrote %s (%d batches, %d ch, cl=%d%s)"
          % (path, batches, channels, cl, ", packed" if packed else ""))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-B", "--batches", type=int, default=1024)
    ap.add_argument("-C", "--channels", type=int, default=64)
    ap.add_argument("--cl", type=int, default=274,
                    help="compute_latency (274 = ChaCha8 + ff + 2 VCXOR)")
    ap.add_argument("--packed", action="store_true",
                    help="pack 8 batches/read-row, 4/write-row (amortize ACT)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    gen(a.out, a.batches, a.channels, a.cl, a.packed)
