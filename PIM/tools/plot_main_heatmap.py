#!/usr/bin/env python3
"""Main-result heatmap: end-to-end speedup over the GPU baseline.

Two panels, one per device, each 6 security configurations x 4 values of the
per-exchange round-trip latency alpha.  Numbers come from l40s_pim_vs_gpu.py so
the figure cannot drift from the tables.

speedup = B1 / max(SPU, SM, NIC)
  B1   GPU baseline runtime: measured DPF+convert (which blocks on the
       exchanges inline) + device-only four-step NTT.  No PCIe on either side.
  SPU  leaves x pim_ns -- expansion and both output-layer passes, in bank.
  SM   the same NTT the baseline runs; on B200 minus the DRU-offloaded
       transposes, on GDDR6 unchanged (no shadow bandwidth).
  NIC  (c^2 n + 2c^2) * alpha -- the same exchange count the reference issues.
       No gang, no coalescing.

The two panels are normalised separately: L40S spans 2.0-3.4 and B200 4.2-9.8,
so one shared scale would flatten the L40S panel to a single colour.  Read
within a panel, not across.

Usage:  python3 plot_main_heatmap.py [out.pdf]
"""
import csv, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l40s_pim_vs_gpu as M

ROOT = os.path.join(HERE, "..", "..", "GPU_baseline")
ROWS = [(80, 2, 64, 17, 4096), (80, 4, 16, 19, 256), (80, 8, 4, 21, 16),
        (128, 2, 128, 16, 16384), (128, 4, 16, 19, 256), (128, 8, 8, 20, 64)]
TIERS = ["nvlink", "pcie", "loopback", "dc"]
ALPHA = [5, 16, 25, 50]
DEV = [("L40S", "L40S (GDDR6)", "data_l40s/ntt_stages.csv"),
       ("B200", "B200 (HBM3e)", "pcg_baseline_out/ntt_stages.csv")]


def move_share(path):
    """(transpose + bit-reversal) / total, per (logN, batch) -- the DRU's share."""
    out = {}
    with open(os.path.join(ROOT, path)) as f:
        for r in csv.DictReader(f):
            if r["backend"] != "square" or r["brev_mode"] != "standalone":
                continue
            if r["status"] == "FAIL":
                continue
            v = [float(r[k]) for k in ("transpose_ms", "brev_ms", "ntt_ms",
                                       "twiddle_ms", "twist_ms", "pointwise_ms")]
            if sum(v) > 0:
                out[(int(r["logN"]), int(r["batch"]))] = (v[0] + v[1]) / sum(v)
    return out


def gmean(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def collect(mach, nttcsv):
    mv = move_share(nttcsv)
    allr = M.main(mach)
    dru = (lambda r: 1 - mv.get((int(r["logN"]), r["c"] ** 2), 0.0)) \
        if mach == "B200" else (lambda r: 1.0)

    S = np.full((len(ROWS), len(TIERS)), np.nan)
    netbound = np.zeros_like(S, dtype=bool)
    trunc = np.zeros(len(ROWS), dtype=bool)
    for j, tier in enumerate(TIERS):
        sel = [r for r in allr if r["tier"] == tier]
        for i, (lam, c, t, _, _) in enumerate(ROWS):
            s = [r for r in sel if r["c"] == c and r["t"] == t]
            if (c, t) == (4, 16):                    # appears in both lambda blocks
                d = {}
                for r in s:
                    d.setdefault(r["logN"], r)
                s = list(d.values())
            if not s:
                continue
            num = [r["b1"] - r["sm_pcie"] for r in s]
            den = [max(r["pim_lane"], r["sm_ntt"] * dru(r), r["net_lane"]) for r in s]
            S[i, j] = gmean([n / d for n, d in zip(num, den)])
            netbound[i, j] = all(
                r["net_lane"] >= max(r["pim_lane"], r["sm_ntt"] * dru(r)) for r in s)
            if max(int(r["logN"]) for r in s) < 24:
                trunc[i] = True
    geo = np.array([gmean([r["b1"] - r["sm_pcie"] for r in allr if r["tier"] == ti]
                          and [(r["b1"] - r["sm_pcie"]) /
                               max(r["pim_lane"], r["sm_ntt"] * dru(r), r["net_lane"])
                               for r in allr if r["tier"] == ti])
                   for ti in TIERS])
    return S, netbound, trunc, geo


def main(outpath):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9),
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.34))
    ylabels = [rf"$\lambda${lam}  ({c},{t})" + "\n" + rf"$n${n}, $B${B}"
               for lam, c, t, n, B in ROWS]

    for ax, (mach, title, nttcsv) in zip(axes, DEV):
        S, netbound, trunc, geo = collect(mach, nttcsv)
        full = np.vstack([S, geo])                    # geomean as a final row
        im = ax.imshow(full, cmap="viridis", aspect="auto")

        lo, hi = np.nanmin(full), np.nanmax(full)
        for i in range(full.shape[0]):
            for j in range(full.shape[1]):
                v = full[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.4,
                        color="white" if v < lo + 0.55 * (hi - lo) else "black",
                        fontweight="bold" if i < len(ROWS) else "normal")
                if i < len(ROWS) and netbound[i, j]:   # NIC sets the steady state
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               ec="crimson", lw=1.6, zorder=3))

        ax.set_xticks(range(len(ALPHA)))
        ax.set_xticklabels([str(a) for a in ALPHA], fontsize=8)
        ax.set_xlabel(r"$\alpha$  ($\mu$s per exchange)", fontsize=8.5)
        ax.set_yticks(range(len(ROWS) + 1))
        ax.set_yticklabels([l + ("$^\\dagger$" if trunc[i] else "")
                            for i, l in enumerate(ylabels)] + ["geomean"], fontsize=7)
        ax.axhline(len(ROWS) - 0.5, color="w", lw=2)
        ax.set_title(title, fontsize=9.5, pad=6)
        if ax is not axes[0]:
            ax.tick_params(labelleft=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(r"speedup $\times$", fontsize=7.5)

    fig.text(0.5, -0.10,
             "Red outline: the NIC lane sets the co-scheduled steady state, so the "
             "memory-side gain is not exposed.\n"
             r"$^\dagger$ $N{=}2^{24}$ omitted on L40S (leaf pool 128--256\,GB vs "
             r"46\,GB). Panels are normalised separately.",
             ha="center", va="top", fontsize=6.8)
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    print("wrote", outpath)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(HERE, "..", "docs", "figures", "main_heatmap.pdf"))
