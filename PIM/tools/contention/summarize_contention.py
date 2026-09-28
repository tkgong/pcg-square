#!/usr/bin/env python3
"""Tables for the GPU-PIM contention study from run_contention.py output.

Per org, lane balance r (= GPU-alone / SPU-alone time) and arbitration policy:
  (a) GPU-only NTT, (b) GPU + SPU, (c) GPU + SPU + DRU
with, for each configuration:
  * per-channel bandwidth utilisation: GPU / DRU data-bus busy fraction while the
    host stream is active, SPU all-bank column-slot occupancy over the SPU run
    (mean [min-max] over channels)
  * DRAM queuing latency of GPU reads (arrive -> first command) and GPU read
    latency (arrive -> data), mean and p99, in ns
  * GPU degradation: host-stream time vs its standalone reference; for (c) also
    NTT completion (SM lane and DRU both done) vs (a)
  * SPU degradation vs SPU alone
Usage: summarize_contention.py DIR [DIR ...]  -> prints markdown, writes DIR/summary.json
"""
import json, os, re, statistics as st, sys

TCK = {"l40s": 0.444, "b200": 0.500}


def chunk_starts(streams_file):
    out = []
    for l in open(streams_file):
        if l.startswith("#") or not l.strip():
            continue
        f = l.split()
        out.append((int(f[0]), int(f[7]) if len(f) > 7 else 0))
    return out


def durations(d, streams_file, cls):
    """sum over this class's chunks of (mean-over-channels finish - start)."""
    tot = 0.0
    for k, (c, start) in enumerate(chunk_starts(streams_file)):
        if c == cls:
            tot += d[f"stream{k}_class{c}_finish_mean"] - start
    return tot


def ch_vals(d, key):
    v = [d[k] for k in d if re.fullmatch(rf"CH\d+_{key}", k)]
    return v


def rng(v, pct=True):
    if not v:
        return "-"
    f = (lambda x: f"{x * 100:.0f}%") if pct else (lambda x: f"{x:.0f}")
    return f"{f(st.mean(v))} [{f(min(v))}-{f(max(v))}]"


def wmean(d, cls, stat):
    cnt = ch_vals(d, f"{cls}_{'q' if stat.startswith('q') else 'rdlat'}_count")
    val = ch_vals(d, f"{cls}_{stat}")
    if not cnt or sum(cnt) == 0:
        return None
    return sum(c * v for c, v in zip(cnt, val)) / sum(cnt)


def cfg_row(D, d, sfile, T_spu, kind, ref):
    tck = TCK[D["org"]]
    r = {}
    active = {c: durations(d, sfile, c) for c in (1, 2)}
    gpu_bus = [b / max(active[1] / 4, 1) / 4 for b in ch_vals(d, "gpu_bus_cycles")]  # per chunk-avg
    # utilisation while the host stream runs: busy cycles / active cycles (sum over chunks)
    gpu_bus = [b / active[1] for b in ch_vals(d, "gpu_bus_cycles")] if active[1] else []
    dru_bus = [b / active[2] for b in ch_vals(d, "dru_bus_cycles")] if active[2] else []
    spu_end = d.get("pim_done_cycles", -1)
    spu_slot = [x / spu_end for x in ch_vals(d, "pim_allbank_slot_cycles")] if spu_end and spu_end > 1 else []
    r["gpu_bw"] = rng(gpu_bus)
    r["dru_bw"] = rng(dru_bus)
    r["spu_slots"] = rng(spu_slot)
    q_mean, q_p99 = wmean(d, "gpu", "q_mean"), max(ch_vals(d, "gpu_q_p99") or [0])
    l_mean, l_p99 = wmean(d, "gpu", "rdlat_mean"), max(ch_vals(d, "gpu_rdlat_p99") or [0])
    r["gpu_q"] = f"{q_mean * tck:.0f} / {q_p99 * tck:.0f}" if q_mean is not None else "-"
    r["gpu_lat"] = f"{l_mean * tck:.0f} / {l_p99 * tck:.0f}" if l_mean is not None else "-"
    hits = sum(ch_vals(d, "gpu_row_hit")); tot = hits + sum(ch_vals(d, "gpu_row_miss")) + sum(ch_vals(d, "gpu_row_conflict"))
    r["gpu_rowhit"] = f"{hits / tot * 100:.0f}%" if tot else "-"
    r["gpu_slow"] = active[1] / ref["gpu"] if ref.get("gpu") else None
    if kind == "c":
        # NTT done when both the SM lane and the DRU transposes of each chunk are done
        ends = {}
        for k, (c, start) in enumerate(chunk_starts(sfile)):
            ends.setdefault(start, []).append(d[f"stream{k}_class{c}_finish_mean"])
        r["ntt_vs_a"] = sum(max(v) - s for s, v in ends.items()) / ref["a"]
    r["spu_slow"] = spu_end / T_spu if spu_end and spu_end > 1 else None
    if spu_end and spu_end > 1 and kind in ("b", "c"):
        host_end = max(d[k] for k in d if re.fullmatch(r"stream\d+_class\d_finish_max", k))
        both = max(spu_end, host_end)
        gpu_alone = ref["gpu_alone_total"]          # host lane alone (sum of chunk durations)
        r["win_vs_max"] = both / max(T_spu, gpu_alone)   # 1.00 = the paper's max() model holds
        r["win_vs_serial"] = both / (T_spu + gpu_alone)  # < 1 = overlap still pays off
    r["_raw"] = dict(active=active, spu_end=spu_end)
    return r


def main():
    for D_dir in sys.argv[1:]:
        D = json.load(open(os.path.join(D_dir, "results.json")))
        R, T = D["results"], D["T_spu"]
        tck = TCK[D["org"]]
        print(f"\n### {D['org'].upper()} ({D['C']} channels simulated), SPU alone = {T:.0f} CK "
              f"({T * tck / 1e3:.0f} us); standalone host-stream BW per channel: "
              + ", ".join(f"{k} {v / 16 * 100:.0f}% of peak" for k, v in D["bw_B_per_ck"].items()))
        rs = sorted({re.match(r"r([0-9.]+)_", k).group(1) for k in R if re.match(r"r[0-9.]+_", k)}, key=float)
        summary = {}
        for rv in rs:
            p = f"r{rv}_"
            sf = lambda tag: os.path.join(D_dir, tag + ".streams")
            ref = {}
            ra = cfg_row(D, R[p + "a_gpufull"], sf(p + "a_gpufull"), T, "a", {})
            ref_a = ra["_raw"]["active"][1]
            ra["gpu_slow"] = 1.0
            rsm = cfg_row(D, R[p + "ref_gpusm"], sf(p + "ref_gpusm"), T, "ref", {})
            ref_sm = rsm["_raw"]["active"][1]
            print(f"\n**r = {rv}** (GPU-alone / SPU-alone time).  GPU-only full NTT = {ref_a:.0f} CK,"
                  f" SM lane alone = {ref_sm:.0f} CK")
            print("| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) |"
                  " GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone |"
                  " both done vs max() | both done vs serial |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            rows = [("(a) GPU-only", "-", ra)]
            pols = sorted({k.split("_")[-1] for k in R if k.startswith(p + "b_")})
            for pol in pols:
                smdru = R.get(p + "ref_smdru")
                smdru_t = (max(v for k, v in smdru.items() if re.fullmatch(r"stream\d+_class\d_finish_max", k))
                           if smdru else ref_sm)
                rb = cfg_row(D, R[p + "b_" + pol], sf(p + "b_" + pol), T, "b",
                             {"gpu": ref_a, "gpu_alone_total": ref_a})
                rc = cfg_row(D, R[p + "c_" + pol], sf(p + "c_" + pol), T, "c",
                             {"gpu": ref_sm, "a": ref_a, "gpu_alone_total": smdru_t})
                rows += [("(b) GPU+SPU", pol, rb), ("(c) GPU+SPU+DRU", pol, rc)]
            for name, pol, x in rows:
                gs = f"{x['gpu_slow']:.2f}x" if x.get("gpu_slow") else "-"
                nv = f"{x['ntt_vs_a']:.2f}x" if x.get("ntt_vs_a") else ("1.00x" if name.startswith("(a)") else "-")
                ss = f"{x['spu_slow']:.2f}x" if x.get("spu_slow") else "-"
                print(f"| {name} | {pol} | {x['gpu_bw']} | {x['dru_bw']} | {x['spu_slots']} | {x['gpu_q']} |"
                      f" {x['gpu_lat']} | {x['gpu_rowhit']} | {gs} | {nv} | {ss} |"
                      f" {x.get('win_vs_max', 0) and format(x['win_vs_max'], '.2f') + 'x' or '-'} |"
                      f" {x.get('win_vs_serial', 0) and format(x['win_vs_serial'], '.2f') + 'x' or '-'} |")
                summary.setdefault(rv, {})[f"{name}|{pol}"] = {k: v for k, v in x.items() if k != "_raw"}
        json.dump(summary, open(os.path.join(D_dir, "summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
