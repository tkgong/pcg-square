#!/bin/bash
# Every reported number of the paper from the committed inputs, in order (final_design.py accounting):
#   calib     DRAM-traffic fractions of the Cheddar-kernel designs (cheddar, f4c, f4b) from the silicon tables
#   bf        main window at 1 / 2 / 4 butterfly units per channel -> <RUN>/dru_bf.json (U per card)
#   cosim     the window set: SPU clock sweep, single ALU, compute-ratio sweep, window-size check, channel-count sweep
#   analysis  results.txt, Table IV, DRU ablation, attribution, NTT designs, contention, arbitration, DSE, energy ...
#   plots     Fig. 8 (+energy), Fig. 9, channel DSE, co-scheduling figures
# Inputs: GPU_baseline/results_<card>/cheddar_bigN_<card>.csv (+ interfere, power) for both cards; PIM/sim/build/ramulator2.
# Every e2e_window.py run uses the same flags for every design (FLAGS below); raw run dirs are <RUN>_<tag>/ next to <RUN>/,
# the per-run rows are copied to <RUN>/e2e_<tag>.json. Cards whose silicon table lacks a design's SM lane simply get no
# rows for it (e2e_window skips them: no estimates).
# Usage: run_final.sh [calib|bf|cosim|analysis|plots|all ...]      env: PCG2_RUN (window set, default win23), JOBS (90)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
FW="$HERE/../../results/contention/final_window"; SIM="$HERE/../../sim/build/ramulator2"
RUN="${PCG2_RUN:-win23}"; W="$FW/$RUN"; F="$FW/final"; WC="$FW/window_check_$RUN"; JOBS="${JOBS:-90}"
GB="$HERE/../../../GPU_baseline"
mkdir -p "$W" "$F" "$WC"
export PCG2_RUN="$RUN"
FLAGS=(--sim "$SIM" --gather --spu-presum --fpu-gate false --spu-lane sim --win-inst 16 --win-depth 12 --dru-stream --occupancy range --gather-dru --f4c-sm-bound --jobs "$JOBS")
DESIGNS=merge,cheddar,f4c,f4csm,f4b
LOG="$W/run_final.log"; log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }
# plots need matplotlib: the first interpreter that has it (PYPLOT overrides)
PY="${PYPLOT:-}"; [ -n "$PY" ] || for c in python3 "$HOME"/miniconda3/envs/*/bin/python "$HOME"/miniconda3/bin/python; do "$c" -c "import matplotlib" 2>/dev/null && { PY="$c"; break; }; done
ubf() { python3 -c "import json,sys; print(json.load(open('$W/dru_bf.json')).get('$1', 1))" 2>/dev/null || echo 1; }
# one window run: tag, then e2e_window args; both cards into the same raw dir, each with its own U, rows concatenated
run() {
  local tag=$1; shift; local O="$FW/${RUN}_$tag"; mkdir -p "$O"
  for org in l40s b200; do
    log "e2e_window $tag $org (U=$(ubf $org)): $*"
    python3 e2e_window.py --out "$O" --orgs "$org" --designs "$DESIGNS" --dru-bf "$(ubf $org)" "${FLAGS[@]}" "$@" > "$O/log_$org.txt" 2>&1
    cp "$O/e2e.json" "$O/e2e_$org.json"
  done
  python3 -c "import json,sys; json.dump([r for o in ('l40s','b200') for r in json.load(open('$O/e2e_'+o+'.json'))], open('$O/e2e.json','w'), indent=1)"
  cp "$O/e2e.json" "$W/e2e_$tag.json"
}
stage_calib() {
  log "calibrate_ntt_bytes (cheddar, f4c, f4b)"
  python3 calibrate_ntt_bytes.py --sim "$SIM" --out "$FW/calib_$RUN" --designs cheddar,f4c,f4b --devs l40s,b200 --jobs "$JOBS" | tee -a "$LOG"
}
stage_bf() {
  rm -f "$W/dru_bf.json"
  for U in 1 2 4; do
    local O="$FW/${RUN}_bf$U"; mkdir -p "$O"
    for org in l40s b200; do
      log "e2e_window bf$U $org"
      python3 e2e_window.py --out "$O" --orgs "$org" --designs "$DESIGNS" --dru-bf "$U" "${FLAGS[@]}" --clock nom > "$O/log_$org.txt" 2>&1; cp "$O/e2e.json" "$O/e2e_$org.json"
    done
    python3 -c "import json; json.dump([r for o in ('l40s','b200') for r in json.load(open('$O/e2e_'+o+'.json'))], open('$O/e2e.json','w'), indent=1)"
    cp "$O/e2e.json" "$W/e2e_bf$U.json"
  done
  python3 dru_bf_sweep.py "$F/dru_bf_sweep.txt" "$W/dru_bf.json" 1="$FW/${RUN}_bf1" 2="$FW/${RUN}_bf2" 4="$FW/${RUN}_bf4" | tee -a "$LOG"
}
stage_cosim() {
  rm -f "$W/final_choice.json"
  for clk in nom 1.75 1.5 1.0; do run "$clk" --clock "$clk"; done
  run 1alu --clock nom --alu-mult 0.5556
  for x in 0.5 0.75 1.25; do run "x$x" --clock nom --alu-mult "$x"; done
  for i in 8 32; do run "i${i}_n12" --clock nom --win-inst "$i"; mkdir -p "$WC/i${i}_n12"; cp "$W/e2e_i${i}_n12.json" "$WC/i${i}_n12/e2e.json"; done
  # channel-count sweep: the card's DRAM with a hypothetical channel count (8 SPUs + 1 DRU per channel)
  local O="$FW/${RUN}_chsweep"; mkdir -p "$O"
  for org in b200 l40s; do
    local chs; [ "$org" = b200 ] && chs="8 16 32 48 96 128 192 512" || chs="8 16 48 96 128 192 256 512"
    local orgs=""; for c in $chs; do orgs="$orgs${orgs:+,}${org}_ch$c"; done
    log "e2e_window chsweep $orgs"
    python3 e2e_window.py --out "$O" --orgs "$orgs" --designs merge,cheddar,f4c,f4b --dru-bf "$(ubf $org)" "${FLAGS[@]}" --clock nom > "$O/log_$org.txt" 2>&1; cp "$O/e2e.json" "$O/e2e_$org.json"
  done
  python3 -c "import json; json.dump([r for o in ('b200','l40s') for r in json.load(open('$O/e2e_'+o+'.json'))], open('$W/channel_sweep.json','w'), indent=1)"
  python3 final_design.py | tee -a "$LOG"
}
stage_analysis() {
  python3 final_design.py | tee "$F/final_choice.txt"
  python3 final_results.py "$W" "$WC" "$F" > /dev/null
  python3 table4.py "$W" "$F/table4.txt" > /dev/null
  python3 dru_ablation.py "$W" "$F/dru_ablation.txt" > /dev/null
  python3 attribution.py "$W" "$F/attribution_vs_paper.txt" > /dev/null
  python3 attribution_per_config.py "$W" "$F/attribution_per_config.txt" > /dev/null
  python3 ntt_designs.py "$W" "$F/ntt_designs.txt" > /dev/null
  python3 cosched_sweep.py "$W" "$F/cosched_sweep.txt" "$F/cosched_sweep.json" > /dev/null
  python3 spu_capability_sweep.py "$W" "$F/spu_capability_sweep.txt" > /dev/null
  python3 compute_ratio_sweep.py "$F/compute_ratio_sweep.txt" x0.5="$FW/${RUN}_x0.5" x0.75="$FW/${RUN}_x0.75" x1="$FW/${RUN}_nom" x1.25="$FW/${RUN}_x1.25" > /dev/null
  python3 channel_dse.py "$W" "$F/channel_dse.txt" > /dev/null
  python3 channel_dse_table.py "$W" "$F/channel_dse_table.txt" > /dev/null
  python3 contention_table.py "$FW/${RUN}_nom" > "$F/contention_table.txt"
  python3 contention_table.py "$FW/${RUN}_nom" all > "$F/contention_table_all.txt"
  python3 contention_geomean.py "$FW/${RUN}_nom" > "$F/contention_geomean.txt"
  python3 arbitration_ablation.py --sim "$SIM" --run "$FW/${RUN}_nom" --out "$FW/arb_$RUN" --jobs "$JOBS" > /dev/null; cp "$FW/arb_$RUN/summary.txt" "$F/arbitration_ablation.txt"
  python3 arbitration_ablation.py --sim "$SIM" --run "$FW/${RUN}_nom" --out "$FW/arb_${RUN}_all" --cells all --jobs "$JOBS" > /dev/null; cp "$FW/arb_${RUN}_all/summary.txt" "$F/arbitration_ablation_all.txt"
  python3 interfere_at_spu.py --sim "$SIM" --out "$FW/ifspu_$RUN" > "$F/interfere_at_spu.txt"
  python3 ntt_nogather.py "$FW/nogather_$RUN" "$FW/${RUN}_nom" > /dev/null; cp "$FW/nogather_$RUN/ratios.txt" "$F/ntt_nogather_ratios.txt"
  for m in L40S B200; do python3 power_energy.py "$m" "$GB/results_$(echo $m | tr A-Z a-z)/power" "$F/energy_$(echo $m | tr A-Z a-z).txt" "$W" > /dev/null; done
  log "analysis written to $F"
}
stage_plots() {
  for c in nom 1.0; do "$PY" plot_fig8_cosim.py "$W/e2e_$c.json" "$F/fig8/fig8_spu_$([ "$c" = nom ] && echo dramclk || echo 1ghz).png" > /dev/null; done
  "$PY" plot_fig8_paper_style.py "$W/e2e_nom.json" "$F/fig8/fig8_paper_style.png" > /dev/null
  "$PY" plot_fig8_energy.py "$W/e2e_nom.json" "$F/fig8/fig8_energy.png" "$F/energy_l40s.json" "$F/energy_b200.json" > /dev/null
  "$PY" plot_speedup_energy.py "$W/e2e_nom.json" "$F/energy_l40s.json" "$F/energy_b200.json" "$F/fig_speedup_energy.png" > /dev/null
  "$PY" plot_fig9_budget.py "$W" "$F/fig9_budget.png" "$F/fig9_budget.txt" > /dev/null
  "$PY" plot_fig9_two.py "$W" "$F/fig9_two.png" > /dev/null
  "$PY" plot_channel_dse.py "$W" "$F/fig_channel_dse.png" > /dev/null
  "$PY" plot_cosched_sweep.py "$F/cosched_sweep.json" "$F/fig_cosched_sweep.png" > /dev/null
  "$PY" plot_cosched_4config.py "$W" "$F/fig_cosched_4config.png" > /dev/null
  for p in e2e norm norm2 speedup; do "$PY" plot_cosched_$p.py "$F/cosched_sweep.json" "$F/fig_cosched_$p.png" > /dev/null; done
  log "plots written to $F"
}
[ $# -eq 0 ] && set -- all
for s in "$@"; do
  case "$s" in
    calib) stage_calib;; bf) stage_bf;; cosim) stage_cosim;; analysis) stage_analysis;; plots) stage_plots;;
    all) stage_calib; stage_bf; stage_cosim; stage_analysis; stage_plots;;
    *) echo "unknown stage $s"; exit 2;;
  esac
done
