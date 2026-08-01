#!/usr/bin/env bash
# FINAL CAMPAIGN: 6 devices x 3 MAU cases x contention variants, one fire.
# Devices: L40S(bc_*, reuse), Ada5K, RTX PRO 6000 (GDDR7), A100(reuse),
# H200-half, B200-domain(reuse). All all-bank broadcast, 128b seeds, --reread,
# gated, full load (4 rounds; n=14 cross at 2 rounds).
# Points/device: {272,155}x{none,mau,fused} + cl0 + mau_sm x2 + mau_ntt1 x2
#                + n14_cl155_mau   (L40S extra: ntt05/ntt2 sensitivity)
# Resume-guard: skip if .out has cycles OR a live sim already runs that trace.
set -u
R=/home/tkgong/PCG-acceleration
S="$(dirname "$0")"
GEN="python3 $R/pim/tools/gen_dpf_tree_trace.py"

run() { # name yaml C iw n I extra...
  local name=$1 yaml=$2 C=$3 iw=$4 n=$5 I=$6; shift 6
  if [ -s "$S/$name.out" ] && grep -q memory_system_cycles "$S/$name.out"; then
    echo "skip(done) $name"; return; fi
  if pgrep -f "ramulator2.*$name.trace" >/dev/null 2>&1; then
    echo "skip(live) $name"; return; fi
  $GEN -n $n -C $C -P 8 -I $I --mode instances --seed-bits 128 --reread \
       --broadcast "$@" -o "$S/$name.trace" 2> "$S/$name.gen"
  nohup $R/pim/sim/build/ramulator2 -f $R/pim/sim/test/$yaml -t "$S/$name.trace" \
    -p MemorySystem.Controller.fpu_gate_issue=true -p Frontend.issue_width=$iw \
    > "$S/$name.out" 2>&1 &
  echo "fire $name"
}

dev() { # prefix yaml C
  local px=$1 yaml=$2 C=$3
  local I=$((C*8*4)) I14=$((C*8*2))
  run ${px}_cl272_none  $yaml $C $C 12 $I  --cl 272 --reduce none
  run ${px}_cl272_mau   $yaml $C $C 12 $I  --cl 272 --reduce mau
  run ${px}_cl272_fused $yaml $C $C 12 $I  --cl 272 --reduce fused
  run ${px}_cl155_none  $yaml $C $C 12 $I  --cl 155 --reduce none
  run ${px}_cl155_mau   $yaml $C $C 12 $I  --cl 155 --reduce mau
  run ${px}_cl155_fused $yaml $C $C 12 $I  --cl 155 --reduce fused
  run ${px}_cl0_none    $yaml $C $C 12 $I  --cl 0   --reduce none
  run ${px}_cl272_mau_sm   $yaml $C $C 12 $I --cl 272 --reduce mau --sm-stream on
  run ${px}_cl155_mau_sm   $yaml $C $C 12 $I --cl 155 --reduce mau --sm-stream on
  run ${px}_cl272_mau_ntt1 $yaml $C $C 12 $I --cl 272 --reduce mau --mau-ntt 1.0
  run ${px}_cl155_mau_ntt1 $yaml $C $C 12 $I --cl 155 --reduce mau --mau-ntt 1.0
  run ${px}_n14_cl155_mau  $yaml $C $C 14 $I14 --cl 155 --reduce mau
}

dev bc     gddr6_dpf_24ch.yaml   24     # L40S    192 PU (reuses existing bc_*)
dev ada5k  ada5k_dpf.yaml        16     # Ada5K   128 PU
dev gddr7  gddr7_dpf.yaml        64     # RTXPro  512 PU
dev a100   hbm2e_dpf_a100.yaml   80     # A100    640 PU (reuses existing)
dev h200h  h200h_dpf.yaml        96     # H200/2  768 PU (x2 = card)
dev b200   hbm3e_dpf_b200d.yaml  128    # B200/2  1024 PU (x2 = card)
# L40S-only mau-ntt sensitivity bounds
run bc_cl155_mau_ntt05 gddr6_dpf_24ch.yaml 24 24 12 768 --cl 155 --reduce mau --mau-ntt 0.5
run bc_cl155_mau_ntt2  gddr6_dpf_24ch.yaml 24 24 12 768 --cl 155 --reduce mau --mau-ntt 2.0

echo "ALL FIRED $(date +%H:%M). Waiting for sims..."
while pgrep -f "ramulator2.*$S" >/dev/null 2>&1; do sleep 120; done
echo "SIMS DONE $(date +%H:%M). Running collector..."
python3 $R/pim/tools/finalize_campaign.py "$S" && echo CAMPAIGN-COMPLETE || echo COLLECTOR-FAILED
