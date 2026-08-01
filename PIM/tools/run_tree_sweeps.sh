#!/usr/bin/env bash
# Tree-dataflow sweeps: real inter-layer GGM tree on 64ch, single-issue serial PU.
#   bash tools/run_tree_sweeps.sh
set -e
cd "$(dirname "$0")/.."
SIM=sim/build/ramulator2
YAML=sim/test/hbm3_dpf_64ch.yaml
G="-p MemorySystem.Controller.fpu_gate_issue=true"
OUT=results/tree_sweeps.txt
mkdir -p results

run() { # run <trace> -> "cycles ACTs RDs WRs"
  $SIM -f $YAML -t "$1" $G 2>&1 | awk '
    /memory_system_cycles:/ {c=$2}
    /CH[0-9]+_num_ACT_commands:/ {a+=$2}
    /CH[0-9]+_num_RD_commands:/ {r+=$2}
    /CH[0-9]+_num_WR_commands:/ {w+=$2}
    END {print c, a, r, w}'
}

{
echo "== GGM tree dataflow (inter-layer read-back), 64ch x 1 PU, serial PU, HBM3 tCK=625ps =="
echo "== columns: n  cl  reduce  ops  cycles  ACT  RD  WR  ns_total  ns_per_leaf =="
for n in 12 16 18 20; do
  for cl in 0 282; do
    ops=$(python3 tools/gen_dpf_tree_trace.py -n $n -C 64 --cl $cl -o sim/test/_tw.trace 2>&1 | grep -oE '[0-9]+ EXTEND' | grep -oE '[0-9]+')
    read cyc act rd wr <<< "$(run sim/test/_tw.trace)"
    ns=$(echo "$cyc*0.625" | bc)
    npl=$(echo "scale=4;$ns/(2^$n)" | bc)
    echo "n=$n cl=$cl reduce=none ops=$ops cycles=$cyc ACT=$act RD=$rd WR=$wr ns=$ns ns/leaf=$npl"
  done
done
# leaf reduction (fused leaf_convert) at n=16/18, cl=282, modmul=20
for n in 16 18; do
  ops=$(python3 tools/gen_dpf_tree_trace.py -n $n -C 64 --cl 282 --reduce fused -o sim/test/_tw.trace 2>&1 | grep -oE '[0-9]+ EXTEND' | grep -oE '[0-9]+')
  read cyc act rd wr <<< "$(run sim/test/_tw.trace)"
  echo "n=$n cl=282 reduce=fused ops=$ops cycles=$cyc ACT=$act RD=$rd WR=$wr"
done
# PCG instances mode anchor: N=2^16,t=8 -> dpf_n=log2(2N/t)=14, I=c^2*t^2=256
python3 tools/gen_dpf_tree_trace.py -n 14 --mode instances -I 256 -C 64 --cl 282 --reduce fused -o sim/test/_tw.trace >/dev/null 2>&1
read cyc act rd wr <<< "$(run sim/test/_tw.trace)"
echo "PCG(N=2^16,t=8): 256 instances x n=14, cycles=$cyc ACT=$act RD=$rd WR=$wr"
} | tee $OUT
rm -f sim/test/_tw.trace
echo "saved -> $OUT"
