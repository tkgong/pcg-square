#!/usr/bin/env bash
# Reproduce the DPF-on-HBM3-PIM experiments. Run from the workspace root:
#   bash tools/run_sweeps.sh
# Requires sim built first:  ( cd sim && cmake -B build && cmake --build build -j )
set -e
cd "$(dirname "$0")/.."
SIM=sim/build/ramulator2
GEN="python3 tools/gen_dpf_trace.py"
OUT=results/sweeps.txt
mkdir -p results
cyc() { $SIM -f "$1" -t "$2" "${@:3}" 2>&1 | grep -oE 'memory_system_cycles: [0-9]+' | grep -oE '[0-9]+'; }

{
echo "== DPF on HBM3-PIM (tCK=625ps), single-issue serial PU. Ref: 1batch=337, 408/b (240 mem + 274 compute serial), packed 347/b =="
$GEN -B 1    -C 64 --cl 274 -o sim/test/_s1.trace   >/dev/null
$GEN -B 1024 -C 64 --cl 274 -o sim/test/_s64.trace  >/dev/null
$GEN -B 1024 -C 1  --cl 274 -o sim/test/_s1ch.trace >/dev/null
$GEN -B 1024 -C 64 --cl 0   -o sim/test/_c0.trace   >/dev/null
$GEN -B 1024 -C 64 --cl 1000 -o sim/test/_c1k.trace >/dev/null
$GEN -B 1024 -C 64 --cl 274 --packed -o sim/test/_pk.trace  >/dev/null
$GEN -B 1024 -C 64 --cl 0   --packed -o sim/test/_pk0.trace >/dev/null

# Single-issue serial PU (one instruction at a time, no overlap) = fpu_gate_issue=true.
G="-p MemorySystem.Controller.fpu_gate_issue=true"
echo "sanity 1batch@64ch (expect 337):        $(cyc sim/test/hbm3_dpf_64ch.yaml sim/test/_s1.trace $G)"
echo "A  64ch B=1024 cl=274 (408/batch):      $(cyc sim/test/hbm3_dpf_64ch.yaml sim/test/_s64.trace $G)   (16 batch/ch)"
echo "A  1ch  B=1024 cl=274 (scaling 65.6x):  $(cyc sim/test/hbm3_dpf.yaml sim/test/_s1ch.trace $G)"
echo "B  64ch cl=0   (mem only, 240/batch):   $(cyc sim/test/hbm3_dpf_64ch.yaml sim/test/_c0.trace $G)"
echo "B  64ch cl=1000:                        $(cyc sim/test/hbm3_dpf_64ch.yaml sim/test/_c1k.trace $G)"
echo "C  64ch packed cl=274 (347/batch):      $(cyc sim/test/hbm3_dpf_64ch.yaml sim/test/_pk.trace $G)"
echo "C  64ch packed cl=0:                    $(cyc sim/test/hbm3_dpf_64ch.yaml sim/test/_pk0.trace $G)"
} | tee "$OUT"
rm -f sim/test/_*.trace
echo "saved -> $OUT"
