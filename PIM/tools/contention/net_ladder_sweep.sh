#!/bin/bash
# SPU window trace with the simulated network ladder, serial vs co-scheduled, at r = T_net/T_compute in {0.1,0.3,1,3,10}.
# Usage: net_ladder_sweep.sh WIN22_RUN_DIR OUT_DIR   (needs the SPU-alone .out files of the run for T_compute)
set -e; W=$1; O=$2; mkdir -p $O; cd "$(dirname "$0")/../.."
SIM=sim/build/ramulator2; GEN=tools/gen_dpf_tree_trace.py
declare -A YAML=([l40s]=sim/test/gddr6_dpf_24ch.yaml [b200]=sim/test/hbm3e_dpf_b200d.yaml)
CTRL="-p MemorySystem.Controller.fpu_gate_issue=false -p MemorySystem.Controller.wr_max_age=1000 -p MemorySystem.Controller.pim_row_wait=0 -p MemorySystem.Controller.class_priority=0,1,3,2 -p MemorySystem.Controller.class_min_run=64 -p MemorySystem.DRAM.org.channel=2 -p Frontend.issue_width=2 -p MemorySystem.Controller.dru_bus_slot=2"
for org in b200 l40s; do
  TC=$(grep -oP "pim_done_cycles:\s*\K[0-9]+" $W/spu_${org}_chacha_x1_i16_n12_clk.out)
  for r in 0.1 0.3 1 3 10; do A=$(python3 -c "print(int($r*$TC/(16*14)))")
    for mode in serial sched; do T=$O/spu_${org}_${mode}_r${r}.trace
      python3 $GEN -n 12 -C 2 -P 8 -I 256 --mode instances --seed-bits 128 --reread --broadcast --cl 155 --reduce chacha --convert-cl 465 --modmul-cl 32 --net $mode --net-alpha-ck $A -o $T >/dev/null
      $SIM -f ${YAML[$org]} -t $T $CTRL > ${T%.trace}.out 2>&1 &
    done; done; done; wait
