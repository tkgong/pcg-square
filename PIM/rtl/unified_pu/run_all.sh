#!/bin/bash
# Regenerates results.txt.  Needs Yosys with yosys-config (conda install -c conda-forge yosys), g++, and the NanGate 45 nm
# open cell library, typical corner, as ./nangate45.lib or LIB=<path>
# (OpenROAD-flow-scripts: flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib).
set -eu
cd "$(dirname "$0")"
export YOSYS=${YOSYS:-yosys} LIB=${LIB:-nangate45.lib}
INC=${CXXRTL_INC:-$(${YOSYS}-config --datdir)/include/backends/cxxrtl/runtime}
mkdir -p out
{
echo "Unified processing unit: the SPU reconfigured to also run AiM's BF16 functions (bf16.v, arx.v, pu.v; one source, three"
echo "configurations).  $($YOSYS -V | cut -d' ' -f1-2), NanGate 45 nm open cell library (typical), clock gates, ABC mapping."
echo "The AiM-function unit is a reconstruction from the public description (16 BF16 multipliers + adder tree per unit, MAC,"
echo "element-wise multiply and add) on an HBM-PIM-style frame (2 x 8 x 256 b registers, 16 x 16 b scalars, 32-instruction"
echo "memory); it is not a vendor netlist.  The SPU follows PIM/docs/pim_isa_assembly.md (dual issue, 16 x 256 b + scalars)."
echo "Not in the RTL: AiM's activation-function table, the last-level fused CONVERT micro-sequence (its mod-ALU hooks are"
echo "in and tested single-stepped).  Areas are post-mapping cell area, no placement, no memory compiler."
echo
echo "1. Function (C++ simulation of the RTL, tb.cc)"
for cfg in "uni -DHAS_ARX -DHAS_BF16" "spu -DHAS_ARX" "aim -DHAS_BF16"; do set -- $cfg; t=$1; shift
  $YOSYS -q -p "read_verilog -sv bf16.v arx.v pu.v; hierarchy -top pu_$t -check; proc; flatten; opt; write_cxxrtl -O4 out/pu_$t.h" > /dev/null
  g++ -O1 -std=c++14 -I"$INC" -Iout -DDESIGN_H="\"pu_$t.h\"" -DTOP=p_pu__$t "$@" tb.cc -o out/tb_$t
  echo "  configuration $t:"; ./out/tb_$t | grep -v "EXTEND batch" | sed 's/^/    /'
done
./run_synth.sh aim hier 444 & ./run_synth.sh spu hier 444 & ./run_synth.sh uni hier 444 &
./run_synth.sh aim tew 444 bf16_acc & ./run_synth.sh spu flat 444 &
./run_synth.sh uni tarx 444 bf16_dp bf16_acc & ./run_synth.sh uni tew 444 bf16_acc xadd vxorl lsu useq foldk &
wait
echo
echo "2. Area (kGE = cell area / NAND2_X1)"
python3 report.py
echo
echo "3. Critical paths, ps (same library; absolute speed is not the claim, the comparison between configurations is)"
python3 timing.py out/aim_tew_444.log "  AiM-function unit (pu = register read -> BF16 element-wise -> register write; bf16_acc = MAC sum stage)" \
  out/uni_tew_444.log "  unified, BF16 side (ARX blocks cut out of the paths)" \
  out/spu_flat_444.log "  SPU (whole unit)" \
  out/uni_tarx_444.log "  unified, ARX side (BF16 datapath cut out of the paths)"
} > results.txt
cat results.txt
