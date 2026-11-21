#!/bin/bash
# usage: run_synth.sh <aim|spu|uni> <tag> <period_ps> [modules kept as hierarchy ...]
#   tag = hier: the source's keep_hierarchy attributes (per-block area); any other tag: only the listed modules stay
#   separate, everything else is flattened into the top (timing of the paths that do not cross the listed modules)
# Yosys, NanGate 45 nm open cell library (typical), integrated clock gates, ABC mapping with a timing report
t=$1; tg=$2; D=$3; shift 3; Y=${YOSYS:-yosys}; L=${LIB:-nangate45.lib}; tag=${t}_${tg}_$D
if [ "$tg" = hier ]; then FL="flatten;"; else FL="setattr -mod -unset keep_hierarchy *;"; for m in "$@"; do FL="$FL setattr -mod -set keep_hierarchy 1 $m;"; done; FL="$FL flatten;"; fi
ABC="+strash;&get,-n;&fraig,-x;&put;scorr;dc2;dretime;strash;&get,-n;&dch,-f;&nf,{D};&put;buffer,-c;topo;upsize,-c;dnsize,-c;topo;stime,-p"
$Y -l out/$tag.log -q -p "
read_verilog -sv bf16.v arx.v pu.v; hierarchy -top pu_$t -check;
proc; $FL
synth -top pu_$t;
opt -full; clockgate -pos CLKGATE_X1 E:CK:GCK -min_net_size 8;
dfflibmap -liberty $L; abc -liberty $L -D $D -script $ABC; opt_clean;
tee -o out/$tag.stat stat -top pu_$t -liberty $L" > out/$tag.err 2>&1
