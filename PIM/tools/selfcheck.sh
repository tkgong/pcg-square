#!/usr/bin/env bash
# Layer-1 self-check (~1 min): syntax, model invariants, trace-op conservation,
# micro sim smoke. Run from repo root: bash pim/tools/selfcheck.sh
set -e
cd "$(dirname "$0")/../.."

echo "== syntax =="
for f in pim/tools/*.py; do python3 -m py_compile "$f"; done && echo OK

echo "== model invariants =="
python3 - <<'EOF'
import sys; sys.path.insert(0, 'pim/tools')
from e2e_stitched import cell
r = cell(4, 20, 985.0, 41.5, PUs=(64, 10**9))
assert abs(r['c1000000000'] - (r['residual'] + r['ntt'])) < 0.01   # PU->inf
assert abs(r['pim64'] - 4*2*(1<<20)*4*0.45/1e6) < 0.01             # pair formula
from superiority_map import polymul_l40s_us
us, flag = polymul_l40s_us(65536)
assert us == 20.2 and flag == ''                                    # silicon anchor
print("OK")
EOF

echo "== trace-op conservation (no sim needed) =="
python3 - <<'EOF'
import subprocess, tempfile, os
def count(args):
    tf = tempfile.NamedTemporaryFile(suffix=".trace", delete=False)
    subprocess.run(["python3","pim/tools/gen_dpf_tree_trace.py","-o",tf.name]+args,
                   capture_output=True, check=True)
    n = open(tf.name).read().count("GGM_REDUCE"); os.unlink(tf.name); return n
base = ["-n","10","--mode","instances","-I","256","-C","64","--reread","--cl","272"]
r0 = count(base+["--reduce","none"])
r1 = count(base+["--reduce","mau"])
r2 = count(base+["--reduce","fused"])
r3 = count(base+["--reduce","mau","--sm-stream","on"])
extends = sum(max(1,(1<<l)//8 if (1<<l)>=8 else 1) for l in range(10))*256
assert r0 == extends and r1 == r2 and r1-r0 == 2048 and r3-r1 == 2048
print("OK  extends=%d mau=+%d sm=+%d" % (r0, r1-r0, r3-r1))
# chacha (new Beaver algo, broadcast): the LAST level is a fused CONVERT op
# (emitted as GGM_REDUCE) instead of an EXTEND, and there is NO re-read reduce
# tail -> total op count == none (conservation), unlike fused/mau (+tail).
bc = ["-n","10","--mode","instances","-I","192","-C","24","-P","8","--reread",
      "--broadcast","--cl","155"]
def wrcount(args, cl):   # count last-level fused ops = GGM_REDUCE with this cl
    tf = tempfile.NamedTemporaryFile(suffix=".trace", delete=False)
    subprocess.run(["python3","pim/tools/gen_dpf_tree_trace.py","-o",tf.name]+args,
                   capture_output=True, check=True)
    body = open(tf.name).read(); os.unlink(tf.name)
    tot = body.count("GGM_REDUCE")
    fused = sum(1 for ln in body.splitlines()
                if ln.startswith("AiM GGM_REDUCE") and ln.split()[7] == str(cl))
    return tot, fused
cn, _ = wrcount(bc+["--reduce","none"], 0)
cc, cf = wrcount(bc+["--reduce","chacha","--convert-cl","465"], 465)
last = 24*((1<<9)//8)  # last level (L9) op count = ceil(2^9/8)*24 chans = 1536
assert cc == cn, "chacha must conserve op count vs none (%d vs %d)" % (cc, cn)
assert cf == last, "chacha fused-CONVERT count must == last-level ops (%d vs %d)" % (cf, last)
print("OK  chacha: total=%d (==none) fused-convert@cl465=%d (last level)" % (cc, cf))
# multi-PU channels (GDDR6_L40S 24ch x 8PU): identical op counts to P=1,
# banks spread over all 8 pairs
g6 = ["-n","10","--mode","instances","-I","384","-C","24","--reread","--cl","272"]
p1 = count(g6+["--reduce","mau"])
p8 = count(g6+["--reduce","mau","-P","8"])
assert p1 == p8, "P=8 must conserve ops vs P=1 (%d vs %d)" % (p8, p1)
import subprocess as sp, tempfile as tf2
t = tf2.NamedTemporaryFile(suffix=".trace", delete=False)
sp.run(["python3","pim/tools/gen_dpf_tree_trace.py","-o",t.name]+g6+["--reduce","mau","-P","8"],
       capture_output=True, check=True)
banks = {int(l.split()[4]) for l in open(t.name) if "GGM_" in l}; os.unlink(t.name)
assert banks == set(range(16)), "P=8 must use banks 0..15, got %s" % sorted(banks)
print("OK  P=8 conserves ops (%d), banks 0..15 all used" % p8)
EOF

echo "== micro sim smoke (n=8, I=128, ~30s) =="
python3 pim/tools/gen_dpf_tree_trace.py -n 8 --mode instances -I 128 -C 64 \
    --reread --cl 272 --reduce mau -o pim/sim/test/_chk.trace 2>/dev/null
pim/sim/build/ramulator2 -f pim/sim/test/hbm3_dpf_64ch.yaml -t pim/sim/test/_chk.trace \
    -p MemorySystem.Controller.fpu_gate_issue=true 2>&1 | grep memory_system_cycles
rm -f pim/sim/test/_chk.trace

echo "== micro sim smoke GDDR6_L40S (n=8, I=192, 24ch x 8PU) =="
python3 pim/tools/gen_dpf_tree_trace.py -n 8 --mode instances -I 192 -C 24 -P 8 \
    --reread --cl 272 --reduce mau -o pim/sim/test/_chk_g6.trace 2>/dev/null
pim/sim/build/ramulator2 -f pim/sim/test/gddr6_dpf_24ch.yaml -t pim/sim/test/_chk_g6.trace \
    -p MemorySystem.Controller.fpu_gate_issue=true -p Frontend.issue_width=24 \
    2>&1 | grep memory_system_cycles
rm -f pim/sim/test/_chk_g6.trace
echo "ALL LAYER-1 CHECKS PASSED"
