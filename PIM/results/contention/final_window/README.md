# End-to-end co-simulation, steady-state window (e2e_window.py)

Per Fig. 8 cell: PCG^2 = max(SPU lane x SPU slowdown, NTT lane x NTT slowdown, NIC),
slowdowns from one SPU-trace window co-run with an NTT kernel pipeline sized to the
cell's lane ratio (kernels >= 20k CK, a size the standalone validation reproduces).
GPU kernels carry measured compute floors; DRU transposes use the repo's DRU tap.
Baseline "serial-net" = GPU DPF + GPU-NTT merge + network (paper's accounting),
"overlap-net" = max(GPU DPF + merge, network) (reviewer D), "paper" = square NTT.

* `summary_*.txt`, `e2e_*.json`, `fig8/`:  fpu_gate_issue=true (the campaign's serial PU:
  no load/compute/store overlap), SPU lane = paper anchor scaled by T(clock)/T(nominal).
* `overlap/`: fpu_gate_issue=false = the LSU overlap described in Sec. IV (next op's loads
  and the previous op's stores proceed while the FPU computes; compute still serialises
  on the one FPU per bank pair). SPU lane = paper anchor scaled.
* `overlap_simlane/`: as overlap, but the SPU lane is taken from the simulation itself
  (--reduce chacha: expand + per-leaf ChaCha8 hash H' + mod-p on the SPU; ceil(I/NSPU)
  instances per SPU). The paper's anchor (718,660 CK) matches --reduce mau (leaf hash not
  charged to the SPU), so this lane is the one consistent with Sec. IV.

SPU ALU bound: 2 SIMD ALUs x 8 lanes, dual issue; 2 ChaCha8 blocks per leaf (expand + H')
= 38.8 SPU cycles per leaf. L40S (192 SPUs): 11.1 G leaves/s at 2.25 GHz, 4.95 at 1 GHz;
the L40S GPU baseline does 3.7-9.8 G leaves/s, so at 1 GHz the geomean bound is ~1.0-1.2x.
