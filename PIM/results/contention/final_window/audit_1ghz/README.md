Audit of the 1 GHz L40S result (SPU model as in Sec. IV: expand + per-leaf H' + mod-p on the SPU, LSU overlap).

Simulator's own FPU accounting (DPFDBG, channel 0, 2120 op lines, each = 8 SPUs in lockstep):
  2.25 GHz: run 674,914 CK, FPU busy 644,312 CK (95.5%)
  1.0 GHz : run 1,471,459 CK, FPU busy 1,439,464 CK (97.8%)  -> memory/sync/refresh exposed 2.2%
Compute is charged once per op line on a per-channel FPU key (AiM_dram_controller.cpp:438-475),
consecutive ops serialise on it; loads/stores overlap (fpu_gate_issue=false, ggm_compute_overlap=true).

Items found:
 1. Occupancy: ceil(I/NSPU) (makespan) vs I/NSPU (the paper's lane model). Fractional raises
    (8,4) at 1 GHz from 1.44x to 1.63x; geomean 1.17x -> 1.22x. At 2.25 GHz: 3.01/2.49 -> 3.31/2.58.
 2. The 128-CK mod-p accumulate ops (ADD64M/VXSUM, 128 of 4240 lines) are not scaled with the SPU
    clock in the trace; scaling them would add ~0.7% at 1 GHz (unfavourable, not applied).
Nothing else in the SPU path changes the 1 GHz result: it is 2 ChaCha8 per leaf x 155 SPU cycles
per 8 blocks on 2 x 8-lane ALUs x 192 SPUs = 9.9 G blocks/s at 1 GHz, vs the measured L40S GPU
baseline sustaining 11.1 G ChaCha8/s (3 per leaf: expand + H' in out_sums + H' in out_scatter).
