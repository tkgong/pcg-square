# DPF split between SPUs and otherwise-idle SMs (e2e_window.py --dpf-split)

Scheduling only (no hardware change): the static scheduler assigns a (1-x) share of the DPF
instances to the SMs, which run them with the measured GPU expand+convert kernel sequentially
with the NTTs; the SPUs run the x share. x balances the two lanes:
    x * T_spu * s_spu = (T_ntt + (1-x) * T_gpu) * s_sm
with the co-run slowdowns s_spu, s_sm from the window (SM lane = NTT kernels + the DPF share as
an 88 B/leaf, 55%-read stream at the kernel's bandwidth with its measured compute floor).
PCG^2 = max(x T_spu s_spu, (T_ntt + (1-x) T_gpu) s_sm, NIC). Model otherwise as overlap_simlane
(H' on the SPU, LSU overlap, contention, SOTA baseline = GPU DPF + merge NTT).

40 Gbps best/geomean (400 Mbps in brackets):
  L40S, SPU 1 GHz     : 1.97 / 1.89   [2.15 / 2.02]   SPU share 0.53-0.73
  L40S, SPU 2.25 GHz  : 3.13 / 2.95   [3.22 / 2.82]   SPU share 0.73-0.96
  B200, SPU 1 GHz     : 6.32 / 5.36   [5.65 / 3.25]   SPU share 0.82-0.98
  B200, SPU 2.0 GHz   : 9.75 / 7.77   [5.97 / 3.47]   SPU share 0.90-1.00
The paper does not describe this policy; it is a change to the scheduler, not to the SPU/DRU.
