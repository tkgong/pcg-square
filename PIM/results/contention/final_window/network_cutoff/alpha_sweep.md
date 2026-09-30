# Network cutoff alpha* and speedup vs alpha (Sec. VI-D grid), from the co-simulation lanes
Lanes with contention (overlap_simlane); NIC = (n+2) alpha; PCG2 = max(SPU lane, NTT lane, NIC);
serial-net baseline = GPU DPF + merge NTT + NIC (paper), overlap-net = max(GPU, NIC) (reviewer D).
alpha* = geomean over cells of lane/(n+2). Knee = smallest SPU budget (steps of 8) whose SPU lane still fits under max(NIC, NTT lane).

L40S, SPU 2.25 GHz: alpha* 11.84 ms (paper 11.69)
  alpha   0.1   0.5   10    20    30    60  ms
  geomean 2.49  2.54  2.68  2.31  2.03  1.61   (best 3.03 3.08 2.95 2.66 2.40 1.99; vs overlap-net 2.47 2.47 1.93 1.56 1.35 1.12)
  knee    192   192   144   101   74    43 SPUs
B200, SPU 2.0 GHz: alpha* 2.00 ms (paper 1.75)
  geomean 7.54  7.52  2.68  1.92  1.64  1.33   (best 9.77 9.21 3.42 2.37 1.94 1.49; vs overlap-net 7.46 7.17 1.80 1.29 1.13 1.02)
  knee    1645  1592  312   156   104   55  SPUs   (paper: 2048 2048 1536 768 512 256)
L40S, SPU 1 GHz: alpha* 25.1 ms
  geomean 1.17  1.20  1.58  1.65  1.62  1.50   (best 1.45 1.48 1.84 1.82 1.74 1.61; vs overlap-net 1.17 1.17 1.14 1.11 1.08 1.04)
  knee    192   192   181   147   121   77
B200, SPU 1 GHz: alpha* 3.12 ms
  geomean 4.83  5.01  2.61  1.92  1.64  1.33   (best 6.21 6.35 3.37 2.37 1.94 1.49; vs overlap-net 4.78 4.78 1.75 1.29 1.13 1.02)
  knee    2048  2048  594   302   195   100
