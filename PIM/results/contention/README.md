# GPU-PIM contention (Reviewers B/D/E)

Co-simulation: one Ramulator2 model per GPU (L40S GDDR6, 8 of 24 channels; B200
HBM3e, 16 channels), SPU DPF trace (class 0, all-bank) + GPU NTT host stream
(class 1) + DRU transpose stream (class 2). Tools: `PIM/tools/contention/`.
Controller fixes used here: age-triggered write drain (`wr_max_age=1000`).

* `v2_l40s*`, `v2_b200*` — sweeps r = 0.12/0.25/0.5/1, configs (a)/(b)/(c),
  policies pim/fair/gpu; `_dru16` = 16 DRU requests in flight (paper: 4).
  `summary.md` has per-channel BW, GPU read queue/latency mean+p99, row hit,
  GPU/SPU slowdown, completion vs max() and vs serial.
* `diag_row_protect/` — L40S r = 0.12/0.5, fair, with `pim_row_wait` (host
  classes may only issue column commands while a PIM row change waits).
  Without it the SPU makes no progress while the GPU stream is active.
* `interfere_l40s.csv`, `../../../GPU_baseline/fused4/results/interfere_*B200*` —
  silicon: NTT slowdown vs a DRAM-bandwidth aggressor, net of the SM-occupancy
  control (L40S 1.05-1.20x at 11-31% BW; B200 1.017x/1.027x/1.05x at 12/23/40%).
* `replay_l40s/` — the silicon experiment replayed in the simulator: the
  simulator's NTT slowdown is 2-4x the measured one.
* `sim_e2e.py`, `contention_e2e.py` — Fig. 8 lane model with the contention
  factors (run from scratchpad e2e_verify, needs reproduce.py/fig8.py).
