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

## Final (calibrated) runs: `v4_l40s`, `v4_b200`

GPU stream model fitted to silicon (`calib_gpu_model/`: 128 reads in flight
per channel, 32 columns per row visit reproduces the L40S aggressor
measurement, 1.09x at 15-30% aggressor BW vs 1.06-1.13x measured), 2 channels
(channels are symmetric), fair arbitration, `pim_row_wait=0`, `wr_max_age=1000`.
Config (c), completion vs the paper's max():

| r    | L40S | B200 |
|------|------|------|
| 0.12 | 1.11 | 1.04 |
| 0.25 | 1.16 | 1.07 |
| 0.5  | 1.14 | 1.11 |
| 1    | 1.19 | 1.16 |

Fig. 8 (40 Gbps, best / geomean): L40S 3.19/2.54 -> 2.70/2.25,
B200 9.49/7.25 -> 8.25/6.60 (`sim_e2e.py`).
Silicon SPU-like (row-burst) aggressor on L40S at the SPU's 11-15% BW share:
NTT 1.06-1.10x, same as a streaming aggressor.
