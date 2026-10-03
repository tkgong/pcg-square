# End-to-end results (one accounting)

GPU baseline = the submission's GPU implementation (DPF with the two-pass leaf conversion) + GPU-NTT merge
(the SOTA NTT), with the network co-scheduled: T_base = max(T_GPU, T_net).
PCG^2 = expansion on the SPUs (SPU at the DRAM clock, per-leaf H' on the SPU, LSU overlap), NTT = four-step on
GPU-NTT kernels with the transposes on the DRU (both machines);
PCG^2 = max(SPU lane, NTT lane, network) with the co-simulated contention (e2e_window.py, 22-multiply window).

* `final/results.txt`   Fig. 8 (all SPU clocks, single ALU), mechanism decomposition, Table IV (DRU rows = the ablation inside the final design), window-size check.  `final/dru_ablation.txt` the per-cell DRU ablation.  `final/fig8/` the figures.  `final/attribution_vs_paper.txt` the step by step diff from the submission's numbers.  `final/energy_*.txt` system energy from the measured board power.  `final/contention_table.txt` per-channel occupancy and queueing.
                        window-size check.  `final/fig8/` the figures.  `final/attribution_vs_paper.txt` the step
                        by step diff from the submission's numbers.  `final/energy_*.txt` system energy from the
                        measured board power.  `final/contention_table.txt` per-channel occupancy and queueing.
* `win22/`              the raw co-simulation rows per cell (e2e_<clock>.json, e2e_1alu.json).
* `fourstep_dru/`       the DRU against the submission's own four-step NTT (ideal and conservative DRU models),
                        NTT-vs-NTT, channel sweep.  `channel_sweep/`, `ntt_vs_ntt/` the same for the final NTT.
* `window_check/`       1x/2x/4x windows.
