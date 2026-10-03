#!/usr/bin/env python3
"""Print the GPU's cumulative energy counter (NVML nvmlDeviceGetTotalEnergyConsumption, mJ) for GPU index
argv[1]; prints NA when pynvml or the counter is unavailable. Read before and after a phase and subtract."""
import sys
try:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
except Exception:
    print("NA")
