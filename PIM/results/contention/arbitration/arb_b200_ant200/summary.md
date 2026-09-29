
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 5% [5%-5%] | - | 17% [17%-17%] | 2670 / 8193 | 2717 / 8193 | 94% | 2.63x | - | 1.05x | 1.05x | 0.94x |
| (c) GPU+SPU+DRU | fair | 5% [5%-5%] | 2% [2%-2%] | 17% [17%-17%] | 3221 / 8193 | 3248 / 8193 | 92% | 3.11x | 3.26x | 1.07x | 1.07x | 0.60x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 7% [7%-7%] | - | 17% [17%-17%] | 1964 / 8193 | 2004 / 8193 | 94% | 2.15x | - | 1.08x | 1.08x | 0.86x |
| (c) GPU+SPU+DRU | fair | 6% [6%-6%] | 2% [2%-2%] | 17% [17%-17%] | 2020 / 8193 | 2058 / 8193 | 92% | 2.29x | 2.32x | 1.09x | 1.09x | 0.60x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 8% [8%-8%] | - | 16% [16%-16%] | 1340 / 8193 | 1374 / 8193 | 94% | 1.78x | - | 1.13x | 1.13x | 0.75x |
| (c) GPU+SPU+DRU | fair | 7% [7%-7%] | 3% [3%-3%] | 16% [16%-16%] | 1442 / 8193 | 1465 / 8193 | 92% | 1.96x | 1.91x | 1.14x | 1.14x | 0.61x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 9% [9%-9%] | - | 16% [16%-16%] | 1084 / 8193 | 1108 / 8193 | 93% | 1.60x | - | 1.16x | 1.16x | 0.58x |
| (c) GPU+SPU+DRU | fair | 9% [9%-9%] | 2% [2%-2%] | 16% [16%-16%] | 1165 / 8193 | 1189 / 8193 | 92% | 1.66x | 2.45x | 1.15x | 1.15x | 0.58x |
