
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 10% [10%-10%] | 1334 / 6814 | 1360 / 6831 | 91% | 4.86x | - | 1.09x | 1.09x | 0.98x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 2% [2%-2%] | 10% [10%-10%] | 1411 / 7251 | 1438 / 7268 | 90% | 5.84x | 14.43x | 1.12x | 1.12x | 0.60x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 16% [16%-16%] | - | 10% [10%-10%] | 1173 / 5903 | 1198 / 5922 | 92% | 4.28x | - | 1.18x | 1.18x | 0.94x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 2% [2%-2%] | 10% [10%-10%] | 1406 / 7034 | 1433 / 7064 | 89% | 5.54x | 13.70x | 1.15x | 1.28x | 0.65x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 10% [10%-10%] | - | 9% [9%-9%] | 1862 / 7275 | 1900 / 7275 | 89% | 6.79x | - | 1.25x | 1.40x | 0.93x |
| (c) GPU+SPU+DRU | fair | 10% [10%-10%] | 3% [3%-3%] | 9% [9%-9%] | 1823 / 7275 | 1855 / 7275 | 88% | 7.15x | 9.51x | 1.21x | 1.35x | 0.73x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 9% [9%-9%] | 1589 / 7275 | 1628 / 7275 | 88% | 5.80x | - | 1.24x | 1.95x | 0.98x |
| (c) GPU+SPU+DRU | fair | 10% [10%-10%] | 4% [4%-4%] | 9% [9%-9%] | 1800 / 7275 | 1840 / 7275 | 85% | 7.11x | 7.45x | 1.21x | 1.36x | 0.85x |
