
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 6% [6%-6%] | - | 17% [17%-17%] | 2669 / 7887 | 2703 / 7913 | 89% | 2.22x | - | 1.03x | 1.03x | 0.92x |
| (c) GPU+SPU+DRU | fair | 5% [5%-5%] | 1% [1%-1%] | 17% [17%-17%] | 3167 / 8193 | 3201 / 8193 | 85% | 2.77x | 3.85x | 1.03x | 1.03x | 0.58x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 8% [8%-8%] | - | 17% [17%-17%] | 1826 / 7013 | 1852 / 7049 | 90% | 1.74x | - | 1.05x | 1.05x | 0.84x |
| (c) GPU+SPU+DRU | fair | 7% [7%-7%] | 2% [2%-2%] | 17% [17%-17%] | 2038 / 8193 | 2067 / 8193 | 87% | 2.06x | 2.62x | 1.06x | 1.06x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 10% [10%-10%] | - | 16% [16%-16%] | 1213 / 6625 | 1237 / 6647 | 91% | 1.41x | - | 1.10x | 1.10x | 0.73x |
| (c) GPU+SPU+DRU | fair | 9% [9%-9%] | 3% [3%-3%] | 16% [16%-16%] | 1274 / 7409 | 1297 / 7431 | 89% | 1.51x | 2.04x | 1.12x | 1.12x | 0.60x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 11% [11%-11%] | - | 15% [15%-15%] | 1010 / 5899 | 1034 / 5917 | 91% | 1.25x | - | 1.18x | 1.18x | 0.59x |
| (c) GPU+SPU+DRU | fair | 10% [10%-10%] | 2% [2%-2%] | 15% [15%-15%] | 1285 / 7625 | 1311 / 7669 | 88% | 1.44x | 2.79x | 1.16x | 1.19x | 0.60x |
