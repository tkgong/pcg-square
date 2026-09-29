
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 23% [23%-23%] | - | 11% [11%-11%] | 836 / 7275 | 861 / 7275 | 95% | 3.11x | - | 1.06x | 1.06x | 0.95x |
| (c) GPU+SPU+DRU | fair | 16% [16%-16%] | 2% [2%-2%] | 10% [10%-10%] | 1107 / 7275 | 1136 / 7275 | 94% | 4.50x | 11.75x | 1.10x | 1.10x | 0.60x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 26% [26%-26%] | - | 10% [10%-10%] | 714 / 7275 | 739 / 7275 | 95% | 2.66x | - | 1.11x | 1.11x | 0.89x |
| (c) GPU+SPU+DRU | fair | 19% [19%-19%] | 2% [2%-2%] | 10% [10%-10%] | 932 / 7275 | 957 / 7275 | 94% | 3.70x | 11.68x | 1.11x | 1.20x | 0.61x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 20% [20%-20%] | - | 10% [10%-10%] | 965 / 7275 | 991 / 7275 | 93% | 3.56x | - | 1.17x | 1.17x | 0.78x |
| (c) GPU+SPU+DRU | fair | 23% [23%-23%] | 3% [3%-3%] | 10% [10%-10%] | 766 / 7275 | 791 / 7275 | 94% | 3.04x | 8.18x | 1.13x | 1.24x | 0.66x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 16% [16%-16%] | - | 9% [9%-9%] | 1182 / 5409 | 1216 / 5475 | 91% | 4.38x | - | 1.19x | 1.64x | 0.82x |
| (c) GPU+SPU+DRU | fair | 16% [16%-16%] | 4% [4%-4%] | 10% [10%-10%] | 1099 / 5450 | 1130 / 5473 | 91% | 4.37x | 6.41x | 1.17x | 1.20x | 0.75x |
