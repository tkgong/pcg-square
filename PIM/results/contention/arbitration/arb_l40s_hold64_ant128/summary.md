
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 21% [21%-21%] | - | 10% [10%-10%] | 897 / 7275 | 922 / 7275 | 94% | 3.29x | - | 1.07x | 1.07x | 0.95x |
| (c) GPU+SPU+DRU | fair | 15% [15%-15%] | 3% [3%-3%] | 10% [10%-10%] | 1162 / 7275 | 1188 / 7275 | 93% | 4.72x | 9.77x | 1.11x | 1.11x | 0.60x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 25% [25%-25%] | - | 10% [10%-10%] | 781 / 7275 | 807 / 7275 | 94% | 2.87x | - | 1.12x | 1.12x | 0.90x |
| (c) GPU+SPU+DRU | fair | 18% [18%-18%] | 3% [3%-3%] | 10% [10%-10%] | 1003 / 7275 | 1028 / 7275 | 93% | 3.97x | 10.77x | 1.12x | 1.19x | 0.61x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 16% [16%-16%] | - | 9% [9%-9%] | 1198 / 5718 | 1228 / 5734 | 92% | 4.42x | - | 1.19x | 1.23x | 0.82x |
| (c) GPU+SPU+DRU | fair | 21% [21%-21%] | 3% [3%-3%] | 10% [10%-10%] | 855 / 5171 | 879 / 5222 | 93% | 3.37x | 8.27x | 1.14x | 1.24x | 0.67x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 9% [9%-9%] | 1383 / 6862 | 1418 / 6881 | 90% | 5.06x | - | 1.21x | 1.80x | 0.90x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 4% [4%-4%] | 9% [9%-9%] | 1338 / 7275 | 1370 / 7275 | 89% | 5.27x | 6.77x | 1.19x | 1.26x | 0.78x |
