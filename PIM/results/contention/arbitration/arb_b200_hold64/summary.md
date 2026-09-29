
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 9% [9%-9%] | - | 17% [17%-17%] | 1295 / 6407 | 1328 / 6433 | 84% | 1.50x | - | 1.04x | 1.04x | 0.93x |
| (c) GPU+SPU+DRU | fair | 10% [10%-10%] | 3% [3%-3%] | 17% [17%-17%] | 895 / 6297 | 922 / 6315 | 83% | 1.46x | 2.01x | 1.05x | 1.05x | 0.59x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 11% [11%-11%] | - | 17% [17%-17%] | 1008 / 5463 | 1036 / 5501 | 84% | 1.30x | - | 1.07x | 1.07x | 0.86x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 3% [3%-3%] | 17% [17%-17%] | 699 / 5011 | 726 / 5043 | 85% | 1.21x | 1.63x | 1.08x | 1.08x | 0.60x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 16% [16%-16%] | 809 / 5067 | 836 / 5111 | 85% | 1.19x | - | 1.12x | 1.12x | 0.75x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 4% [4%-4%] | 16% [16%-16%] | 720 / 5019 | 746 / 5043 | 85% | 1.21x | 1.34x | 1.12x | 1.12x | 0.60x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 15% [15%-15%] | 559 / 4933 | 583 / 4963 | 88% | 1.12x | - | 1.20x | 1.20x | 0.60x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 15% [15%-15%] | 661 / 5073 | 687 / 5099 | 86% | 1.17x | 1.14x | 1.18x | 1.18x | 0.60x |
