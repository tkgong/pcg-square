
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 11% [11%-11%] | - | 17% [17%-17%] | 942 / 6411 | 970 / 6437 | 88% | 1.31x | - | 1.03x | 1.03x | 0.92x |
| (c) GPU+SPU+DRU | fair | 10% [10%-10%] | 4% [4%-4%] | 17% [17%-17%] | 1012 / 6833 | 1039 / 7145 | 87% | 1.44x | 1.58x | 1.04x | 1.04x | 0.59x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 17% [17%-17%] | 685 / 5457 | 711 / 5521 | 89% | 1.19x | - | 1.06x | 1.06x | 0.85x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 17% [17%-17%] | 616 / 6109 | 639 / 6127 | 88% | 1.23x | 1.26x | 1.06x | 1.06x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 16% [16%-16%] | 462 / 4839 | 485 / 4879 | 90% | 1.16x | - | 1.11x | 1.11x | 0.74x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 16% [16%-16%] | 489 / 4813 | 512 / 4851 | 88% | 1.15x | 1.13x | 1.10x | 1.10x | 0.59x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 15% [15%-15%] | 284 / 4277 | 305 / 4309 | 91% | 1.10x | - | 1.17x | 1.17x | 0.58x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 16% [16%-16%] | 407 / 4433 | 429 / 4463 | 88% | 1.13x | 0.92x | 1.15x | 1.15x | 0.58x |
