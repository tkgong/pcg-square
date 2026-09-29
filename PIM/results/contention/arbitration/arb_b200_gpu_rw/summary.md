
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 11% [11%-11%] | - | 17% [17%-17%] | 942 / 6411 | 970 / 6437 | 88% | 1.31x | - | 1.03x | 1.03x | 0.92x |
| (c) GPU+SPU+DRU | gpu | 11% [11%-11%] | 3% [3%-3%] | 17% [17%-17%] | 861 / 5583 | 888 / 5605 | 86% | 1.30x | 1.70x | 1.04x | 1.04x | 0.59x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 12% [12%-12%] | - | 17% [17%-17%] | 685 / 5457 | 711 / 5521 | 89% | 1.19x | - | 1.06x | 1.06x | 0.85x |
| (c) GPU+SPU+DRU | gpu | 12% [12%-12%] | 4% [4%-4%] | 17% [17%-17%] | 580 / 5453 | 604 / 5493 | 87% | 1.19x | 1.50x | 1.07x | 1.07x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 12% [12%-12%] | - | 16% [16%-16%] | 462 / 4839 | 485 / 4879 | 90% | 1.16x | - | 1.11x | 1.11x | 0.74x |
| (c) GPU+SPU+DRU | gpu | 12% [12%-12%] | 4% [4%-4%] | 16% [16%-16%] | 462 / 4367 | 486 / 4393 | 87% | 1.14x | 1.28x | 1.11x | 1.11x | 0.60x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 15% [15%-15%] | 279 / 4051 | 300 / 4089 | 91% | 1.10x | - | 1.17x | 1.17x | 0.59x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 5% [5%-5%] | 15% [15%-15%] | 393 / 5457 | 415 / 5541 | 87% | 1.14x | 1.06x | 1.16x | 1.16x | 0.59x |
