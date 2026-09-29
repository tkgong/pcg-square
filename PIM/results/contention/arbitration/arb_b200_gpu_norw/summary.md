
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 17% [17%-17%] | 328 / 2691 | 349 / 2717 | 83% | 1.08x | - | 1.05x | 1.05x | 0.94x |
| (c) GPU+SPU+DRU | gpu | 12% [12%-12%] | 5% [5%-5%] | 17% [17%-17%] | 308 / 3061 | 329 / 3087 | 81% | 1.17x | 1.23x | 1.05x | 1.05x | 0.59x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 14% [14%-14%] | - | 16% [16%-16%] | 246 / 2733 | 266 / 2751 | 85% | 1.04x | - | 1.10x | 1.10x | 0.88x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 5% [5%-5%] | 16% [16%-16%] | 242 / 2461 | 263 / 2477 | 82% | 1.09x | 1.16x | 1.09x | 1.09x | 0.60x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 14% [14%-14%] | - | 15% [15%-15%] | 224 / 3577 | 244 / 3601 | 86% | 1.04x | - | 1.20x | 1.20x | 0.80x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 5% [5%-5%] | 15% [15%-15%] | 236 / 3527 | 256 / 3543 | 82% | 1.07x | 1.10x | 1.16x | 1.16x | 0.62x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 13% [13%-13%] | 245 / 3547 | 265 / 3567 | 85% | 1.07x | - | 1.40x | 1.40x | 0.70x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 5% [5%-5%] | 14% [14%-14%] | 254 / 3623 | 275 / 3645 | 82% | 1.08x | 1.18x | 1.32x | 1.32x | 0.67x |
