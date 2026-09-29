
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 17% [17%-17%] | 355 / 4129 | 376 / 4145 | 85% | 1.13x | - | 1.05x | 1.05x | 0.94x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 17% [17%-17%] | 337 / 2837 | 358 / 2855 | 84% | 1.15x | 1.10x | 1.05x | 1.05x | 0.59x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 16% [16%-16%] | 237 / 2259 | 257 / 2285 | 86% | 1.08x | - | 1.09x | 1.09x | 0.87x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 16% [16%-16%] | 220 / 4099 | 240 / 4113 | 86% | 1.10x | 0.99x | 1.09x | 1.09x | 0.60x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 15% [15%-15%] | 202 / 2335 | 222 / 2359 | 87% | 1.03x | - | 1.19x | 1.19x | 0.79x |
| (c) GPU+SPU+DRU | fair | 14% [14%-14%] | 6% [6%-6%] | 15% [15%-15%] | 191 / 2293 | 211 / 2321 | 85% | 1.04x | 0.96x | 1.18x | 1.18x | 0.64x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 13% [13%-13%] | 230 / 3419 | 250 / 3445 | 86% | 1.06x | - | 1.38x | 1.38x | 0.69x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 13% [13%-13%] | 292 / 3357 | 313 / 3377 | 83% | 1.08x | 0.97x | 1.37x | 1.37x | 0.69x |
