
### L40S (2 channels simulated), SPU alone = 3134846 CK (1392 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 373950 CK, SM lane alone = 225116 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 260 / 1019 | 280 / 1036 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 50% [50%-50%] | - | 3% [3%-3%] | 371 / 2161 | 393 / 2235 | 94% | 1.41x | - | 1.02x | 1.02x | 0.91x |
| (c) GPU+SPU+DRU | fair | 47% [47%-47%] | 8% [8%-8%] | 3% [3%-3%] | 372 / 2075 | 393 / 2091 | 93% | 1.51x | 3.77x | 1.02x | 1.02x | 0.55x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 780544 CK, SM lane alone = 469537 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 262 / 994 | 282 / 1010 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 52% [52%-52%] | - | 3% [3%-3%] | 360 / 1891 | 382 / 1919 | 94% | 1.36x | - | 1.04x | 1.04x | 0.83x |
| (c) GPU+SPU+DRU | fair | 50% [50%-50%] | 9% [9%-9%] | 3% [3%-3%] | 349 / 1949 | 369 / 1967 | 93% | 1.42x | 3.33x | 1.03x | 1.03x | 0.53x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 1565232 CK, SM lane alone = 937738 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 264 / 1010 | 283 / 1027 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 56% [56%-56%] | - | 3% [3%-3%] | 332 / 1356 | 352 / 1379 | 94% | 1.27x | - | 1.05x | 1.05x | 0.70x |
| (c) GPU+SPU+DRU | fair | 46% [46%-46%] | 7% [7%-7%] | 3% [3%-3%] | 372 / 1551 | 392 / 1576 | 93% | 1.53x | 4.08x | 1.06x | 1.02x | 0.55x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 3133044 CK, SM lane alone = 1877296 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 264 / 992 | 284 / 1008 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 39% [39%-39%] | - | 3% [3%-3%] | 476 / 1550 | 498 / 1571 | 93% | 1.80x | - | 1.09x | 1.19x | 0.59x |
| (c) GPU+SPU+DRU | fair | 48% [48%-48%] | 6% [6%-6%] | 3% [3%-3%] | 365 / 1342 | 385 / 1361 | 92% | 1.49x | 4.64x | 1.08x | 1.02x | 0.63x |
