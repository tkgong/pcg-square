
### L40S (2 channels simulated), SPU alone = 1642408 CK (729 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 199276 CK, SM lane alone = 119329 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 264 / 1036 | 284 / 1052 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 42% [42%-42%] | - | 6% [6%-6%] | 437 / 3334 | 459 / 3352 | 93% | 1.66x | - | 1.04x | 1.04x | 0.93x |
| (c) GPU+SPU+DRU | fair | 49% [49%-49%] | 7% [7%-7%] | 6% [6%-6%] | 347 / 1574 | 367 / 1617 | 93% | 1.42x | 4.18x | 1.09x | 1.09x | 0.59x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 410436 CK, SM lane alone = 249315 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 262 / 992 | 282 / 1008 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 46% [46%-46%] | - | 6% [6%-6%] | 405 / 2365 | 426 / 2383 | 93% | 1.52x | - | 1.08x | 1.08x | 0.87x |
| (c) GPU+SPU+DRU | fair | 51% [51%-51%] | 8% [8%-8%] | 5% [5%-5%] | 344 / 1280 | 364 / 1319 | 93% | 1.38x | 3.64x | 1.20x | 1.20x | 0.61x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 820338 CK, SM lane alone = 493069 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 263 / 969 | 283 / 983 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 51% [51%-51%] | - | 5% [5%-5%] | 369 / 1715 | 390 / 1734 | 94% | 1.39x | - | 1.16x | 1.16x | 0.78x |
| (c) GPU+SPU+DRU | fair | 43% [43%-43%] | 6% [6%-6%] | 5% [5%-5%] | 400 / 1551 | 421 / 1573 | 92% | 1.62x | 4.67x | 1.26x | 1.09x | 0.58x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 1640275 CK, SM lane alone = 984668 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 264 / 971 | 283 / 988 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 29% [29%-29%] | - | 5% [5%-5%] | 653 / 2722 | 677 / 2755 | 92% | 2.45x | - | 1.32x | 1.35x | 0.67x |
| (c) GPU+SPU+DRU | fair | 40% [40%-40%] | 6% [6%-6%] | 5% [5%-5%] | 431 / 1846 | 452 / 1878 | 92% | 1.75x | 5.09x | 1.31x | 1.06x | 0.66x |
