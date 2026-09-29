
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 41% [41%-41%] | - | 9% [9%-9%] | 455 / 1376 | 477 / 1399 | 83% | 1.70x | - | 1.20x | 1.20x | 1.07x |
| (c) GPU+SPU+DRU | gpu | 41% [41%-41%] | 6% [6%-6%] | 9% [9%-9%] | 421 / 1216 | 442 / 1230 | 83% | 1.76x | 4.75x | 1.19x | 1.19x | 0.64x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 41% [41%-41%] | - | 8% [8%-8%] | 463 / 1381 | 486 / 1397 | 83% | 1.73x | - | 1.43x | 1.43x | 1.14x |
| (c) GPU+SPU+DRU | gpu | 40% [40%-40%] | 6% [6%-6%] | 8% [8%-8%] | 431 / 1290 | 452 / 1308 | 83% | 1.73x | 5.11x | 1.39x | 1.39x | 0.71x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 41% [41%-41%] | - | 6% [6%-6%] | 453 / 1352 | 476 / 1370 | 83% | 1.71x | - | 1.85x | 1.85x | 1.23x |
| (c) GPU+SPU+DRU | gpu | 40% [40%-40%] | 4% [4%-4%] | 7% [7%-7%] | 435 / 1319 | 456 / 1338 | 82% | 1.74x | 6.74x | 1.66x | 1.43x | 0.77x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 26% [26%-26%] | - | 5% [5%-5%] | 750 / 1656 | 776 / 1677 | 82% | 2.73x | - | 2.45x | 2.45x | 1.22x |
| (c) GPU+SPU+DRU | gpu | 42% [42%-42%] | 4% [4%-4%] | 5% [5%-5%] | 416 / 1292 | 437 / 1310 | 84% | 1.67x | 6.81x | 2.32x | 1.40x | 0.87x |
