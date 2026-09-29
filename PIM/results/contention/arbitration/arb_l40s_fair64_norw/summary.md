
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 40% [40%-40%] | - | 9% [9%-9%] | 461 / 1415 | 483 / 1432 | 82% | 1.74x | - | 1.20x | 1.20x | 1.07x |
| (c) GPU+SPU+DRU | fair | 39% [39%-39%] | 6% [6%-6%] | 9% [9%-9%] | 432 / 1296 | 454 / 1313 | 82% | 1.83x | 4.35x | 1.20x | 1.20x | 0.65x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 40% [40%-40%] | - | 8% [8%-8%] | 473 / 1413 | 495 / 1431 | 82% | 1.76x | - | 1.43x | 1.43x | 1.14x |
| (c) GPU+SPU+DRU | fair | 39% [39%-39%] | 5% [5%-5%] | 8% [8%-8%] | 445 / 1468 | 467 / 1496 | 83% | 1.80x | 5.29x | 1.48x | 1.48x | 0.76x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 41% [41%-41%] | - | 6% [6%-6%] | 462 / 1374 | 485 / 1393 | 82% | 1.74x | - | 1.84x | 1.84x | 1.23x |
| (c) GPU+SPU+DRU | fair | 40% [40%-40%] | 5% [5%-5%] | 6% [6%-6%] | 436 / 1447 | 458 / 1464 | 83% | 1.76x | 5.89x | 2.03x | 1.74x | 0.94x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 26% [26%-26%] | - | 5% [5%-5%] | 729 / 1551 | 756 / 1571 | 82% | 2.66x | - | 2.40x | 2.40x | 1.20x |
| (c) GPU+SPU+DRU | fair | 35% [35%-35%] | 5% [5%-5%] | 4% [4%-4%] | 501 / 1656 | 524 / 1681 | 83% | 2.03x | 5.96x | 2.87x | 1.73x | 1.08x |
