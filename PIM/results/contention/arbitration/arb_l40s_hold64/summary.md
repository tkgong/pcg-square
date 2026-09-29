
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 27% [27%-27%] | - | 10% [10%-10%] | 708 / 4400 | 734 / 4439 | 90% | 2.64x | - | 1.08x | 1.08x | 0.96x |
| (c) GPU+SPU+DRU | fair | 25% [25%-25%] | 4% [4%-4%] | 10% [10%-10%] | 686 / 4002 | 711 / 4020 | 90% | 2.86x | 6.38x | 1.13x | 1.13x | 0.61x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 31% [31%-31%] | - | 10% [10%-10%] | 608 / 3958 | 631 / 3976 | 91% | 2.29x | - | 1.13x | 1.13x | 0.91x |
| (c) GPU+SPU+DRU | fair | 24% [24%-24%] | 4% [4%-4%] | 9% [9%-9%] | 712 / 4324 | 737 / 4423 | 89% | 2.88x | 6.75x | 1.19x | 1.19x | 0.61x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 23% [23%-23%] | - | 9% [9%-9%] | 828 / 4309 | 854 / 4341 | 90% | 3.11x | - | 1.27x | 1.27x | 0.85x |
| (c) GPU+SPU+DRU | fair | 24% [24%-24%] | 4% [4%-4%] | 9% [9%-9%] | 736 / 4396 | 761 / 4414 | 89% | 2.94x | 7.77x | 1.23x | 1.21x | 0.65x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 15% [15%-15%] | - | 9% [9%-9%] | 1300 / 7275 | 1336 / 7275 | 89% | 4.84x | - | 1.32x | 1.80x | 0.90x |
| (c) GPU+SPU+DRU | fair | 15% [15%-15%] | 4% [4%-4%] | 9% [9%-9%] | 1137 / 5697 | 1170 / 5744 | 87% | 4.59x | 6.66x | 1.29x | 1.25x | 0.78x |
