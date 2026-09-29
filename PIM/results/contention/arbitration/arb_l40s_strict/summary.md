
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | pim | 3% [3%-3%] | - | 11% [11%-11%] | 6947 / 7275 | 7002 / 7275 | 89% | 24.52x | - | 1.01x | 1.12x | 1.00x |
| (c) GPU+SPU+DRU | pim | 2% [2%-2%] | 1% [1%-1%] | 11% [11%-11%] | 10103 / 7275 | 10175 / 7275 | 85% | 39.36x | 26.01x | 1.01x | 1.16x | 0.63x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | pim | 5% [5%-5%] | - | 11% [11%-11%] | 3923 / 7275 | 3964 / 7275 | 90% | 13.94x | - | 1.01x | 1.25x | 1.00x |
| (c) GPU+SPU+DRU | pim | 3% [3%-3%] | 2% [2%-2%] | 11% [11%-11%] | 5390 / 7275 | 5440 / 7275 | 85% | 20.54x | 15.48x | 1.01x | 1.35x | 0.69x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | pim | 8% [8%-8%] | - | 11% [11%-11%] | 2518 / 7275 | 2553 / 7275 | 90% | 9.00x | - | 1.01x | 1.51x | 1.00x |
| (c) GPU+SPU+DRU | pim | 6% [6%-6%] | 3% [3%-3%] | 11% [11%-11%] | 3209 / 7275 | 3249 / 7275 | 85% | 12.28x | 10.52x | 1.01x | 1.46x | 0.79x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | pim | 11% [11%-11%] | - | 11% [11%-11%] | 1811 / 4963 | 1843 / 4981 | 90% | 6.48x | - | 1.01x | 2.00x | 1.00x |
| (c) GPU+SPU+DRU | pim | 9% [9%-9%] | 4% [4%-4%] | 11% [11%-11%] | 2117 / 7275 | 2152 / 7275 | 85% | 8.18x | 8.00x | 1.01x | 1.45x | 0.90x |
