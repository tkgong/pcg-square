
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 34% [34%-34%] | - | 10% [10%-10%] | 560 / 3549 | 584 / 3569 | 91% | 2.08x | - | 1.07x | 1.07x | 0.95x |
| (c) GPU+SPU+DRU | gpu | 31% [31%-31%] | 5% [5%-5%] | 10% [10%-10%] | 545 / 4306 | 570 / 4329 | 91% | 2.28x | 5.72x | 1.11x | 1.11x | 0.60x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 37% [37%-37%] | - | 10% [10%-10%] | 514 / 3560 | 537 / 3638 | 92% | 1.91x | - | 1.13x | 1.13x | 0.90x |
| (c) GPU+SPU+DRU | gpu | 32% [32%-32%] | 5% [5%-5%] | 10% [10%-10%] | 554 / 4169 | 579 / 4196 | 90% | 2.21x | 5.83x | 1.16x | 1.16x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 38% [38%-38%] | - | 9% [9%-9%] | 499 / 3368 | 523 / 3407 | 92% | 1.87x | - | 1.22x | 1.22x | 0.81x |
| (c) GPU+SPU+DRU | gpu | 32% [32%-32%] | 4% [4%-4%] | 9% [9%-9%] | 551 / 3963 | 575 / 4100 | 90% | 2.19x | 6.96x | 1.21x | 1.16x | 0.63x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | gpu | 18% [18%-18%] | - | 9% [9%-9%] | 1094 / 5309 | 1129 / 5348 | 88% | 4.00x | - | 1.31x | 1.63x | 0.81x |
| (c) GPU+SPU+DRU | gpu | 22% [22%-22%] | 4% [4%-4%] | 9% [9%-9%] | 819 / 5089 | 849 / 5121 | 87% | 3.28x | 6.29x | 1.28x | 1.19x | 0.74x |
