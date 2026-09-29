
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 30% [30%-30%] | - | 11% [11%-11%] | 625 / 4514 | 651 / 4574 | 92% | 2.34x | - | 1.06x | 1.06x | 0.95x |
| (c) GPU+SPU+DRU | fair | 30% [30%-30%] | 5% [5%-5%] | 10% [10%-10%] | 588 / 3880 | 613 / 3944 | 92% | 2.43x | 6.10x | 1.10x | 1.10x | 0.60x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 32% [32%-32%] | - | 10% [10%-10%] | 583 / 4411 | 608 / 4437 | 92% | 2.18x | - | 1.11x | 1.11x | 0.89x |
| (c) GPU+SPU+DRU | fair | 27% [27%-27%] | 4% [4%-4%] | 10% [10%-10%] | 636 / 4359 | 662 / 4381 | 91% | 2.56x | 6.53x | 1.12x | 1.12x | 0.57x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 36% [36%-36%] | - | 9% [9%-9%] | 527 / 3837 | 551 / 3864 | 93% | 1.97x | - | 1.18x | 1.18x | 0.79x |
| (c) GPU+SPU+DRU | fair | 27% [27%-27%] | 4% [4%-4%] | 10% [10%-10%] | 635 / 4073 | 660 / 4096 | 91% | 2.56x | 7.07x | 1.15x | 1.17x | 0.63x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 18% [18%-18%] | - | 9% [9%-9%] | 1031 / 5547 | 1064 / 5579 | 90% | 3.83x | - | 1.22x | 1.58x | 0.79x |
| (c) GPU+SPU+DRU | fair | 21% [21%-21%] | 5% [5%-5%] | 9% [9%-9%] | 853 / 4656 | 882 / 4677 | 90% | 3.42x | 6.25x | 1.20x | 1.20x | 0.75x |
