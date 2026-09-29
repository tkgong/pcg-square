
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 34% [34%-34%] | - | 10% [10%-10%] | 557 / 3549 | 581 / 3569 | 91% | 2.08x | - | 1.07x | 1.07x | 0.95x |
| (c) GPU+SPU+DRU | fair | 33% [33%-33%] | 5% [5%-5%] | 10% [10%-10%] | 510 / 3604 | 533 / 3718 | 91% | 2.17x | 5.72x | 1.11x | 1.11x | 0.60x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 37% [37%-37%] | - | 10% [10%-10%] | 518 / 3508 | 541 / 3539 | 92% | 1.93x | - | 1.12x | 1.12x | 0.90x |
| (c) GPU+SPU+DRU | fair | 28% [28%-28%] | 5% [5%-5%] | 10% [10%-10%] | 626 / 4262 | 652 / 4286 | 90% | 2.49x | 5.37x | 1.16x | 1.16x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 39% [39%-39%] | - | 9% [9%-9%] | 488 / 3230 | 510 / 3350 | 92% | 1.82x | - | 1.22x | 1.22x | 0.81x |
| (c) GPU+SPU+DRU | fair | 26% [26%-26%] | 4% [4%-4%] | 9% [9%-9%] | 676 / 4677 | 702 / 4707 | 89% | 2.70x | 6.62x | 1.21x | 1.14x | 0.61x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 18% [18%-18%] | - | 9% [9%-9%] | 1091 / 5546 | 1126 / 5581 | 88% | 4.00x | - | 1.30x | 1.62x | 0.81x |
| (c) GPU+SPU+DRU | fair | 20% [20%-20%] | 5% [5%-5%] | 9% [9%-9%] | 890 / 4975 | 919 / 5006 | 88% | 3.57x | 6.26x | 1.28x | 1.20x | 0.75x |
