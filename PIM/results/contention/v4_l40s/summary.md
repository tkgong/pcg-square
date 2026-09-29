
### L40S (2 channels simulated), SPU alone = 935111 CK (415 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112882 CK, SM lane alone = 66147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 257 / 1066 | 276 / 1082 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 34% [34%-34%] | - | 10% [10%-10%] | 558 / 3549 | 581 / 3569 | 91% | 2.08x | - | 1.07x | 1.07x | 0.95x |
| (c) GPU+SPU+DRU | fair | 31% [31%-31%] | 5% [5%-5%] | 10% [10%-10%] | 562 / 4194 | 586 / 4210 | 91% | 2.34x | 5.67x | 1.11x | 1.11x | 0.60x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234130 CK, SM lane alone = 141134 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1052 | 281 / 1068 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 37% [37%-37%] | - | 10% [10%-10%] | 513 / 3560 | 535 / 3601 | 92% | 1.91x | - | 1.12x | 1.12x | 0.90x |
| (c) GPU+SPU+DRU | fair | 29% [29%-29%] | 5% [5%-5%] | 10% [10%-10%] | 617 / 4203 | 641 / 4228 | 90% | 2.45x | 5.58x | 1.16x | 1.16x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 467184 CK, SM lane alone = 281850 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 261 / 1040 | 281 / 1056 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 39% [39%-39%] | - | 9% [9%-9%] | 488 / 3143 | 510 / 3162 | 92% | 1.82x | - | 1.21x | 1.21x | 0.81x |
| (c) GPU+SPU+DRU | fair | 27% [27%-27%] | 4% [4%-4%] | 9% [9%-9%] | 648 / 4665 | 672 / 4686 | 89% | 2.56x | 6.64x | 1.20x | 1.14x | 0.62x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 935990 CK, SM lane alone = 560344 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 263 / 1001 | 283 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 18% [18%-18%] | - | 9% [9%-9%] | 1088 / 5519 | 1120 / 5538 | 88% | 3.97x | - | 1.31x | 1.61x | 0.81x |
| (c) GPU+SPU+DRU | fair | 20% [20%-20%] | 5% [5%-5%] | 9% [9%-9%] | 882 / 5086 | 910 / 5139 | 87% | 3.52x | 6.25x | 1.27x | 1.19x | 0.74x |
