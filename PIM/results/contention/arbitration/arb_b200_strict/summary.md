
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | pim | 1% [1%-1%] | - | 18% [18%-18%] | 33394 / 8193 | 33537 / 8193 | 88% | 21.81x | - | 1.01x | 1.04x | 0.93x |
| (c) GPU+SPU+DRU | pim | 0% [0%-0%] | 0% [0%-0%] | 18% [18%-18%] | 42054 / 8193 | 42158 / 8193 | 81% | 31.90x | 21.97x | 1.01x | 1.04x | 0.58x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | pim | 1% [1%-1%] | - | 18% [18%-18%] | 16086 / 8193 | 16164 / 8193 | 91% | 10.98x | - | 1.01x | 1.07x | 0.86x |
| (c) GPU+SPU+DRU | pim | 1% [1%-1%] | 1% [1%-1%] | 18% [18%-18%] | 24936 / 8193 | 25005 / 8193 | 86% | 17.67x | 11.11x | 1.01x | 1.07x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | pim | 2% [2%-2%] | - | 18% [18%-18%] | 8105 / 8193 | 8153 / 8193 | 92% | 6.00x | - | 1.01x | 1.13x | 0.76x |
| (c) GPU+SPU+DRU | pim | 2% [2%-2%] | 1% [1%-1%] | 18% [18%-18%] | 12603 / 8193 | 12649 / 8193 | 88% | 9.34x | 6.06x | 1.01x | 1.13x | 0.61x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | pim | 4% [4%-4%] | - | 18% [18%-18%] | 4117 / 8193 | 4151 / 8193 | 93% | 3.50x | - | 1.01x | 1.26x | 0.63x |
| (c) GPU+SPU+DRU | pim | 3% [3%-3%] | 2% [2%-2%] | 18% [18%-18%] | 6442 / 8193 | 6475 / 8193 | 89% | 5.17x | 3.53x | 1.01x | 1.26x | 0.64x |
