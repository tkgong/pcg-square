
### B200 (2 channels simulated), SPU alone = 778798 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 93556 CK, SM lane alone = 56228 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 351 | 42 / 373 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 11% [11%-11%] | - | 17% [17%-17%] | 837 / 5653 | 863 / 5677 | 90% | 1.27x | - | 1.03x | 1.03x | 0.92x |
| (c) GPU+SPU+DRU | fair | 11% [11%-11%] | 3% [3%-3%] | 17% [17%-17%] | 803 / 5737 | 829 / 5753 | 88% | 1.32x | 2.02x | 1.05x | 1.05x | 0.59x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 194958 CK, SM lane alone = 116940 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 29 / 355 | 45 / 377 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 17% [17%-17%] | 653 / 5569 | 677 / 5599 | 90% | 1.19x | - | 1.06x | 1.06x | 0.85x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 3% [3%-3%] | 17% [17%-17%] | 577 / 5731 | 603 / 5761 | 88% | 1.18x | 1.68x | 1.07x | 1.07x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389508 CK, SM lane alone = 233852 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 16% [16%-16%] | 420 / 5135 | 441 / 5161 | 91% | 1.14x | - | 1.10x | 1.10x | 0.74x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 4% [4%-4%] | 16% [16%-16%] | 557 / 5459 | 581 / 5483 | 89% | 1.18x | 1.30x | 1.10x | 1.10x | 0.59x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 778962 CK, SM lane alone = 467392 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 353 | 44 / 375 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 15% [15%-15%] | 270 / 4225 | 290 / 4251 | 92% | 1.09x | - | 1.16x | 1.16x | 0.58x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 16% [16%-16%] | 411 / 4865 | 433 / 4897 | 89% | 1.13x | 1.04x | 1.15x | 1.15x | 0.58x |
