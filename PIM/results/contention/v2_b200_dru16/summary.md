
### B200 (16 channels simulated), SPU alone = 778952 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 66% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 94625 CK, SM lane alone = 56314 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 375 | 46 / 405 | 86% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 17% [17%-17%] | 169 / 2083 | 191 / 2117 | 75% | 1.13x | - | 1.05x | 1.05x | 0.94x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 24% [24%-24%] | 17% [17%-17%] | 240 / 2139 | 264 / 2169 | 74% | 1.16x | 0.69x | 1.04x | 1.04x | 0.59x |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 17% [17%-17%] | 156 / 2291 | 179 / 2313 | 73% | 1.09x | - | 1.05x | 1.05x | 0.94x |
| (c) GPU+SPU+DRU | gpu | 12% [12%-12%] | 14% [14%-14%] | 17% [17%-17%] | 209 / 2409 | 234 / 2425 | 69% | 1.14x | 0.69x | 1.04x | 1.04x | 0.59x |
| (b) GPU+SPU | pim | 1% [1%-1%] | - | 18% [18%-18%] | 16481 / 8193 | 16722 / 8193 | 83% | 21.50x | - | 1.01x | 1.04x | 0.93x |
| (c) GPU+SPU+DRU | pim | 0% [0%-0%] | 0% [0%-0%] | 18% [18%-18%] | 20500 / 8193 | 20754 / 8193 | 72% | 29.01x | 21.37x | 1.01x | 1.03x | 0.58x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 195020 CK, SM lane alone = 117156 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 33 / 385 | 51 / 415 | 85% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 16% [16%-16%] | 122 / 1623 | 143 / 1653 | 75% | 1.04x | - | 1.10x | 1.10x | 0.88x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 26% [26%-26%] | 17% [17%-17%] | 206 / 1787 | 229 / 1827 | 75% | 1.10x | 0.66x | 1.08x | 1.08x | 0.61x |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 16% [16%-16%] | 131 / 2045 | 153 / 2071 | 74% | 1.06x | - | 1.10x | 1.10x | 0.88x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 21% [21%-21%] | 17% [17%-17%] | 169 / 2007 | 194 / 2023 | 71% | 1.09x | 0.65x | 1.08x | 1.08x | 0.60x |
| (b) GPU+SPU | pim | 1% [1%-1%] | - | 18% [18%-18%] | 8003 / 8193 | 8131 / 8193 | 85% | 10.96x | - | 1.01x | 1.07x | 0.86x |
| (c) GPU+SPU+DRU | pim | 1% [1%-1%] | 1% [1%-1%] | 18% [18%-18%] | 12411 / 8193 | 12550 / 8193 | 75% | 17.62x | 10.74x | 1.01x | 1.05x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389645 CK, SM lane alone = 233794 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 31 / 387 | 50 / 411 | 85% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 15% [15%-15%] | 111 / 1543 | 133 / 1565 | 76% | 1.03x | - | 1.19x | 1.19x | 0.79x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 27% [27%-27%] | 15% [15%-15%] | 171 / 1789 | 194 / 1815 | 76% | 1.08x | 0.65x | 1.17x | 1.17x | 0.64x |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 15% [15%-15%] | 124 / 2105 | 146 / 2131 | 74% | 1.07x | - | 1.20x | 1.20x | 0.80x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 18% [18%-18%] | 16% [16%-16%] | 146 / 1713 | 170 / 1731 | 72% | 1.06x | 0.63x | 1.15x | 1.15x | 0.63x |
| (b) GPU+SPU | pim | 2% [2%-2%] | - | 18% [18%-18%] | 4079 / 8193 | 4155 / 8193 | 85% | 5.99x | - | 1.01x | 1.13x | 0.76x |
| (c) GPU+SPU+DRU | pim | 2% [2%-2%] | 1% [1%-1%] | 18% [18%-18%] | 6348 / 8193 | 6434 / 8193 | 75% | 9.35x | 5.75x | 1.01x | 1.10x | 0.60x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 781582 CK, SM lane alone = 467558 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 31 / 391 | 50 / 415 | 85% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 13% [13%-13%] | 132 / 2305 | 154 / 2357 | 76% | 1.07x | - | 1.39x | 1.38x | 0.69x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 27% [27%-27%] | 13% [13%-13%] | 195 / 3289 | 218 / 3319 | 75% | 1.11x | 0.67x | 1.37x | 1.37x | 0.72x |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 13% [13%-13%] | 151 / 2405 | 174 / 2433 | 74% | 1.09x | - | 1.42x | 1.42x | 0.71x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 14% [14%-14%] | 14% [14%-14%] | 155 / 2017 | 179 / 2041 | 72% | 1.07x | 0.64x | 1.33x | 1.33x | 0.70x |
| (b) GPU+SPU | pim | 4% [4%-4%] | - | 18% [18%-18%] | 2126 / 8193 | 2177 / 8193 | 86% | 3.48x | - | 1.01x | 1.26x | 0.63x |
| (c) GPU+SPU+DRU | pim | 3% [3%-3%] | 2% [2%-2%] | 18% [18%-18%] | 3313 / 8193 | 3372 / 8193 | 75% | 5.19x | 3.23x | 1.01x | 1.19x | 0.63x |
