
### B200 (16 channels simulated), SPU alone = 778952 CK (389 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 94625 CK, SM lane alone = 56314 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 375 | 46 / 405 | 86% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 17% [17%-17%] | 169 / 2083 | 191 / 2117 | 75% | 1.13x | - | 1.05x | 1.05x | 0.94x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 7% [7%-7%] | 17% [17%-17%] | 182 / 4737 | 205 / 4775 | 74% | 1.16x | 0.83x | 1.05x | 1.05x | 0.59x |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 17% [17%-17%] | 156 / 2291 | 179 / 2313 | 73% | 1.09x | - | 1.05x | 1.05x | 0.94x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 6% [6%-6%] | 17% [17%-17%] | 174 / 2379 | 198 / 2409 | 71% | 1.11x | 0.94x | 1.05x | 1.05x | 0.60x |
| (b) GPU+SPU | pim | 1% [1%-1%] | - | 18% [18%-18%] | 16481 / 8193 | 16722 / 8193 | 83% | 21.50x | - | 1.01x | 1.04x | 0.93x |
| (c) GPU+SPU+DRU | pim | 0% [0%-0%] | 0% [0%-0%] | 18% [18%-18%] | 22897 / 8193 | 23202 / 8193 | 76% | 32.71x | 21.50x | 1.01x | 1.03x | 0.58x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 195020 CK, SM lane alone = 117156 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 33 / 385 | 51 / 415 | 85% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 16% [16%-16%] | 122 / 1623 | 143 / 1653 | 75% | 1.04x | - | 1.10x | 1.10x | 0.88x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 7% [7%-7%] | 17% [17%-17%] | 110 / 1189 | 131 / 1209 | 75% | 1.07x | 0.80x | 1.09x | 1.09x | 0.61x |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 16% [16%-16%] | 131 / 2045 | 153 / 2071 | 74% | 1.06x | - | 1.10x | 1.10x | 0.88x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 6% [6%-6%] | 16% [16%-16%] | 129 / 1727 | 152 / 1751 | 72% | 1.07x | 0.89x | 1.09x | 1.09x | 0.61x |
| (b) GPU+SPU | pim | 1% [1%-1%] | - | 18% [18%-18%] | 8003 / 8193 | 8131 / 8193 | 85% | 10.96x | - | 1.01x | 1.07x | 0.86x |
| (c) GPU+SPU+DRU | pim | 1% [1%-1%] | 1% [1%-1%] | 18% [18%-18%] | 12293 / 8193 | 12478 / 8193 | 80% | 17.55x | 10.92x | 1.01x | 1.06x | 0.59x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 389645 CK, SM lane alone = 233794 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 31 / 387 | 50 / 411 | 85% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 15% [15%-15%] | 111 / 1543 | 133 / 1565 | 76% | 1.03x | - | 1.19x | 1.19x | 0.79x |
| (c) GPU+SPU+DRU | fair | 14% [14%-14%] | 8% [8%-8%] | 16% [16%-16%] | 109 / 1273 | 131 / 1295 | 75% | 1.04x | 0.75x | 1.16x | 1.16x | 0.63x |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 15% [15%-15%] | 124 / 2105 | 146 / 2131 | 74% | 1.07x | - | 1.20x | 1.20x | 0.80x |
| (c) GPU+SPU+DRU | gpu | 14% [14%-14%] | 7% [7%-7%] | 15% [15%-15%] | 107 / 967 | 129 / 997 | 73% | 1.03x | 0.80x | 1.16x | 1.16x | 0.63x |
| (b) GPU+SPU | pim | 2% [2%-2%] | - | 18% [18%-18%] | 4079 / 8193 | 4155 / 8193 | 85% | 5.99x | - | 1.01x | 1.13x | 0.76x |
| (c) GPU+SPU+DRU | pim | 2% [2%-2%] | 1% [1%-1%] | 18% [18%-18%] | 6258 / 8193 | 6364 / 8193 | 81% | 9.30x | 5.93x | 1.01x | 1.12x | 0.61x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 781582 CK, SM lane alone = 467558 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 31 / 391 | 50 / 415 | 85% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 13% [13%-13%] | 132 / 2305 | 154 / 2357 | 76% | 1.07x | - | 1.39x | 1.38x | 0.69x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 7% [7%-7%] | 14% [14%-14%] | 149 / 3011 | 171 / 3041 | 74% | 1.09x | 0.76x | 1.32x | 1.32x | 0.69x |
| (b) GPU+SPU | gpu | 13% [13%-13%] | - | 13% [13%-13%] | 151 / 2405 | 174 / 2433 | 74% | 1.09x | - | 1.42x | 1.42x | 0.71x |
| (c) GPU+SPU+DRU | gpu | 13% [13%-13%] | 7% [7%-7%] | 14% [14%-14%] | 135 / 2265 | 158 / 2279 | 72% | 1.09x | 0.79x | 1.32x | 1.32x | 0.69x |
| (b) GPU+SPU | pim | 4% [4%-4%] | - | 18% [18%-18%] | 2126 / 8193 | 2177 / 8193 | 86% | 3.48x | - | 1.01x | 1.26x | 0.63x |
| (c) GPU+SPU+DRU | pim | 3% [3%-3%] | 2% [2%-2%] | 18% [18%-18%] | 3238 / 8193 | 3304 / 8193 | 82% | 5.15x | 3.43x | 1.01x | 1.24x | 0.65x |
