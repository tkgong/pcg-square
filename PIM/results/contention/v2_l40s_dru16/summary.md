
### L40S (8 channels simulated), SPU alone = 935117 CK (415 us); standalone host-stream BW per channel: sm 73% of peak, full 73% of peak, dru 53% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112968 CK, SM lane alone = 66519 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 72% [72%-72%] | - | - | 114 / 894 | 135 / 912 | 90% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 55% [55%-55%] | - | 10% [10%-10%] | 154 / 907 | 178 / 928 | 78% | 1.32x | - | 1.16x | 1.16x | 1.03x |
| (c) GPU+SPU+DRU | fair | 51% [51%-51%] | 17% [17%-17%] | 9% [9%-9%] | 149 / 1022 | 172 / 1063 | 79% | 1.44x | 1.71x | 1.20x | 1.20x | 0.67x |
| (b) GPU+SPU | gpu | 55% [55%-55%] | - | 10% [10%-10%] | 154 / 917 | 178 / 937 | 78% | 1.32x | - | 1.16x | 1.16x | 1.03x |
| (c) GPU+SPU+DRU | gpu | 53% [53%-53%] | 14% [14%-14%] | 10% [10%-10%] | 145 / 907 | 168 / 933 | 78% | 1.39x | 2.01x | 1.13x | 1.13x | 0.63x |
| (b) GPU+SPU | pim | 3% [3%-3%] | - | 11% [11%-11%] | 3365 / 7275 | 3456 / 7275 | 80% | 24.87x | - | 1.01x | 1.14x | 1.01x |
| (c) GPU+SPU+DRU | pim | 2% [2%-2%] | 1% [1%-1%] | 11% [11%-11%] | 4852 / 7275 | 4978 / 7275 | 76% | 39.36x | 24.90x | 1.01x | 1.13x | 0.63x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234236 CK, SM lane alone = 140144 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 72% [72%-72%] | - | - | 116 / 884 | 137 / 901 | 90% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 55% [55%-55%] | - | 8% [8%-8%] | 154 / 896 | 179 / 914 | 79% | 1.31x | - | 1.33x | 1.33x | 1.06x |
| (c) GPU+SPU+DRU | fair | 49% [49%-49%] | 17% [17%-17%] | 8% [8%-8%] | 150 / 953 | 174 / 985 | 79% | 1.47x | 1.71x | 1.43x | 1.43x | 0.78x |
| (b) GPU+SPU | gpu | 55% [55%-55%] | - | 8% [8%-8%] | 155 / 898 | 179 / 919 | 78% | 1.31x | - | 1.33x | 1.33x | 1.06x |
| (c) GPU+SPU+DRU | gpu | 54% [54%-54%] | 14% [14%-14%] | 9% [9%-9%] | 144 / 853 | 167 / 880 | 78% | 1.34x | 2.02x | 1.28x | 1.28x | 0.70x |
| (b) GPU+SPU | pim | 5% [5%-5%] | - | 11% [11%-11%] | 1907 / 7275 | 1972 / 7275 | 80% | 14.26x | - | 1.01x | 1.28x | 1.02x |
| (c) GPU+SPU+DRU | pim | 3% [3%-3%] | 2% [2%-2%] | 11% [11%-11%] | 2599 / 7275 | 2681 / 7275 | 76% | 20.96x | 14.21x | 1.01x | 1.27x | 0.69x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 468754 CK, SM lane alone = 281219 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 72% [72%-72%] | - | - | 116 / 864 | 138 / 892 | 90% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 56% [56%-56%] | - | 7% [7%-7%] | 155 / 887 | 179 / 905 | 78% | 1.30x | - | 1.65x | 1.65x | 1.10x |
| (c) GPU+SPU+DRU | fair | 53% [53%-53%] | 17% [17%-17%] | 6% [6%-6%] | 139 / 875 | 162 / 894 | 79% | 1.38x | 1.69x | 1.85x | 1.85x | 0.97x |
| (b) GPU+SPU | gpu | 56% [56%-56%] | - | 7% [7%-7%] | 155 / 882 | 180 / 905 | 78% | 1.30x | - | 1.65x | 1.65x | 1.10x |
| (c) GPU+SPU+DRU | gpu | 55% [55%-55%] | 12% [12%-12%] | 7% [7%-7%] | 144 / 837 | 167 / 857 | 79% | 1.32x | 2.41x | 1.54x | 1.54x | 0.81x |
| (b) GPU+SPU | pim | 8% [8%-8%] | - | 11% [11%-11%] | 1238 / 7275 | 1292 / 7275 | 80% | 9.33x | - | 1.01x | 1.57x | 1.05x |
| (c) GPU+SPU+DRU | pim | 6% [6%-6%] | 3% [3%-3%] | 11% [11%-11%] | 1564 / 7275 | 1625 / 7275 | 76% | 12.67x | 9.24x | 1.01x | 1.54x | 0.81x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 945752 CK, SM lane alone = 561772 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 72% [72%-72%] | - | - | 118 / 843 | 139 / 875 | 90% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 43% [43%-43%] | - | 5% [5%-5%] | 206 / 1022 | 234 / 1047 | 80% | 1.68x | - | 2.18x | 2.16x | 1.08x |
| (c) GPU+SPU+DRU | fair | 40% [40%-40%] | 10% [10%-10%] | 5% [5%-5%] | 197 / 1061 | 223 / 1091 | 79% | 1.81x | 2.77x | 2.28x | 1.97x | 1.06x |
| (b) GPU+SPU | gpu | 40% [40%-40%] | - | 5% [5%-5%] | 220 / 1058 | 248 / 1082 | 78% | 1.78x | - | 2.22x | 2.20x | 1.10x |
| (c) GPU+SPU+DRU | gpu | 54% [54%-54%] | 6% [6%-6%] | 5% [5%-5%] | 136 / 837 | 158 / 859 | 79% | 1.34x | 4.44x | 2.04x | 1.76x | 0.94x |
| (b) GPU+SPU | pim | 11% [11%-11%] | - | 11% [11%-11%] | 905 / 1612 | 953 / 1640 | 80% | 6.82x | - | 1.01x | 2.13x | 1.07x |
| (c) GPU+SPU+DRU | pim | 8% [8%-8%] | 4% [4%-4%] | 11% [11%-11%] | 1046 / 4198 | 1097 / 4395 | 76% | 8.56x | 6.70x | 1.01x | 1.79x | 0.96x |
