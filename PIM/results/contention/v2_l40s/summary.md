
### L40S (8 channels simulated), SPU alone = 935117 CK (415 us); standalone host-stream BW per channel: sm 73% of peak, full 73% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 112968 CK, SM lane alone = 66519 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 72% [72%-72%] | - | - | 114 / 894 | 135 / 912 | 90% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 55% [55%-55%] | - | 10% [10%-10%] | 154 / 907 | 178 / 928 | 78% | 1.32x | - | 1.16x | 1.16x | 1.03x |
| (c) GPU+SPU+DRU | fair | 53% [53%-53%] | 7% [7%-7%] | 9% [9%-9%] | 142 / 892 | 165 / 917 | 79% | 1.38x | 4.14x | 1.18x | 1.18x | 0.64x |
| (b) GPU+SPU | gpu | 55% [55%-55%] | - | 10% [10%-10%] | 154 / 917 | 178 / 937 | 78% | 1.32x | - | 1.16x | 1.16x | 1.03x |
| (c) GPU+SPU+DRU | gpu | 53% [53%-53%] | 6% [6%-6%] | 10% [10%-10%] | 141 / 873 | 164 / 894 | 79% | 1.38x | 4.57x | 1.17x | 1.17x | 0.63x |
| (b) GPU+SPU | pim | 3% [3%-3%] | - | 11% [11%-11%] | 3365 / 7275 | 3456 / 7275 | 80% | 24.87x | - | 1.01x | 1.14x | 1.01x |
| (c) GPU+SPU+DRU | pim | 2% [2%-2%] | 1% [1%-1%] | 11% [11%-11%] | 4823 / 7275 | 4948 / 7275 | 79% | 39.14x | 26.27x | 1.01x | 1.17x | 0.63x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 234236 CK, SM lane alone = 140144 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 72% [72%-72%] | - | - | 116 / 884 | 137 / 901 | 90% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 55% [55%-55%] | - | 8% [8%-8%] | 154 / 896 | 179 / 914 | 79% | 1.31x | - | 1.33x | 1.33x | 1.06x |
| (c) GPU+SPU+DRU | fair | 55% [55%-55%] | 6% [6%-6%] | 8% [8%-8%] | 142 / 852 | 165 / 873 | 79% | 1.32x | 4.70x | 1.42x | 1.42x | 0.72x |
| (b) GPU+SPU | gpu | 55% [55%-55%] | - | 8% [8%-8%] | 155 / 898 | 179 / 919 | 78% | 1.31x | - | 1.33x | 1.33x | 1.06x |
| (c) GPU+SPU+DRU | gpu | 54% [54%-54%] | 6% [6%-6%] | 8% [8%-8%] | 145 / 859 | 168 / 876 | 79% | 1.34x | 4.65x | 1.33x | 1.33x | 0.67x |
| (b) GPU+SPU | pim | 5% [5%-5%] | - | 11% [11%-11%] | 1907 / 7275 | 1972 / 7275 | 80% | 14.26x | - | 1.01x | 1.28x | 1.02x |
| (c) GPU+SPU+DRU | pim | 3% [3%-3%] | 2% [2%-2%] | 11% [11%-11%] | 2574 / 7275 | 2655 / 7275 | 79% | 20.76x | 15.56x | 1.01x | 1.36x | 0.69x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 468754 CK, SM lane alone = 281219 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 72% [72%-72%] | - | - | 116 / 864 | 138 / 892 | 90% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 56% [56%-56%] | - | 7% [7%-7%] | 155 / 887 | 179 / 905 | 78% | 1.30x | - | 1.65x | 1.65x | 1.10x |
| (c) GPU+SPU+DRU | fair | 55% [55%-55%] | 6% [6%-6%] | 6% [6%-6%] | 144 / 852 | 167 / 873 | 79% | 1.31x | 5.18x | 1.97x | 1.67x | 0.91x |
| (b) GPU+SPU | gpu | 56% [56%-56%] | - | 7% [7%-7%] | 155 / 882 | 180 / 905 | 78% | 1.30x | - | 1.65x | 1.65x | 1.10x |
| (c) GPU+SPU+DRU | gpu | 55% [55%-55%] | 5% [5%-5%] | 7% [7%-7%] | 144 / 841 | 167 / 864 | 79% | 1.32x | 6.03x | 1.56x | 1.33x | 0.72x |
| (b) GPU+SPU | pim | 8% [8%-8%] | - | 11% [11%-11%] | 1238 / 7275 | 1292 / 7275 | 80% | 9.33x | - | 1.01x | 1.57x | 1.05x |
| (c) GPU+SPU+DRU | pim | 6% [6%-6%] | 3% [3%-3%] | 11% [11%-11%] | 1536 / 7275 | 1596 / 7275 | 79% | 12.44x | 10.59x | 1.01x | 1.46x | 0.79x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 945752 CK, SM lane alone = 561772 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 72% [72%-72%] | - | - | 118 / 843 | 139 / 875 | 90% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 43% [43%-43%] | - | 5% [5%-5%] | 206 / 1022 | 234 / 1047 | 80% | 1.68x | - | 2.18x | 2.16x | 1.08x |
| (c) GPU+SPU+DRU | fair | 54% [54%-54%] | 5% [5%-5%] | 4% [4%-4%] | 144 / 844 | 167 / 866 | 79% | 1.34x | 5.33x | 2.70x | 1.60x | 1.00x |
| (b) GPU+SPU | gpu | 40% [40%-40%] | - | 5% [5%-5%] | 220 / 1058 | 248 / 1082 | 78% | 1.78x | - | 2.22x | 2.20x | 1.10x |
| (c) GPU+SPU+DRU | gpu | 55% [55%-55%] | 4% [4%-4%] | 5% [5%-5%] | 144 / 830 | 167 / 852 | 79% | 1.31x | 6.46x | 2.09x | 1.24x | 0.78x |
| (b) GPU+SPU | pim | 11% [11%-11%] | - | 11% [11%-11%] | 905 / 1612 | 953 / 1640 | 80% | 6.82x | - | 1.01x | 2.13x | 1.07x |
| (c) GPU+SPU+DRU | pim | 9% [9%-9%] | 4% [4%-4%] | 11% [11%-11%] | 1019 / 6560 | 1068 / 6608 | 79% | 8.33x | 7.95x | 1.01x | 1.44x | 0.90x |
