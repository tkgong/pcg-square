
### B200 (2 channels simulated), SPU alone = 995188 CK (498 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 119496 CK, SM lane alone = 72046 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 349 | 42 / 369 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 11% [11%-11%] | - | 14% [14%-14%] | 734 / 4979 | 760 / 5011 | 91% | 1.24x | - | 1.03x | 1.03x | 0.92x |
| (c) GPU+SPU+DRU | fair | 11% [11%-11%] | 4% [4%-4%] | 14% [14%-14%] | 882 / 5425 | 907 / 5449 | 89% | 1.34x | 1.36x | 1.03x | 1.03x | 0.58x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 248868 CK, SM lane alone = 149408 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 383 | 44 / 405 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 13% [13%-13%] | 585 / 4841 | 609 / 4877 | 91% | 1.18x | - | 1.05x | 1.05x | 0.84x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 13% [13%-13%] | 596 / 5415 | 619 / 5923 | 88% | 1.20x | 1.26x | 1.05x | 1.05x | 0.58x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 497644 CK, SM lane alone = 298668 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 357 | 44 / 381 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 13% [13%-13%] | 363 / 4473 | 385 / 4497 | 91% | 1.14x | - | 1.08x | 1.08x | 0.72x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 13% [13%-13%] | 487 / 4703 | 509 / 4723 | 89% | 1.16x | 1.08x | 1.07x | 1.07x | 0.58x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 995132 CK, SM lane alone = 597791 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 355 | 44 / 377 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 13% [13%-13%] | 213 / 3895 | 232 / 3921 | 92% | 1.08x | - | 1.12x | 1.12x | 0.56x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 13% [13%-13%] | 323 / 4353 | 344 / 4395 | 88% | 1.13x | 0.91x | 1.11x | 1.11x | 0.56x |
