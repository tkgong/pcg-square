
### B200 (2 channels simulated), SPU alone = 1426731 CK (713 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 171354 CK, SM lane alone = 103589 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 24 / 337 | 40 / 351 | 96% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 12% [12%-12%] | - | 10% [10%-10%] | 561 / 5357 | 585 / 5413 | 92% | 1.16x | - | 1.02x | 1.02x | 0.91x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 10% [10%-10%] | 589 / 4861 | 614 / 4889 | 90% | 1.19x | 1.20x | 1.02x | 1.02x | 0.57x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 356780 CK, SM lane alone = 214594 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 26 / 347 | 42 / 365 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 9% [9%-9%] | 433 / 4581 | 455 / 4617 | 92% | 1.13x | - | 1.03x | 1.03x | 0.83x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 10% [10%-10%] | 461 / 4587 | 483 / 4607 | 90% | 1.15x | 1.11x | 1.03x | 1.03x | 0.57x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 713863 CK, SM lane alone = 428147 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 349 | 43 / 367 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 9% [9%-9%] | 257 / 4319 | 277 / 4359 | 93% | 1.11x | - | 1.05x | 1.05x | 0.70x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 9% [9%-9%] | 352 / 4253 | 373 / 4277 | 89% | 1.13x | 0.96x | 1.05x | 1.05x | 0.56x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 1426644 CK, SM lane alone = 856039 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 351 | 43 / 369 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 9% [9%-9%] | 158 / 3525 | 177 / 3553 | 93% | 1.05x | - | 1.08x | 1.08x | 0.54x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 7% [7%-7%] | 9% [9%-9%] | 226 / 3923 | 246 / 3955 | 89% | 1.09x | 0.85x | 1.07x | 1.07x | 0.54x |
