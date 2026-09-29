
### B200 (2 channels simulated), SPU alone = 2739228 CK (1370 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 328862 CK, SM lane alone = 197548 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 349 | 43 / 369 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 5% [5%-5%] | 270 / 4425 | 291 / 4475 | 93% | 1.07x | - | 1.01x | 1.01x | 0.90x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 5% [5%-5%] | 327 / 4463 | 348 / 4483 | 90% | 1.12x | 0.95x | 1.01x | 1.01x | 0.57x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 684834 CK, SM lane alone = 411018 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 349 | 43 / 369 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 5% [5%-5%] | 211 / 3911 | 231 / 3935 | 93% | 1.06x | - | 1.02x | 1.02x | 0.81x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 5% [5%-5%] | 277 / 4373 | 297 / 4393 | 89% | 1.09x | 0.90x | 1.01x | 1.01x | 0.56x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 1369644 CK, SM lane alone = 821907 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 351 | 43 / 369 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 5% [5%-5%] | 135 / 3345 | 153 / 3379 | 94% | 1.05x | - | 1.02x | 1.02x | 0.68x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 7% [7%-7%] | 5% [5%-5%] | 205 / 3803 | 225 / 3823 | 89% | 1.08x | 0.83x | 1.02x | 1.02x | 0.55x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 2738984 CK, SM lane alone = 1643807 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 349 | 43 / 369 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 5% [5%-5%] | 102 / 2321 | 119 / 2369 | 95% | 1.03x | - | 1.02x | 1.02x | 0.51x |
| (c) GPU+SPU+DRU | fair | 14% [14%-14%] | 7% [7%-7%] | 5% [5%-5%] | 134 / 3109 | 153 / 3137 | 88% | 1.05x | 0.79x | 1.02x | 1.02x | 0.52x |
