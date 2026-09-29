
### B200 (2 channels simulated), SPU alone = 1881228 CK (941 us); standalone host-stream BW per channel: sm 14% of peak, full 14% of peak, dru 14% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 226178 CK, SM lane alone = 135731 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 28 / 357 | 45 / 379 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 7% [7%-7%] | 406 / 4687 | 429 / 4715 | 93% | 1.11x | - | 1.01x | 1.01x | 0.90x |
| (c) GPU+SPU+DRU | fair | 12% [12%-12%] | 5% [5%-5%] | 7% [7%-7%] | 454 / 4479 | 476 / 4509 | 90% | 1.16x | 1.08x | 1.01x | 1.01x | 0.57x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 470770 CK, SM lane alone = 282781 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 351 | 44 / 371 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 7% [7%-7%] | 307 / 4299 | 328 / 4321 | 93% | 1.09x | - | 1.02x | 1.02x | 0.82x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 7% [7%-7%] | 348 / 4511 | 369 / 4537 | 90% | 1.11x | 1.00x | 1.02x | 1.02x | 0.57x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 941284 CK, SM lane alone = 564701 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 351 | 43 / 373 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 13% [13%-13%] | - | 7% [7%-7%] | 188 / 3731 | 208 / 3761 | 93% | 1.08x | - | 1.03x | 1.03x | 0.69x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 6% [6%-6%] | 7% [7%-7%] | 274 / 4257 | 294 / 4281 | 89% | 1.11x | 0.92x | 1.03x | 1.03x | 0.55x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 1881086 CK, SM lane alone = 1129182 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 14% [14%-14%] | - | - | 27 / 351 | 43 / 371 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 14% [14%-14%] | - | 7% [7%-7%] | 139 / 3515 | 158 / 3543 | 94% | 1.05x | - | 1.03x | 1.03x | 0.52x |
| (c) GPU+SPU+DRU | fair | 13% [13%-13%] | 7% [7%-7%] | 7% [7%-7%] | 175 / 3713 | 195 / 3741 | 89% | 1.07x | 0.83x | 1.04x | 1.04x | 0.52x |
