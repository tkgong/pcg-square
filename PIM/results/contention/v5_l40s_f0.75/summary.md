
### L40S (2 channels simulated), SPU alone = 2163286 CK (960 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 259994 CK, SM lane alone = 155712 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 261 / 1068 | 281 / 1084 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 45% [45%-45%] | - | 5% [5%-5%] | 411 / 2356 | 433 / 2372 | 94% | 1.55x | - | 1.03x | 1.03x | 0.92x |
| (c) GPU+SPU+DRU | fair | 46% [46%-46%] | 7% [7%-7%] | 5% [5%-5%] | 373 / 1986 | 394 / 2004 | 93% | 1.53x | 4.12x | 1.04x | 1.04x | 0.56x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 540556 CK, SM lane alone = 322983 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 263 / 1049 | 283 / 1065 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 48% [48%-48%] | - | 5% [5%-5%] | 392 / 2351 | 413 / 2370 | 94% | 1.47x | - | 1.06x | 1.06x | 0.85x |
| (c) GPU+SPU+DRU | fair | 49% [49%-49%] | 8% [8%-8%] | 4% [4%-4%] | 355 / 1857 | 376 / 1885 | 93% | 1.45x | 3.47x | 1.08x | 1.08x | 0.55x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 1080720 CK, SM lane alone = 645779 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 263 / 1026 | 283 / 1042 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 52% [52%-52%] | - | 4% [4%-4%] | 359 / 1509 | 380 / 1535 | 94% | 1.36x | - | 1.11x | 1.11x | 0.74x |
| (c) GPU+SPU+DRU | fair | 42% [42%-42%] | 6% [6%-6%] | 4% [4%-4%] | 413 / 2018 | 434 / 2068 | 92% | 1.68x | 4.40x | 1.12x | 1.03x | 0.55x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 2162345 CK, SM lane alone = 1294306 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 264 / 1003 | 284 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 33% [33%-33%] | - | 4% [4%-4%] | 567 / 1908 | 590 / 1947 | 92% | 2.12x | - | 1.21x | 1.27x | 0.63x |
| (c) GPU+SPU+DRU | fair | 44% [44%-44%] | 6% [6%-6%] | 4% [4%-4%] | 400 / 1571 | 420 / 1596 | 92% | 1.62x | 4.91x | 1.16x | 1.04x | 0.65x |
