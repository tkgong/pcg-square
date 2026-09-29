
### L40S (2 channels simulated), SPU alone = 1272643 CK (565 us); standalone host-stream BW per channel: sm 71% of peak, full 71% of peak, dru 10% of peak

**r = 0.12** (GPU-alone / SPU-alone time).  GPU-only full NTT = 153540 CK, SM lane alone = 93772 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 258 / 1070 | 278 / 1088 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 39% [39%-39%] | - | 8% [8%-8%] | 475 / 3810 | 498 / 3828 | 93% | 1.78x | - | 1.05x | 1.05x | 0.94x |
| (c) GPU+SPU+DRU | fair | 39% [39%-39%] | 6% [6%-6%] | 8% [8%-8%] | 443 / 3070 | 465 / 3116 | 92% | 1.77x | 4.82x | 1.07x | 1.07x | 0.58x |

**r = 0.25** (GPU-alone / SPU-alone time).  GPU-only full NTT = 318510 CK, SM lane alone = 191734 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 70% [70%-70%] | - | - | 262 / 1017 | 281 / 1033 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 42% [42%-42%] | - | 8% [8%-8%] | 453 / 3047 | 476 / 3505 | 93% | 1.69x | - | 1.09x | 1.09x | 0.87x |
| (c) GPU+SPU+DRU | fair | 38% [38%-38%] | 7% [7%-7%] | 8% [8%-8%] | 456 / 3624 | 478 / 3658 | 92% | 1.85x | 3.88x | 1.09x | 1.09x | 0.56x |

**r = 0.5** (GPU-alone / SPU-alone time).  GPU-only full NTT = 636339 CK, SM lane alone = 382334 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 263 / 1003 | 282 / 1019 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 43% [43%-43%] | - | 7% [7%-7%] | 434 / 2754 | 456 / 2857 | 93% | 1.63x | - | 1.15x | 1.15x | 0.77x |
| (c) GPU+SPU+DRU | fair | 34% [34%-34%] | 5% [5%-5%] | 7% [7%-7%] | 522 / 3782 | 544 / 3800 | 91% | 2.08x | 5.39x | 1.14x | 1.08x | 0.58x |

**r = 1** (GPU-alone / SPU-alone time).  GPU-only full NTT = 1273483 CK, SM lane alone = 763972 CK
| config | policy | GPU BW util | DRU BW util | SPU slot occ. | GPU rd queue mean/p99 (ns) | GPU rd latency mean/p99 (ns) | GPU row hit | GPU time vs alone | NTT done vs (a) | SPU time vs alone | both done vs max() | both done vs serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) GPU-only | - | 71% [71%-71%] | - | - | 264 / 999 | 284 / 1017 | 95% | 1.00x | 1.00x | - | - | - |
| (b) GPU+SPU | fair | 24% [24%-24%] | - | 7% [7%-7%] | 810 / 4240 | 839 / 4299 | 90% | 2.99x | - | 1.21x | 1.42x | 0.71x |
| (c) GPU+SPU+DRU | fair | 33% [33%-33%] | 5% [5%-5%] | 7% [7%-7%] | 539 / 2690 | 561 / 2715 | 91% | 2.15x | 5.50x | 1.20x | 1.11x | 0.69x |
