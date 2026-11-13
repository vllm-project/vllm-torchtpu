# Measured conclusions

Measured on 2026-11-13, TPU7x-8, eight devices running concurrently. These are
finite 32 MiB transfers in the Pallas implementation described in [README.md](README.md),
not guaranteed hardware limits or end-to-end QSA performance.

## Single-DMA latency

For contiguous reads from 4 KiB to 32 MiB:

```text
L(M) = 0.8751 us + M / (3.4139 TB/s)
```

The maximum relative residual is 21.9%, so the intercept is an approximation,
not a strictly constant delay. Fitting only transfers of at least 1 MiB gives
0.9857 us + M / 3.4719 TB/s, with a maximum residual of 3.0%.

Representative contiguous latencies: 4 KiB 0.719 us, 64 KiB 0.876 us,
256 KiB 0.989 us, 1 MiB 1.251 us, 8 MiB 3.466 us, 32 MiB 10.642 us.
Small-page random64M latencies: 1 KiB 0.671 us, 2 KiB 0.693 us,
4 KiB 0.719 us, 8 KiB 0.750 us, 16 KiB 0.792 us.

## Windowed contiguous reads

The first tested windows reaching an eight-device mean of 80% MBU were:

| DMA size | First tested Q >= 80% | Best measured MBU |
| --- | ---: | ---: |
| 4 KiB | None | 15.25% |
| 8 KiB | None | 29.02% |
| 16 KiB | None | 53.06% |
| 32 KiB | 256 | 82.91% |
| 64 KiB | 64 | 85.18% |
| 128 KiB | 32 | 85.40% |
| 256 KiB | 16 | 85.49% |
| 512 KiB | 16 | 85.56% |
| 1 MiB | 8 | 85.50% |
| 2 MiB | 4 | 85.45% |
| 4 MiB | 2 | 85.44% |
| 8 MiB | 2 | 85.37% |

The sweep is discrete; these are not proven minimum windows. Peak throughput
here is about 3.15 TB/s/device, including fill/drain of the finite transfer.
Non-power-of-two Q=192 regressed in some small-block cases; the exact cause
was not isolated. Q is a software outstanding limit, not measured hardware
occupancy. The eight-way unrolled issue loop is part of this implementation.

## Logical pages and fragmented transfers

Continuous addresses crossing a half-page boundary did not materially degrade
performance. Page-table lookup and random addresses had modest effects at large
windows. For eight logical 1 MiB blocks in flight (8 MiB payload buffers):

| Fragments/block | Physical DMA window | Sequential pages: us / MBU | Random64M: us / MBU |
| --- | ---: | --- | --- |
| 1 x 1 MiB | 8 | 10.634 / 85.51% | 10.908 / 83.37% |
| 4 x 256 KiB | 32 | 10.639 / 85.48% | 10.826 / 83.99% |
| 16 x 64 KiB | 128 | 10.662 / 85.29% | 11.127 / 81.73% |

64 KiB random fragments were about 4.4% slower than sequential fragments at
matched buffer capacity. By contrast, an independent window of only eight
64 KiB DMAs reached about 15.5% MBU. Losing in-flight byte capacity can matter
much more than randomized logical page IDs.

## Small pages

The supplement tested Q=1,8,16,32,64,128,256,512 for every size and address
pattern. No 1–16 KiB configuration reached 80% MBU.

| DMA size | Sequential page table: best MBU / Q | Random64M: best MBU / Q | Random64M time for 32 MiB |
| --- | --- | --- | ---: |
| 16 KiB | 54.22% / 256 | 54.21% / 256 | 16.773 us |
| 8 KiB | 29.29% / 256 | 29.29% / 256 | 31.046 us |
| 4 KiB | 15.36% / 128 | 15.26% / 256 | 59.591 us |
| 2 KiB | 7.83% / 128 | 7.81% / 128 | 116.493 us |
| 1 KiB | 3.95% / 128 | 3.94% / 128 | 230.617 us |

Random-page throughput closely matched sequential-page throughput. Increasing
Q beyond 128–256 gave little benefit; Q=512 regressed. Best-window amortized
cost per DMA was approximately 8.19, 7.58, 7.27, 7.11 and 7.04 ns respectively.
This supports an issue/scheduling throughput constraint in this implementation.
It is distinct from the roughly 0.7 us completion latency of a single DMA,
and does not establish an absolute hardware issue-rate limit.

## Implications for QSA

Measure actual bytes per DMA and the number of consumable requests in flight.
If DMAs are only 1–16 KiB, consider coalescing contiguous regions or changing
the tile layout to issue larger transfers; merely increasing Q is unlikely
to recover high MBU in this implementation. For larger transfers, enough
in-flight bytes can hide much of the single-transfer latency.

Validate changes in the actual QSA kernel. This benchmark excludes page-table
HBM preloading, attention computation, and competition for VMEM/registers.
It does not control physical-page translation, and sampled correctness does
not validate every byte transferred. The supplied data are observations from
one machine/environment, not confidence intervals across independent runs.

All 501 configuration tables and fits are reproducible with `summarize.py`.
The JSON records retain individual call durations so other aggregations can
be computed without rounding the published tables.
