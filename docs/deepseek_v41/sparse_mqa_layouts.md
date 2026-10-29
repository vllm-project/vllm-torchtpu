# Sparse MQA logits: token-major layout experiment

Base: `1fb85eb97`; Falcon experiment `exp-mx25v52hnu`.
This isolates sparse K gather and MQA scoring. Cache packing, candidate preparation,
Top-K, incremental cache writes and serving execution are outside the timed region.
Production kernels are unchanged.

## Workload and measurement

v7x, P7, one device of an eight-device allocation; JAX/jaxlib 0.11.2,
libtpu 0.0.50. 48 requests × 6 queries, context 98816, H=32, D=128,
2048 candidate blocks × 8 tokens, physical page 128, two banks, nominal overlap 0.7.
All variants use identical quantized bits, scales, pair windows and 16384-column
outputs. Seeds: 826/827. Cache layouts use 132 bytes per token.
The primary time is the median of five warmed XPlane XLA Modules samples;
50 host samples are retained separately. Compilation is excluded.

## Result

The fastest measured gather + logits variant is `tm4-native`, at **470.776 µs**
versus **651.001 µs** for production field-major. It requires four-token byte
packing. The fastest measured pure token-major scorer is `tm32-lanes`:
565.146 µs, with 859.704 µs for gather + logits; it does not beat production.
The benchmark defaults to `field,tm4-native,tm32-lanes`. Other names retain the
failed or slower experiments for explicit reproduction; `tm33` fails compilation
and fully static `tm32` runs out of VMEM at the full shape.

## Layouts

- `field`: production `[page, 33 fields, 128 tokens]` packed words.
- `tm32`: each token has 32 consecutive K words; scales use a separate stream.
- `tm4-native`: each word contains one channel from four adjacent tokens.
  Four-token groups are consecutive. This is a different byte packing from
  plain token-major records, though eight-token sparse blocks remain contiguous.
- `tm32-lanes`: plain token-major input, with fixed lane gathers and bit operations
  inside the scorer to form the same four-token packing before the MXU dot.

## Measurements

| Variant | Gather µs | Logits µs | Gather + logits µs |
| --- | ---: | ---: | ---: |
| field | 496.391 | 156.625 | 651.001 |
| tm32, generic transpose, 4K loop / depth 4 | 299.853 | 2740.144 | 3035.944 |
| tm32, generic transpose, 8K loop / depth 8 | 299.633 | 3381.686 | 3678.193 |
| tm32, FP8 transpose, 4K loop / depth 4 | 298.782 | 10812.783 | 11108.848 |
| tm4-native, static / depth 16 | 299.126 | 175.026 | 470.776 |
| tm4-native, 4K loop / depth 4 | 299.047 | 345.052 | 640.429 |
| tm32-lanes, static / depth 16 | 298.254 | 565.146 | 859.704 |
| tm32-lanes, 4K loop / depth 4 | 298.606 | 698.574 | 993.037 |
| tm32-pack, hardware elementwise pack | 298.272 | 569.879 | 865.452 |
| tm32-butterfly, static / depth 16 | 298.475 | 1828.973 | 2122.639 |
| tm32-butterfly, static / depth 4 | 298.774 | 1767.398 | 2062.841 |

These are independent measurements; adding component medians does not reproduce
an independently compiled composition exactly. The native packing improves the
combined time by 27.7%, but its scorer is 11.7% slower than the field-major scorer.
This does not include the cost of changing the cache writer.

## LLO evidence

The original full static token-major reshape/transpose does not fit VMEM:
109.76 MiB requested against 63.94 MiB, including 92.60 MiB of reserved register
spill slots. Reducing issue depth to one still requests 108.37 MiB.
TM33 directly fails layout inference on `vector<1x66x128xi32> → vector<256x33xi32>`.

| Static kernel | vmatmul | vselect | vrot | vxpose |
| --- | ---: | ---: | ---: | ---: |
| field | 1376 | 1212 | 1442 | 0 |
| tm32 generic transpose | 1376 | 148668 | 112034 | 36864 |
| tm4-native | 1376 | 1212 | 1442 | 0 |

Counts cover all mutually exclusive static prefix branches in pre-RA LLO.
They are code-size/instruction evidence, not dynamic cycle counts or runtime shares.
The identical MXU counts and disappearance of extra rotations/selects identify
layout conversion as the main target. Loop variants compile, but their timings do
not make them an optimization; loop structure and issue depth both differ.

## Why the native scorer is still slower

A control gathers native-packed K, then combines K and scales into one 33-row
page buffer **before timing the scorer**. Logits are **174.785 µs**, versus
175.026 µs with two separate K/scale streams and 156.625 µs for field-major.
Removing the extra DMA stream therefore does not explain the 18.4 µs gap.
This control's buffer conversion is deliberately not a proposed pipeline: its
measured gather/conversion is 2369.733 µs and composition is 2543.576 µs.

XPlane places the gap inside the custom call: 153.469 µs for field-major versus
171.756 µs for native; the two input-layout copies remain approximately 3.1 µs
in both. LLO shows `gain_latch_mode<f8e4m3b>` for field-major versus
`gain_latch_mode<xpose.f8e4m3b>` for native. The native RHS is loaded through the
MXU transpose mode. This locates the remaining difference in matrix-input
layout/preparation and scheduling; it does not isolate a hardware transpose
stall latency of exactly 18 µs.

## Post-RA spill attribution

The generic-transpose 4K loop variant has a scheduled spill live peak of
**11.36328125 MiB**, versus **11.42578125 MiB** across 2925 symbolic slots.
Its reserved physical spill frame was not separately measured. The 92.60 MiB
reserved frame above belongs to the different, fully static OOM variant.
All 3160 spill stores were attributed with filename-aware source locations;
**98.50% of spill occupancy area** comes from the narrow-column K reshape.
The exact source snapshots, final bundles, HTML timeline and cause/fix table
are saved in the experiment's `spill-analysis/` directory. The horizontal axis
is a scheduled bundle position, not a dynamic XProf execution clock.

## Validation and limits

Each completed run compares sampled gathered words exactly (up to roughly 4096
blocks), every logit's finite mask exactly, and all finite logits with
`rtol=atol=0.01`. Completed variants have maximum absolute logit error zero.
Small fixtures include an empty query and causal holes; additional fixtures exercise
all sixteen eight-token offsets within a 128-token page and singleton groups.

This is a synthetic isolated-kernel experiment. No cache writer integration,
full Indexer, model quality or serving evaluation has been performed.

## Reproduction and artifacts

With the repository's TPU runtime installed:

```bash
python scripts/dsv41_bench/bench_sparse_mqa_layouts.py \
  --out /tmp/mqa-layouts --variants field,tm4-native,tm32-lanes --repeats 50
python scripts/dsv41_bench/bench_sparse_mqa_layouts.py \
  --out /tmp/mqa-offsets --edge-case offsets --variants tm4-native,tm32-lanes
python scripts/dsv41_bench/bench_sparse_mqa_layouts.py \
  --out /tmp/mqa-singleton --edge-case singleton --variants tm4-native,tm32-lanes
```

The Falcon artifact retains the exact repository snapshot, per-iteration source
archives, standalone JAX bootstrap (including the production fori-lowering patch),
package versions, metrics, all validation reports, XPlane traces and representative
LLO. A minimal JAX image needs that bootstrap; a normal repository installation
loads the patch through package initialization. Source archives, not just the base
commit, are required to reproduce these uncommitted experimental kernels.

Local changed-file pre-commit checks cover Ruff, formatting and baseline Pyrefly.
These checks do not substitute for type checking with the full serving dependencies.
