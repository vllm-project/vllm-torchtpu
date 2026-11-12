# CSA2 kernel

`csa2.py:csa2` is the serving attention entry point for decode and both prefill
stages. Callers supply request starts `[B+1]`, per-token positions `[T]`, packed
main KV, FP32 scales, SWA word-row pages and visibility metadata directly.
There are no decode adapters, query-compaction passes or output-reordering passes.

Encoder/decoder identify the first and last 20 layers of the 40-layer DSV41
configuration; they do not identify causal/noncausal visibility modes.
Optional `swa_bounds` independently controls explicit visibility.

Both 32- and 64-token SWA pages, ragged requests, repeated request boundaries,
output padding and split SWA banks are supported. The page size comes from the
native owner shape. Six-query decode uses `bq=6`; longer queries use `bq=8` or
`bq=16`, subject to the accumulator capacity for the number of heads.

The kernel builds query tiles in SMEM. Four Q/KV/SWA buffers overlap DMA and
compute; two output buffers carry PV across tiles. Valid output rows are written
directly to their original token positions, with padding zeroed in the kernel.
Each 64-row packed-KV read is converted in four 16-row slices shared by QK/PV.
FP32 scale and inverse-RoPE tables remain resident when they fit within a quarter
of core VMEM; larger tables are streamed through bounded buffers. Query, packed
KV and SWA owners are constrained to HBM so the compiler does not copy a complete
owner into another memory space before entering the kernel.

`cache.py` provides cache publication and SparseCore gathers. `pipeline.py`
overlaps those gathers with calls to the same `csa2` kernel and retains gathered
KV for reuse within a model stage. `reference.py` contains independent attention
math for correctness checks. Benchmark-only mathematical ablations are not part
of the production kernel.

Run the common real-input benchmark on TPU:

```bash
python scripts/dsv41_bench/bench_csa2.py --out /tmp/csa2-decode \
  --batch 48 --num-tokens 6 --bq 6 --page-size 32
python scripts/dsv41_bench/bench_csa2.py --out /tmp/csa2-prefill \
  --batch 1 --num-tokens 2048 --bq 8 --page-size 64
```

`--baseline /path/to/csa2.py` optionally compares another implementation with the
same interface. Reports separate kernel and complete XLA module timings, checks
sampled outputs against NumPy, and compares full outputs for A/B (atol=3e-3, rtol=3e-2), reporting
bitwise equality separately.

## Native FP8 matmul W8A16 pipeline

`csa2_matmulw8a16.csa2_matmulw8a16` is an explicit TPU7x entry point for
row-packed main KV. Both QK and PV push an FP8 e4m3fn RHS against a BF16 LHS,
without a decoded BF16 cache or shifts/masks to unpack main KV. It uses the
existing tile/window helpers, four input buffers, alternating RHS staging,
three-query softmax chunks over 256-wide key blocks, and post-PV normalization.
The next tile's QK overlaps the current softmax; output DMA starts after issuing
current PV so finishing the previous output does not serialize that work.

Main KV is INT32 `[T,128,512]`: byte `r` of word `[t,k,d]` contains the FP8
value `[t,4*k+r,d]`. Scales stay FP32 `[T,512]`, one scale per key row. This
is a different packing from the legacy `[T,512,128]` interface, not a reshape.
SWA must use native BF16 pages, optionally split across two owners. Ragged
queries and explicit SWA visibility retain their semantics. The function returns
only the attention output.
`uniform_queries=True` requires a single SWA owner and exactly `bq` queries per request with
`starts = arange(B+1)*bq`; otherwise use the default ragged path.

Reproduce a same-input legacy/W8A16 comparison with:

```bash
python scripts/dsv41_bench/bench_csa2.py --matmulw8a16 \
  --batch 48 --num-tokens 6 --heads 64 --page-size 64 --bq 6 \
  --out /tmp/csa2-matmulw8a16
```

The benchmark converts fixture bytes before compilation/timing and records each
layout. It validates against the independent FP32 reference and saves XProf
traces, HLO and source hashes. The serving adapter still uses the legacy entry
point: direct row-packed cache production, SWA-only layers and shared INT32 SWA
word-pool integration remain follow-up work. Do not insert the benchmark's host
packing conversion into the serving path or include it in the kernel timing.
