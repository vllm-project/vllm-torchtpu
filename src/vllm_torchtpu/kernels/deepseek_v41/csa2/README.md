# CSA2 kernel

`csa2.py:csa2` is the single attention entry point for decode and both prefill
stages. Callers supply request starts `[B+1]`, per-token positions `[T]`, packed
main KV, FP32 scales, native BF16 SWA pages and visibility metadata directly.
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
