# Adaptive sparse GQA prefill

TensorCore (TC) attention reuses KV across query blocks; SparseCore (SC)
gathers each query's selected KV before grouped attention. The QSA caller
supplies rank-local `kv_heads` and the configured compression `ratio`.
Eligibility checks packed cache layout, dtype, GQA divisibility, alignment,
unit K/V scales, available VMEM and SparseCore support. Physical page size,
page-table width and selected-group budget are inputs to the work estimate.

## Metadata and scheduling

Callers provide int32 offsets/IDs and nondecreasing request starts from zero.
Each active query has a unique valid prefix of
`min(length // ratio, selected_group_budget)` complete groups, followed by -1
padding; the partial tail is implicit. Inactive rows have zero lengths and
-1 IDs and produce zero even for NaN Q/gate. Repeated padded starts are valid.

The adaptive schedule uses TC query blocks of 256 and SC groups of eight.
The TC KV tile follows the resource-selected geometry; the SC attention tile
contains 256 KV rows. These are kernel scheduling choices. Ragged TC work
uses request-local query tiles, bounded packs and source windows. Its cost
accounts for the resulting calls and padded query rows. SC gathers each
query's own page table, including requests that start inside a TC block.
See [admission and dispatch][dispatch].

## Work model and choice

The [work model][cost] derives byte traffic and QK/PV matrix-product FLOPs
from actual query heads/dimensions, KV heads, compression ratio, cache packing
and dtype, selected-group budget, page-table width and scheduled tiles.
Page size determines the logical tile extent; table width contributes index
traffic. SC accounting includes padded index/table loads, compact gather
traffic, packed query/gate/output boundaries and padded attention heads.
TC accounting includes metadata, mask words, repeated KV-tile loads and
request-local pack padding.

Traffic is divided by public TPU memory bandwidth; matrix-product FLOPs are
divided by public BF16 throughput, using the reported core count. These are
peak-rate work estimates, not measured latency predictions. They exclude
runtime launch latency and do not model every compiler allocation or
non-matrix operation.

The selector chooses SC when estimated TC cost exceeds `1.02 * SC cost`.
The margin is a policy, not a numerical error bound or measured optimum.
Nonnegative work terms support lower/upper bounds; exact selected-tile
unions are computed only when those bounds leave the choice unresolved.
The model has no fitted coefficient table or calibration-data dependency.

## Memory and arithmetic

Outer dispatch chunks contain at most 8192 queries. SC microchunks are
bounded by dtype-specific row ceilings and a soft selected-KV workspace
target in [the scheduling policy][tuning]; actual geometry can reduce the
microchunk or use a TC resource fallback. TC metadata and request-local packs
have separate soft workspace targets. These logical planning estimates are
not compiler peak-memory guarantees. Compiler memory analysis and measured
resident memory describe different allocation/resource observations.

FP32 accumulation and explicit BF16 output/gate rounding preserve QSA
arithmetic. Each query keeps its own softmax. The cache is returned unchanged;
standalone non-donating JIT returns can still materialize an array. The serving
custom op donates its pool.

## Checks and benchmark

Run the focused CPU work-model and dispatch tests in the project environment:

```bash
JAX_PLATFORMS=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest \
  tests/kernels/test_sparse_gqa_prefill_work_model.py \
  tests/kernels/test_sparse_gqa_prefill_cost_model.py \
  tests/kernels/test_sparse_gqa_adaptive_prefill.py -m 'not tpu_test'
python scripts/benchmark_qsa_prefill.py --help
```

Hardware tests use the `tpu_test` marker. The benchmark includes
attention/gating and the selected dispatch/gather path, excluding compilation,
indexer scoring, cache writes and other model layers. Kernel fixtures are not
checkpoint or distributed-serving qualification.

[dispatch]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/adaptive.py
[cost]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/cost_model.py
[tuning]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/tuned_block_sizes.py
