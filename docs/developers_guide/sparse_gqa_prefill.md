# Adaptive sparse GQA prefill

TensorCore (TC) attention reuses KV across query blocks; SparseCore (SC)
gathers each query's selected KV before grouped attention. The Qwen QSA
adapter selects this kernel only within its calibrated envelope and retains
TC otherwise. Adding this standalone kernel does not change model defaults.

## Admission and metadata

Adaptive admission currently requires TPU7x with SparseCore, BF16 Q/gate
`[T, 24, 256]`, two KV heads, ratio four, 512 selected-group slots, unit K/V
scales, and native BF16 or FP8 E4M3FN cache. The validated guards are page size
1664, `1 <= T <= 262144`, up to 158 page-table columns and a 16 GiB cache.
These scope the calibration, not physical addressing or hardware capacity.
Other valid geometries/scales retain TC. Paired TC eligibility derives from
local heads, GQA divisibility, dimension alignment, packing, page segments and
working-set capacity; it does not assume a TP degree or a D256 upper bound.

Callers provide valid int32 offsets/IDs, nondecreasing request starts from
zero, and a unique valid prefix of `min(length // 4, 512)` complete groups,
then -1 IDs; the partial tail is implicit. Inactive rows have zero lengths
and -1 IDs and produce zero even for NaN Q/gate. Repeated padded starts are
valid. Boundaries inside Q256 retain request-aware TC alignment because its
padding is outside the calibrated union costs. See [admission and dispatch][dispatch].

## Cost and choice

Predictions are milliseconds, with separate BF16/FP8 [fitted coefficients][cost]:

```text
TC = a0 + a1*Qtc + a2*Qtc*W + a3*V
SC = b0 + b1*Qsc + b2*A
Select SC if TC > 1.02*SC; otherwise TC.
```

`Qtc` and `Qsc` count scheduled rows, including branch padding and tail bounds.
`W = ceil(2*num_tiles/128)*128` counts aligned uint32 mask words. `V` sums
unique KV256 tiles per Q256 block; `A` sums KV256 tiles per eight-query group,
using each group's largest selected length and at least one tile when inactive.
Static/span bounds, sampled lower bounds and coarse upper bounds avoid exact
unions where possible. Sampling may establish SC, never TC by missing queries;
nonnegative coefficients keep these work bounds monotone.

The 1.02 margin prefers TC near crossover; it is a policy, not an error bound
or measured optimum. The [fit inputs, expected fit and commands][fit] preserve
all 14 coefficients from the calibrated TPU7x geometry. They do not qualify
other chips/shapes or current full-model latency; historical validation feedback
also informed dispatch, so it was not an untouched blind test of the final policy.

## Memory and arithmetic

Outer chunks contain at most 8192 queries. BF16/FP8 gathers use microchunks
of 32/64 queries, bounding selected-KV storage to 144 MiB at the supported
budget. Only the final tile/microchunk is padded; output writes and scratch
reuse avoid whole-context selected-KV or alignment workspaces. Compiler scratch
checks are regression targets, not measured peak HBM.

FP32 accumulation and explicit BF16 output/gate rounding preserve QSA arithmetic.
Each query keeps its own softmax. The cache is returned unchanged; standalone
non-donating JIT returns can still materialize an array. The serving custom op
already donates its pool; this kernel does not change that ownership contract.

## Checks and benchmark

Use the project environment and `tpu_test` markers for hardware tests:

```bash
pytest tests/kernels/test_sparse_gqa_adaptive_prefill.py \
  tests/kernels/test_sparse_gqa_prefill_cost_model.py \
  tests/kernels/test_sparse_gqa_grouped_prefill.py
python scripts/benchmark_qsa_prefill.py --help
```

The benchmark includes dispatch/gather/attention/gating, excluding compilation,
indexer scoring, cache writes and other model layers. Kernel fixtures are not
checkpoint or distributed-serving qualification.

[dispatch]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/adaptive.py
[cost]: ../../src/vllm_torchtpu/kernels/sparse_gqa_prefill/cost_model.py
[fit]: ../../scripts/sparse_gqa_prefill_cost_model/README.md
