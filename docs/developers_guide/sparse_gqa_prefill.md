# Sparse GQA prefill

Normal prefill uses one TC/SC selector with work estimates derived from tensor
geometry and installed TPU resources. It lazily refines bounds on TC tile-union
visits and compares the estimated TC and SC costs. `sampled_lower` is a lower
bound on TC union visits. Ambiguous decisions prepare exact TC metadata once;
if TC is selected, it consumes that same branch-local metadata bundle.
Membership certificates preserve TC set semantics for duplicates and invalid,
future or padded selections. Unsupported schedules retain the TC path.

Set the optional positive integer `VLLM_TPU_QSA_SC_MIN_CONTEXT` to prefer SC
when the maximum effective causal length among active queries in this prefill
slice reaches the value, inclusively. Inactive and clipped-out rows do not
contribute. Below that value, the cost model still chooses either TC or SC.
All geometry, scale, membership and page-plan safety checks remain; a rejected
SC path retains TC semantics. Preferring SC can be slower for high-reuse input.
Unset or empty keeps the default work-based selector.

## Bounded SparseCore execution

The SparseCore path stages request-local live page prefixes into a bounded
source independent of persistent KV-pool capacity. Device certificates check
request namespaces, physical page bounds and meaningful selected groups.
Padding reads an initialized null page; unused staging capacity is not read.
An uncertified page plan falls back to TC.

Staging windows and metadata slabs must satisfy the SparseCore 16 GiB parameter
limit and the kernels' VMEM and DMA constraints. An internal workspace ceiling
includes the null page; it bounds allocation but does not guarantee free HBM.

Metadata is packed once for the whole query batch. Four SparseCore DMA streams
produce alternating owned HBM Ref slots. Ref effects enforce write-after-read
ordering before a slot is reused. Batched QK/PV products keep each query's KV
list independent, with FP32 accumulation and the original masks. Attention
rounds to BF16 before the outer FP32 sigmoid/product and final BF16 cast.

The output-only SC helper keeps the native cache read-only and does not donate
or return it. The outer pooled custom op still donates `pool` and `index_pool`,
operands 10 and 11.

Hardware peak rates are not achieved latency. Submission costs, efficiency and
physical SC/TC overlap are not fully modeled; compiler scheduling does not
guarantee physical overlap.

## Tests

Run from the repository root in the project test environment described in
[Testing & CI](testing_and_ci.md). The CPU contracts use explicit target-info
fixtures and mocked kernel calls where hardware is required:

```bash
PYTHONPATH=src:. JAX_PLATFORMS=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest -m "not tpu_test" \
  tests/kernels/test_qsa_selector.py \
  tests/kernels/test_sparse_gqa_prefill_cost_model.py \
  tests/kernels/test_qsa_owner_window.py \
  tests/kernels/test_qsa_default_dispatch.py
```

On a supported TPU, run the grouped FP8/BF16 numerical and cache-readonly
contracts:

```bash
PYTHONPATH=src:. JAX_PLATFORMS=tpu,cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest -m tpu_test tests/kernels/test_sparse_gqa_grouped_prefill.py
```
