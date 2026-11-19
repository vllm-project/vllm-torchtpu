# Adaptive Fused MoE

Experimental fused expert-parallel MoE, derived from `fused_moe/v2` at
`0a05f11d9a77ee4516cfdcfde946203c03f4ff1f`. The separate implementation keeps
buffer and scale-pipeline experiments isolated from the production v2 path.

## Weight buffers

At kernel construction, estimate padded scratch and temporary storage against
the device VMEM budget. Prefer three W1 buffers and two W2 buffers; fall back
to two W1 buffers and one W2 buffer. Reject shapes that exceed the budget even
with the smaller plan. The policy uses shapes and formats, without model-name
branches or changes based on live request concurrency.

W1 and W2 have independent ring slots and DMA semaphores. Issue W2 DMA before
waiting for W1, then wait once per expert at the first down-projection read:

- **3+2:** prefetch the next visited expert's W2 during current-expert work.
- **2+1:** load the current expert's W2 so its transfer can overlap W1 compute.

FP4 block scales stream with the W1 ring. Hidden sizes above 4096 are supported
subject to alignment and VMEM checks. The serving adapter rejects TPU
generations below 7 before building the op. On TPU7x with 64 local experts and tile
capacity 128, the tested Qwen3.5 FP8 geometry selects 3+2 (40.39 MiB estimate),
and Qwen3.8 NVFP4 with runtime block 64 selects 2+1 (60.77 MiB estimate).
These are conservative static estimates, not measured compiler peaks.

## Explicit entry point

```python
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
    WeightFormat,
    adaptive_fused_moe,
    select_weight_buffers,
)

plan = select_weight_buffers(
    g_local=64, capacity=128, hidden=4096, inter=1024,
    weight_format=WeightFormat.FP8,
)
```

`adaptive_fused_moe` accepts the original K-major tensor layouts. Weight format
is inferred from the dtype when omitted; for FP4, the contraction block size is
inferred from the scale layout. An explicit format/block must match the weights.
Native FP4 uses W4A8 automatically, without `MOE_FUSED_EP_ENABLE_W4A8`.

Routing always uses the sharded plan. The experimental API has no
`sharded_plan` argument and does not read `MOE_FUSED_EP_V2_SHARDED_PLAN`.
Unsupported shapes, dtypes, scale layouts or memory requirements raise errors.

### Serving integration

`vllm_adapter.prebuild_adaptive_fused_moe` accepts a layer and its prepared
K-major Torch weights, automatically chooses FP8/FP4 from their dtype, and
returns the experimental Torch/JAX op. It raises on unavailable EP, unsupported
weight/routing contracts, insufficient SMEM/VMEM, or a deployment below
`MOE_FUSED_EP_KERNEL_MIN_TOKENS`. It never returns a GMM fallback. Store the op
on the quantization-method owner under `vllm_adapter.FUSED_MOE_EP_OP_ATTR`, then
use `vllm_adapter.run_adaptive_fused_moe`; a missing op is an error.

Select the experimental FP8/NVFP4 serving path with one variable:

```bash
MOE_FUSED_EP_KERNEL_IMPL=adaptive
```

No additional fused-enable, W4A8 or sharded-plan flag is required. The selection
is forwarded to workers and bound during weight loading; forward uses the bound
op. NVFP4 reuses the existing scale fusion and K-major requantization, choosing
runtime block 64 when no block override is supplied. The adapter validates the
proposed layout before converting checkpoint weights. vLLM recognizes the
experimental op as the owner of EP/TP/PCP communication and final reduction.

When this variable is unset, the legacy `USE_MOE_FUSED_EP_KERNEL` behavior is
unchanged. Explicit `v2` enables the original implementation and retains its
existing W4A8/routing-plan controls. Invalid implementation names raise errors.
The original v2 kernel and adapter remain unchanged.

## TP-replicated inputs

Ported the token-ownership fix from [PR #1084](https://github.com/vllm-project/vllm-torchtpu/pull/1084)
(commit `196d27f7f8717a005c12856b23535b3dbdab9f93`) into this experimental
implementation. At load time the adapter maps native TP group membership to
EP mesh positions. Replicated TP members split input rows before routing and
quantization, then gather the combined rows within the group to restore the
complete output on every member. Sequence-parallel inputs use singleton groups.
Singletons introduce no slicing, padding, barrier or extra collective. Quantized
replicated inputs retain the pre-quantization barrier to preserve BF16 rounding.
Both the layer and adapter cache keys include replica layout/group information.
The original v2 kernel and its adapter are unchanged by this port.

## FP4 scale memory

The v2 snapshot this experiment was derived from (`0a05f11d9a77`) kept
both FP4 scale tables resident for all local experts.
At hidden 8192 / intermediate 2048 / block 64, the two scale tables cost
3 MiB per expert: 24 / 192 / 768 MiB for 8 / 64 / 256 local experts.
Production v2 now also streams scales with its weight slots (PR #1137), so
the current v2 implementation is no longer a resident-scale baseline.
The experimental implementation streams both tables with the W1
prefetch ring: 2+1 uses 6 MiB and 3+2 uses 9 MiB for scales at all three
expert counts. W1/W2
scales have their own DMA semaphores, and both remain valid through every
token tile of the current expert. The scale-memory test checks actual scratch
declarations against these fixed budgets using TPU7x padding, independently
of production v2; these are allocation estimates.

## Validation

Run from the repository root in the supported TPU environment:

```bash
python -m pytest tests/kernels/experimental/adaptive_fused_moe -v -s
```

The device tests require eight TPU devices. Coverage includes FP8/BF16/FP4/INT8,
sharded routing against the original v2 reference, expert remapping, empty expert visits,
multi-tile weight reuse, VMEM selection boundaries, full Qwen3.5 layer geometry,
and full Qwen3.8 NVFP4 layer geometry at 2048 and 16384 tokens. The Qwen3.8 cases
use synthetic packed checkpoint weights, the existing block-16 to block-64
requantizer, and a dense reference with unchanged accuracy thresholds.
Additional FP4 tests exercise sparse noncontiguous expert IDs, scale-ring wrap,
multiple tiles per expert, and entirely empty ranks. These DMA stress cases
require bit-exact equality with the production v2 implementation.
Their dense-reference errors are reported separately: strongly unequal scales
can exceed the 0.075 worst-token bound in both kernels (observed 0.0770), so
that bound is not asserted for these stress inputs. The dedicated numerical
tests, including full Qwen layer geometries, retain their existing bounds.
Adapter contract tests
verify automatic format selection and explicit failures instead of fallback.

The preceding POC measured one Qwen3.5 DP8/SPEED prefill round: high-concurrency
3+2 throughput was close to the original 3+3 baseline, with an unresolved C16
difference. Full-model Qwen3.8 performance has not been validated. Migration
correctness tests do not constitute another end-to-end performance measurement.
See [issue #1024](https://github.com/vllm-project/vllm-torchtpu/issues/1024).
