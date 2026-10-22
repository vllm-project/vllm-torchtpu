# DeepSeek V4.1 on TorchTPU

Serving registers `deepseek_v41.v2.model:DeepseekV41ForCausalLM` and the v2
DSpark draft. Model construction, checkpoint names and cache ownership follow
pinned vLLM v0.31.1rc0. The v1 modules remain a standalone test reference;
they are no longer a selectable serving backend.

## Serving configuration

Keep topology, KV dtype, scheduling, compilation buckets and speculation in
vLLM's standard arguments. Engram placement uses the existing typed
`--engram-config`; TPU performs host lookups rather than CUDA UVA accesses.
Two TPU-specific decisions use vLLM's OOT config transport:

```bash
--engram-config '{"cpu_offload":true}' \
--additional-config '{"deepseek_v41":{"execution":"ced","moe":"fused_ep_w4a8"}}'
```

`models/vllm/deepseek_v41/config.py` parses that namespace into a frozen typed
policy, rejects unknown keys and invalid values, validates combinations, and
publishes canonical defaults before workers are created. Consumers use that
policy rather than reading the dictionary themselves. The canonical values
participate in vLLM's existing additional-config hash and worker serialization.
This transport avoids patching upstream's CLI/enums: its pinned `moe_backend`
enum has no TPU fused-EP entry. Leave `--moe-backend=auto` for this model.

| Choice | Values | Default without legacy exports |
| --- | --- | --- |
| `deepseek_v41.execution` | `full`, `ced` | `full` |
| `deepseek_v41.moe` | `gmm`, `fused_ep_w4a8` | `gmm` |
| `engram_config.cpu_offload` | `false`, `true` | `false` (legacy TPU default) |

`fused_ep_w4a8` explicitly selects FP4 weights with FP8 activations; selecting
it is a numeric choice, not a layout optimization. It requires EP and supported
FP4 expert layers. It uses row-padded scales, a sharded routing plan and kernel
height tuning. The old enable flags and deployment token threshold cannot
silently move it to GMM. Unsupported weight/mesh/hardware contracts stop loading
with an error; the generic `gmm` choice remains available. Other models retain
their existing MoE admission behavior.

### V2 producer-side MoE activation quantization

The V2 adapter can pass prequantized FP8 activations and FP32 row scales explicitly
to the routed MoE kernel, skipping local `rowquant_fp8` inside `quantize_local`.

Shifted mHC produces normalized BF16 values together with the routed-MoE FP8
representation (`moe_q`, `moe_scale`) in its epilogue. Router gates retain BF16;
an external gate supplies actual router logits, while a runner-owned gate computes
them once inside the runner.
Routed MoE preserves its original quantization arithmetic: FP32 scale (`amax / 448`),
BF16 reciprocal multiplication, and zero/nonfinite handling. Dispatch accepts the
routed representation directly, preserving token layout, routing exchange, expert
calculation, and output combine.

Shared gate/up projections with runtime e4m3fn weights and FP32 per-output-channel
scales reuse the same FP8 values and row scales, without another activation
quantization. The matmul accumulates/scales in FP32 and restores the original
output dtype before SwiGLU. Down projections still quantize their own post-SwiGLU
inputs. Unsupported shared projections and explicitly different shared inputs
retain their existing BF16 entry path; routed padding never changes the shared
projection's input width.

Shared reuse adopts the producer's rounding and zero/nonfinite handling rather
than the old shared-entry quantizer's arithmetic. It is not bitwise equivalent to
the previous shared path, so full-model accuracy must be checked as well as the
matmul contract.

Both routed-MoE activation variants are built at weight-loading time and cached
per distinct closure configuration, not registered anew for every layer. They share routing,
replica groups and `logical_intermediate_size`, so FP8 input does not discard the
MXFP4 trailing-padding contract.

The Pallas epilogue writes distinct residual, BF16 value, FP8 value and row-scale
outputs from VMEM to HBM. Buffer-reuse and final-drain waits complete those DMAs;
they do not issue additional copies. Prequantization retains the BF16 output and
adds FP8/scale intermediates rather than eliminating all activation traffic.

Preserving the quantization formula does not establish bitwise equivalence of the
surrounding program. Small FP32 mHC gate differences (≤4 ULP) at T288 remain an
unresolved variance.

### Cache ownership

Cache ownership and compression order remain V4.1's. SWA retains BF16 values;
compressed KV uses V4's native FP8 NoPE packing with separate BF16 RoPE storage.
Bindings retain the original tensor owners. Both eager and compiled auxiliary
operations use the canonical 2D `int32[..., 128]` pool; SWA and main attention
derive their native 3D kernel shapes from that owner. Cache operations
derive physical record indices using layout constants validated at binding time.
Like V4, the attention and compressor paths consume the shared TPU
`AttentionMetadata`; model adapters select logical slots from each cache's block
table and compression geometry. The indexer resolves its block table inside the
existing update-and-select operation, avoiding a Torch-side full-context slot
matrix. CED derives only its fixed 128-token SWA window on the Torch side, where
replay masks are already available. Its record conversion stays inside the op.
Frozen record geometry is unpacked into scalar constants at each JAX bridge.
Overlay validation checks both arrays by storage identity within each cache
group; cross-group sharing follows the scheduler's disjoint block ownership.
Engram retains its separate int32 token
history, and the indexer uses native FP8 key packing and scoring.

`ced` deliberately approximates decoder SWA state and is not numerically
equivalent to a full decoder pass for long prompts. It processes all encoder
rows, publishes global KV, and evaluates the decoder prompt tail. Prefix caching
requires upstream SWA bounded replay; PCP must be one. Full forward remains the
quality reference. Existing target compilation buckets are unchanged: a 24-row
bucket is not padded to 48 to obtain a preferred kernel shape.

Sequence parallelism follows upstream's topology policy. Local-Q follows the
actual attention path and its weight/layout contracts. Encoder splitting derives
from the last host-offloaded Engram layer and row ownership; these are not
independent serving switches. The old TP-only SP opt-in has been removed.

## Migration

Remove the selectors listed in [kernel routes](kernel_routes.md). The old
`TPU_DEEPSEEK_V41_CED` and `TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD` exports are
transitional aliases, resolved before worker creation. Legacy fused-EP enable
exports can initialize an omitted policy. Explicit config conflicting with these
aliases fails; matching aliases warn. Remove the aliases after downstream
launchers have migrated rather than maintaining two permanent interfaces.
`TPU_DEEPSEEK_V41_IMPL=v1` is rejected; remove the implementation selector.
Runtime settings such as `LIBTPU_INIT_ARGS` remain environment variables.

TP1/DP>1 requires EP and CPU Engram placement. The existing
`engram_config.dp_shared_memory` controls whether node-local replicas share a
host table or hold/exchange head shards. Shared tables require a shared IPC
namespace and approximately 190 GiB of `/dev/shm` for Flash, plus runtime space.
The host-placement choice does not change the checkpoint's numeric format.

## Benchmarks and CI

- `scripts/dsv41_bench/serve_cached_prefill.sh`: TP1/DP8, CED, CPU Engram,
  fused EP W4A8 and DSpark K5. Trailing serve arguments can override scheduler
  limits or compilation buckets.
- [server steps](../../scripts/dsv41_bench/server_steps/README.md): frozen-input
  cached extend and steady decode; formal measurements exclude profiling.
- `scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep.sh`: DP8 full
  forward smoke and GSM8K. The accuracy floor is inherited from the previous
  TP8 quality gate; it is not a newly measured DP8 baseline.
- `.buildkite/pipeline_dsv41.yml`: CPU policy/route checks, TPU scorer/route
  correctness, DP8 quality and the two DP8 performance scenarios.

The six TP8 serving configs and the old `bench_100k_prefill.sh` /
`integrated_prefill_suite.sh` launchers have been retired. Standalone kernel
benchmarks and generic HTTP/decode clients remain useful and are retained.
DSpark loading uses the draft config/context so EP filtering selects the draft's
expert count rather than the target's.

## Validation limits

Tuning measurements in the kernel tables identify the runtime they were measured
on. The epic upgrade to JAX 0.11.2 / libtpu 0.0.50 / TorchTPU 0.2.0rc1 requires
fresh accelerator validation; old measurements are not evidence for these pins.
The configuration convergence is a draft until its current-head CI, quality and
performance results have been reviewed. Kernel correctness at long context does
not substitute for a 512K Agent evaluation.
