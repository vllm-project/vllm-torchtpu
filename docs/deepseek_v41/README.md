# DeepSeek V4.1 on TorchTPU

The runtime entry is
`vllm_torchtpu.models.vllm.deepseek_v41.v1.model:DeepseekV41ForCausalLM`.
Model construction, decoder order, checkpoint names and cache ownership are
inherited from pinned vLLM v0.31.1rc0.

The retained path targets DeepSeek-V4.1-Flash text generation with MXFP4 routed
experts, FP8 indexer keys, BF16 SWA and packed FP8 compressed KV, TP/EP, PP1 and
DP1. Scheduling is synchronous, without speculation or EPLB. Vision and alternative expert/indexer
formats are outside this implementation.

## Runtime files

| File | Responsibility |
| --- | --- |
| `models/vllm/deepseek_v41/v1/model.py` | Model assembly and compiled encoder/decoder stages |
| `models/vllm/deepseek_v41/v1/replay_plan.py` | Prompt-tail replay planning across scheduler chunks |
| `models/vllm/deepseek_v41/v1/attention.py` | MLA adapter, cache binding and output projection |
| `models/vllm/deepseek_v41/v1/engram.py` | Upstream hashing, request lookbacks and TPU lookup |
| `models/vllm/deepseek_v41/common/mhc.py` | Delayed mHC (shared by v1 and v2): load-time packed mix weights, Pallas prefill pass, compact Sinkhorn and residual flow |
| `models/vllm/deepseek_v41/v1/ops.py` | RoPE tables and tensor rotation |
| `layers/adapter/custom_ops/deepseek_v41/deepseek_v41_cache.py` | Native binding, byte/record and index-page geometry, public slot helpers and cache operations |
| `layers/adapter/custom_ops/deepseek_v41/deepseek_v41_backend.py` | Shared vLLM backend contract, separate from cache layout arithmetic |
| `layers/adapter/custom_ops/deepseek_v41/deepseek_v41_ced_attention_op.py` | Read-only CED attention over the selected SWA window and compressed KV |
| `layers/adapter/custom_ops/deepseek_v41/deepseek_v41_compressor.py` | Upstream compressor/state adapters, C1/C2 pooling and KV packing/publication |
| `layers/adapter/custom_ops/deepseek_v41/deepseek_v41_indexer.py` | Indexer cache adapter, FP8 publication and index selection |
| `layers/adapter/upstream_bindings.py` | Scoped device bindings for reused upstream code |
| `layers/adapter/normalization.py` | Shared Torch RMS normalization with FP32 accumulation |
| `layers/adapter/quantization/deepseek_v41.py` | V4 quantization dispatch and MoE model bindings |
| `runner/kv_cache_dsv4.py` | Shared native cache allocation and V4.1 cache specifications |

Paths in the table are relative to `src/vllm_torchtpu/`.

V4.1 uses the same quantization methods and runtime kernels as the native
TorchTPU V4 model. Dense FP8 weights are dequantized and requantized using the
shared linear configuration (per-channel FP8 by default). MXFP4 expert weights
use the V4 requantizer (`MOE_REQUANTIZE_BLOCK_SIZE`, default 512). The shared
GMM/fused-EP paths own activation quantization, expert weighting and reduction.
Shared experts use the V4 MLP and the MoE runner's combined TP reduction;
`apply_routed_scale_to_output=True` matches the native V4 model. `wo_a` retains
runtime FP8 weights and uses V4's fused inverse-RoPE/FP8 output-projection kernel.
There is no V4.1-specific source-format arithmetic path.

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

For a 40 KiB physical page, indexers pack 256 128-byte FP8 keys into the
32 KiB NoPE array. Their E8M0 scale bytes occupy the low byte of 256 int32
words in the existing 8 KiB RoPE array. The upper bytes of written scale words
are zero; the remaining companion capacity is padding. This gives 160 allocated
bytes per key while preserving the common physical block budget and the original
array bindings. C1 index pages cover 256 tokens and C2 pages cover 512 tokens.
Main-attention NoPE/RoPE packing and indexer quantization/scoring are unchanged.

Hierarchical reindex builds a compact key-record list before scoring. For a
fixed candidate pool, later indexers' key reads and QK work are bounded by that
pool, including tile padding. The first indexer still scores the full visible
context. Candidate sorting, duplicate removal, short-context bypass and
underfull top-k padding preserve the full-score path's selected logical rows.

`TPU_DEEPSEEK_V41_CED=1` enables decoder SWA bounded replay. The encoder processes
every scheduled token and publishes decoder global KV before selecting decoder
rows. Only the final SWA window of the complete prompt enters the decoder;
the window starts at the same position across all prefill chunks and subsequent
decode steps. The two stages compile and dispatch with independent token
buckets. Mixed batches retain request block-table ownership and restore output
rows to the runner's original layout.

Bounded replay deliberately approximates decoder SWA state; it is not
numerically equivalent to a full decoder pass for long prompts. The default
remains the full-forward reference until matched quality and performance
validation is complete. Bounded replay requires unsharded token rows
(PCP1, no sequence parallelism); existing PP1/DP1 and speculation restrictions
also apply.

With prefix caching, the v2 implementation follows upstream's SWA bounded
replay contract (`cache_config.swa_bounded_replay`, on by default). The SWA
group leaves prefix caching, and the scheduler resumes each hit one window
(128 tokens) early. Those replayed tokens rebuild SWA state:

- SWA attention reads no row below the replay start, in both the encoder and
  the decoder;
- compressed KV and indexer rows of replayed tokens stay as cached;
- the decoder window starts at the later of the prompt tail and the replay
  start.

A preempted request resuming past its prompt reads its Engram lookback from
committed tokens on that step. v1, or v2 with `swa_bounded_replay` off, still
falls back to the full-forward path with prefix caching.

The shared runner changes provide only registration, native-cache binding,
metadata, Engram lookbacks and compilation-cache invalidation. The vLLM version
bump also requires the small NVFP4 linear schema adaptation imported by the
shared quantization registry; it does not select NVFP4 for this model.

## Launch and validation

Use the existing benchmark flow with
`scripts/vllm/benchmarking/configs/deepseek-v41-flash-tp8-ep.sh`.
That configuration sets the V4.1 gather workaround explicitly, without changing
other models' automatic libtpu flags.

v2 DP (TP1, EP, `TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD=1`) requires
`engram_config.dp_shared_memory`, which vLLM enables by default for DP. The DP
replicas on a node map one copy of each host Engram table from `/dev/shm`, and
each replica looks up only its own tokens, so a step has no Engram DP
collective. `/dev/shm` must hold every table (about 190 GiB for
DeepSeek-V4.1-Flash) in addition to other users such as the compilation cache.
When it cannot, vLLM disables sharing and the model refuses to start.

Validation scripts and raw experiment evidence are maintained separately from
the runtime change. Cache validation alone does not establish full-model
numerical equivalence, memory usage or serving performance.

DeepSeek V4.1 benchmark configurations and the standalone prefill scripts
now enable prefix caching by default. Set `ENABLE_PREFIX_CACHING=false` for
uncached baseline measurements. The shared benchmark runner uses its normal
prefix-cache switch; no custom scheduler is required. Repeated prompts may
hit cached prefixes, so these measurements are not interchangeable with the
previous uncached baselines. Chunk size, concurrency, KV budget, loading and
Engram offload settings remain specific to each benchmark configuration.
The DSpark configuration inherits this default; the combination of DSpark
and prefix caching still requires separate validation.
