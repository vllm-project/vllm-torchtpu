# Environment Variables Reference

All environment variables read by `vllm-torchtpu` are declared and validated in [`src/vllm_torchtpu/envs.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/envs.py) (enforced in CI by `tools/check_env_reads.py`). Boolean variables accept `"1"`, `"true"`, `"yes"`, `"on"` or `"0"`, `"false"`, `"no"`, `"off"` (case-insensitive); invalid values raise a `ValueError` at startup.

At startup, `vllm-torchtpu` reports set environment variables that neither vLLM nor this package reads, and suggests the closest name for a likely typo. `--fail-on-environ-validation` turns the report into an error.

---

## 1. Platform, Runtime & Compilation Cache

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | Directory path (or `gs://` URI) for persistent XLA/PJRT compilation cache artifacts. See [Compilation & Warmup Cache](../user_guide/compilation_cache.md). |
| `VLLM_XLA_CHECK_RECOMPILATION` | `False` | Logs or raises when an unexpected XLA graph recompilation occurs after warmup. |
| `TPU_PARALLEL_PRECOMPILE` | `False` | Enables parallel compilation during model warmup. |
| `TPU_PRECOMPILE_PEER_REPLAY` | `False` | With `TPU_PARALLEL_PRECOMPILE`, each compile range is compiled by one rank of the host and replayed by the others from vLLM's compile cache. Requires the compile cache: startup fails with `VLLM_DISABLE_COMPILE_CACHE=1` (point `VLLM_CACHE_ROOT` at a temporary directory for a cold start with nothing kept across runs). A rank waits up to `TPU_PRECOMPILE_PEER_WAIT_S` for a range's owner to save it. An owner that compiles a range but cannot save it (a failed save, nothing serializable) leaves a `<key>.failed` marker in its compile cache dir, and the waiting ranks then compile the range at once, so the wait only runs out for an owner that died. Markers from an earlier run are ignored. |
| `TPU_PRECOMPILE_PEER_WAIT_S` | `600` | With `TPU_PRECOMPILE_PEER_REPLAY`, seconds a rank waits for a range's owner to save it before compiling the range itself (`0`: check once, then compile). Must be finite and `>= 0`. The default leaves room over the biggest Qwen3.5-397B range at DP8 (about 165 s with every rank of the host compiling); raise it if your biggest range compiles in longer, since a waiter that times out compiles concurrently with the owner. |
| `TPU_PRECOMPILE_TRACE_FIRST` | `False` | With `TPU_PARALLEL_PRECOMPILE` and shape-specialized backbone traces, traces every token-bucket group before compiling any bucket, then compiles all buckets across all traces in one pass (incompatible with `VLLM_USE_MEGA_AOT_ARTIFACT=1`). |
| `TPU_MULTIHOST_BACKEND` | `""` (`mp` default, or `"ray"`) | Selects the multi-host execution backend (`"ray"` uses `RayDistributedExecutor`; unset uses native multiprocessing `mp`). See [Multi-Host Serving](../user_guide/multihost.md). |
| `TORCH_TPU_BASE_PORT` | `8070` | Base TCP port for `torch_tpu` worker slice-builder communication. |
| `TORCH_TPU_MP_RENDEZVOUS_PORT` | `None` | Optional explicit TCP port for multi-host `mp` `TCPStore` rendezvous. |
| `TPU_ACCELERATOR_TYPE` | `None` | Explicit accelerator type override (e.g., `"tpu7x"`, `"v6e-8"`) when running offline compilation or CPU tests. |
| `TPU_LIBRARY_PATH` | `None` | `libtpu.so` loaded by the TPU runtime. `import libtpu` sets it to the pip package's own library. A custom path (any other file) takes precedence over an installed `libtpu` package: a fingerprint of that file replaces the package version in the compile and Raiden layout cache keys, and the minimum libtpu version check is skipped. |
| `TPU_NORM_WEIGHTS_FP32` | `True` | Stores learned RMSNorm weights in float32 at load, removing a per-norm weight convert from every step. Outputs are unchanged where every norm goes through the TPU `rms_norm` providers (verified on Gemma4); code that reads norm weights directly sees float32. |

---

## 2. Parallelism, Mesh & Scheduling

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DP_SCHED_ENABLED` | `False` | Enables TPU data-parallel request scheduling optimizations. |
| `VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS` | `8` | Consecutive `schedule()` steps a running prefill chunk may go without getting at least one KV block of tokens in `SrptPrefillScheduler` (held back or crowded out by shorter prefills) before it moves ahead of the other prefill chunks and is no longer held until it gets a block (`0` disables hold-back). |
| `VLLM_TPU_SRPT_ATTN_WORK_CAP` | `0` | In `SrptPrefillScheduler`, cap on a step's attention work (`sum(n * (c + n / 2))` over scheduled chunks with `n` new tokens after `c` computed ones); while this rank decodes, or another DP rank decodes under EP lockstep (`DP > 1` with expert parallel), prefill chunks are shrunk to stay under it (`0` disables). |
| `VLLM_TPU_SRPT_ATTN_TILE_TOKENS` | `32` | In `SrptPrefillScheduler`, a chunk shrunk by `VLLM_TPU_SRPT_ATTN_WORK_CAP` ends on a multiple of this many tokens, rounded down (the attention kernel's query tile). |
| `VLLM_TPU_SOLO_PREFILL_OVER_TOKENS` | `8192` | Installs `PrefillFirstScheduler`. A waiting request with more than this many tokens to compute (none computed yet, no encoder inputs) is prefilled whole in a step of its own, without reading the prefix cache, while running requests sit that step out; every other step schedules at most this many tokens and admits no waiting request from the first such prompt onward. For context-parallel programs keyed on the padded token bucket: the bucket holding this many tokens must be below their threshold and the next bucket at or above it. Prompts longer than the step budget (`--max-num-batched-tokens`) are prefilled in regular chunks. |
| `VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE` | `0` | In `PrefillFirstScheduler`, solo prefill steps in a row while requests are running (`0`: no limit). |
| `VLLM_TPU_SOLO_PREFILL_DECODE_STEPS` | `1` | In `PrefillFirstScheduler`, regular steps that schedule tokens between two runs of `VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE` solo steps. |
| `TPU_PP_DYNAMIC_CHUNKS` | `True` | Dynamically sizes microbatch chunks in Pipeline Parallelism (`--pipeline-parallel-size`). |
| `TPU_PP_CHUNK_SLACK` | `0.1` | Slack ratio for dynamic pipeline-parallel chunk sizing. |
| `TPU_PP_HANDOFF` | `wave` | How pipeline stages hand activations on: `wave` launches one collective on every stage per forward; `ring` sends each forward's output point to point to the next stage as soon as it is done (needs libtpu 0.0.50 or later with torch_tpu 0.2.0 or later). |
| `TPU_PP_RING_DEPTH` | `2` | With `TPU_PP_HANDOFF=ring`, the forwards a stage may queue for sending before it waits for room; each device holds up to this many outgoing buffers and up to 7 landing buffers for receives posted ahead of their forwards, each `--max-num-batched-tokens` x hidden size in the model dtype, and that memory is reserved before the KV cache is sized. |
| `TPU_PCP_TOPOLOGY_AWARE_MESH` | `True` | Orders Prefill Context Parallel (`PCP`) ranks along adjacent ICI torus neighbors to minimize ring hop count. |
| `TPU_DCP_MLA_TOPK_CAP` | `0` | Caps top-k selection per Decode Context Parallel (`DCP`) rank in Sparse MLA (`0` disables cap). |
| `TPU_DCP_LSE_MERGE_A2A` | `False` | Uses all-to-all LSE merge for Decode Context Parallelism when token count is below `TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS` (`64`). |
| `TPU_TOKEN_BUCKET_EXTRA` | `[]` | Comma-separated list of additional token padding bucket sizes to precompile. |
| `TPU_TOKEN_BUCKET_LINEAR` | `False` | Uses linearly spaced token compilation buckets instead of exponential buckets. |

---

## 3. Mixture-of-Experts (MoE) & Quantization

| Variable | Default | Description |
| :--- | :--- | :--- |
| `USE_MOE_SPARSE_CORE` | `True` | Enables TPU SparseCore offload for MoE token permutation and ragged gather/reduce. The SparseCore kernels are validated on TPU v5p, v6e, 7, and 7x; other chips fall back to TensorCore. |
| `ONEHOT_MOE_PERMUTE_THRESHOLD` | `None` | Token threshold for switching between one-hot matmul and SparseCore sort/permute in MoE routing. |
| `USE_MOE_FUSED_EP_KERNEL` | `False` | Enables the fused Expert-Parallel (`EP`) Pallas MoE kernel. |
| `MOE_FUSED_EP_KERNEL_IMPL` | `None` | Selects a specific fused EP MoE kernel implementation (`"v1"` or `"v2"`). |
| `MOE_FUSED_EP_ENABLE_W4A8` | `False` | Enables 4-bit weight / 8-bit activation (`W4A8`) execution in the fused EP MoE kernel. |
| `MOE_FUSED_EP_SORTED_GATHER` | `False` | Keeps the fused EP MoE mesh in EP-rank order and performs the activation all-gather outside the kernel over a process group in ascending device-id order. |
| `FUSED_MOE_DECODE_KERNEL` | `False` | Serves eligible MoE layers (TP experts, or EP with TP attention) up to `FUSED_MOE_DECODE_KERNEL_MAX_TOKENS` tokens with the fused MoE decode Pallas kernel (routed experts, shared expert, and all-reduce in one call); arming refuses a cap whose buffers do not fit in VMEM. |
| `FUSED_MOE_DECODE_KERNEL_MAX_TOKENS` | `256` | Maximum token count per step for `FUSED_MOE_DECODE_KERNEL`; larger steps fall back to the grouped-matmul MoE path. |
| `USE_MOE_TP_DECODE_KERNEL` | `False` | Serves eligible tensor-parallel FP8 MoE layers with a fused TP decode MoE Pallas kernel (in-kernel routing, shared expert, and all-reduce; requires `TPU_MOE_SKIP_PADDED_TOKENS=1`, and arms nothing while `FUSED_MOE_DECODE_KERNEL` is set). |
| `TPU_MOE_HIERARCHICAL_EP` | `False` | Enables two-level intra-chip / inter-chip hierarchical Expert Parallelism on dual-core TPU v7x hosts. |
| `TPU_MOE_SEQUENCE_PARALLEL` | `False` | Enables sequence parallelism around MoE blocks. |
| `TPU_ATTN_MOE_OVERLAP` | `False` | Overlaps Qwen3.5 MoE post-attention norm and router gate with TP collectives on token buckets >= 8 MiB per rank. |
| `TPU_GEMMA4_SEQUENCE_PARALLEL` | `True` | Gemma4 at TP > 1: token buckets of >= 1 MiB of residual stream per rank keep it token-sharded (reduce-scatter instead of all-reduce, norms on T/tp rows, FP8 all-gather into the next projection). Not with drafters that read auxiliary hidden states (EAGLE3, DFlash, DSpark). |
| `TPU_GEMMA4_SP_MICROBATCHES` | `1` | With `TPU_GEMMA4_SEQUENCE_PARALLEL`: run a large bucket as this many contiguous token chunks, one dataflow each outside attention, so one chunk's collectives can overlap another's projections. Applied only while every chunk keeps >= 1 MiB per rank. |
| `TPU_MOE_ROUTER_TOPK` | `"rowmax"` | MoE router top-k kernel implementation (`"rowmax"` or standard top-k). |
| `TPU_MOE_FP8_ALL_GATHER` | `False` | Performs MoE activation all-gather in `fp8` before expert computation. |
| `TPU_GEMMA4_TOPK_SOFTMAX_ROUTING` | `True` | Gemma4 MoE routing as a softmax over the top-k router logits, with `per_expert_scale` selected without a gather, instead of upstream's full softmax, one-hot renormalization and two gathers. Same top-k ids; weights match to float32 rounding for float32 logits. |
| `TPU_MOE_GATHER_GMM` | `True` | Runs FP8 per-channel MoE layers (EP or not) with the gather-fused expert kernels (`kernels/megablox/gather_gmm.py`): the expert kernel gathers token rows itself and the combine gathers each token's top-k rows, so no permuted `[tokens * topk, hidden]` copy reaches HBM. Same per-row expert arithmetic as `gmm_v2`; the top-k sum is F32 (outputs within one BF16 ulp of the grouped-matmul path). Batches below `TPU_MOE_GATHER_GMM_MIN_TOKENS` and unsupported shapes keep the grouped-matmul path. |
| `TPU_MOE_GATHER_GMM_MIN_TOKENS` | `1024` | Smallest token bucket `TPU_MOE_GATHER_GMM` applies to; decode buckets below it keep the grouped-matmul path. |
| `TPU_MOE_TP_EXPERT_SPLIT` | `False` | MoE layers without `--enable-expert-parallel` on a TP group (no DP/PCP/SP inside the engine): each TP rank holds whole experts (EP over the TP group) instead of a slice of every expert, and computes only the routes to them; DP replicas stay independent. Changes the MoE arithmetic (each expert's down projection runs over its whole intermediate on one rank), so it needs the tolerance and accuracy tiers. Cannot be combined with `TPU_MOE_HIERARCHICAL_EP`. |
| `TPU_GEMMA4_TOPK_SOFTMAX_ROUTING` | `True` | Routes Gemma4 MoE layers with a softmax over the top-k router logits times `per_expert_scale`, selected without a gather (`models/vllm/gemma4_routing_patch.py`), instead of upstream's full softmax, one-hot renormalization and two gathers. Same top-k ids; weights equal up to F32 rounding for the F32 logits the TPU gate emits. |
| `ENABLE_QUANTIZED_MATMUL_KERNEL` | `False` | Dispatches quantized linear layers to the custom Pallas quantized matmul kernel. |
| `QUANTIZE_ON_LOAD_PREFIXES` | `[]` | Comma-separated dot-bounded module prefix names to quantize to FP8 on load (e.g., `"self_attn,shared_experts,layers.0.mlp"`), including `VocabParallelEmbedding` and `ParallelLMHead`. |
| `REQUANTIZE_WEIGHT_DTYPE` | `"float8_e4m3fn"` | Target weight dtype for load-time linear weight requantization. |
| `REQUANTIZE_BLOCK_SIZE` | `None` | Target subchannel block size for load-time linear weight requantization. |
| `TPU_FP8_LINEAR_W8A16_MAX_ROWS` | `0` | Opt-in. FP8 dense-linear steps whose padded token bucket has at most this many rows keep bf16 activations (W8A16) instead of quantizing them to FP8. Per-channel path with FP8 weights only; measured on v7x. |
| `TPU_GDN_QKVZ_KERNEL_LAYOUT` | `False` | Opt-in. Stores the GDN `in_proj_qkvz` weight in the matmul's (k, n) layout instead of vLLM's [n_out, n_in], removing a per-step weight copy on the per-block FP8 path. Ignored with prefill context parallelism; the experimental AOT export rejects it at TP > 1. |
| `MOE_REQUANTIZE_WEIGHT_DTYPE` | `"float8_e4m3fn"` | Target weight dtype for load-time MoE expert weight requantization (e.g., `"fp4"`). |
| `MOE_REQUANTIZE_BLOCK_SIZE` | `None` | Target block size for load-time MoE expert weight requantization. |
| `VLLM_TPU_MXFP4_DEVICE_DEQUANT` | `True` | Performs MXFP4 weight dequantization on TPU device during load rather than on CPU. |

---

## 4. Attention & KV Cache Layouts

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TPU_FP8_ATTENTION` | `True` | Batched RPA (`--attention-backend CUSTOM`) runs the mixed (prefill) kernel's QK^T and PV as fp8 x fp8 matmuls on layers with an e4m3 KV cache and a multiple of 4 query heads per KV head (Gemma4's full-attention layers): queries are cast to the KV dtype at scale 1.0 and the softmax probabilities are cast as well. The decode kernel is unchanged. Changes numerics (tolerance level). |
| `TPU_RPA_SLIDING_PREFILL_MAX_BKV` | `0` | Opt-in. Largest batched-RPA mixed-kernel KV and query block (tokens) on sliding-window layers, rounded down to the block stride and no smaller than the default minimum block. The VMEM-filling default (1024 tokens on Gemma4-31B at TP8, 1536 on Gemma4-26B) computes 2-4x the window per query. 0 keeps the default. Changes the online-softmax block order. |
| `VLLM_TPU_BLOCK_MAJOR_KV` | `False` | Selects the block-major KV cache memory layout for Ragged Paged Attention. |
| `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL` | `None` | Enables the unified KV cache block pool across hybrid attention layers (e.g., GDN + Full Attention or MLA). |
| `TPU_SKIP_MAMBA_SCHEDULER_BLOCKS` | `True` | On the compact-mamba layout (`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0`) with prefix caching off, GDN layers take no blocks from the attention pool: the runner keeps their state in its own per-request slots. Applies only to GDN layers without speculative decoding or a KV connector; elsewhere it has no effect. Set `0` to restore vLLM's stock accounting (one block per GDN group per request) for debugging. |
| `TPU_LATENT_PROJ_INTRA_CHIP_TP` | `False` | Shards MLA latent projections across the 2 TensorCores of a single TPU v7x chip when `TP=1`. |
| `TPU_MLA_MASKED_DENSE_ENABLED` | `False` | Enables the masked-dense MLA attention path for short KV lengths. |
| `TPU_SPARSE_MLA_NOPE_LAYOUT` | `"sparsecore"` | Memory layout for Sparse MLA non-positional (`nope`) KV cache (`"sparsecore"` or `"tc"`). |
| `TPU_SPARSE_MLA_ROPE_LAYOUT` | `"sparsecore"` | Memory layout for Sparse MLA RoPE KV cache (`"sparsecore"` or `"tc"`). |
| `TPU_SPARSE_MLA_CHUNK_TOKENS` | `None` | Query-token chunk size for Sparse MLA attention (each chunk gathers its tokens' top-k KV rows into one buffer and attends over it; unset keeps the kernel default of `64`). |
| `USE_RPA_PIPELINED_DMA_STAGING` | `False` | Enables pipelined DMA staging in Ragged Paged Attention. |
| `TPU_RPA_SKIP_EMPTY_LAUNCH` | `True` | Batched RPA (`--attention-backend CUSTOM`) does no work for a decode or mixed kernel whose batch segment is empty: the kernel returns at entry, its schedule writes only the step count, and on buckets of at most `max_num_seqs` (times the decode query size) tokens an XLA conditional skips the launch. Outputs are unchanged. |
| `TPU_RPA_FENCE_OUTPUT` | `True` | Batched RPA (`--attention-backend CUSTOM`) materializes its token-major output behind an XLA optimization barrier. Without it XLA fuses the head-major to token-major transpose into the output's consumer (the o_proj's FP8 activation quantization) and moves the attention output twice in f32. Other RPA kernels, including PCP streaming attention, are not fenced. Outputs are unchanged. |
| `TPU_GEMMA4_FUSED_ATTN_PROLOGUE` | `True` | Gemma4 attention hands its qkv projection's rows to the batched RPA backend (`--attention-backend CUSTOM`), which runs the per-head q/k/v RMSNorms, the neox rotary, the FP8 KV quantization and the kernel's input layouts in one Pallas pass. Needs an FP8 e4m3 KV cache in the NHD layout; refuses sinks, KV sharing, PCP/DCP and the block-major pool. The norms' sums of squares are added in another order, so values can move one rounding step. |
| `TPU_RPA_CAP_SLIDING_DECODE_BLOCK` | `True` | Caps the batched-RPA decode KV block of sliding-window layers at half the window (rounded down to the block stride) instead of filling VMEM (1792 tokens for Gemma4's 1024-token window), so decode launches zero-fill and compute less. Changes the online-softmax block order. |
| `TPU_REUSE_STAGED_BLOCK_TABLES` | `True` | Reuses the previous step's staged device block tables when every staged row is unchanged (most decode steps), skipping the host-side pack, upload and split. Not applied to dummy runs or mamba groups. |

---

## 5. Disaggregated Serving (Raiden & KV Transfer)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TPU_USE_RAIDEN_KV_CACHE_MANAGER` | `False` | Enables the Raiden KV cache manager for Prefill/Decode disaggregation. |
| `TPU_RAIDEN_CONTROLLER_ADDRESS` | `""` | Address (`host:port`) of the Raiden P/D coordinator (`run_raiden_controller.py`). |
| `TPU_RAIDEN_JOB_NAME` | `""` | Role name (`"prefill"` or `"decode"`) registered with the Raiden controller. |
| `TPU_RAIDEN_ENGINE_ID` | `"0"` | Engine replica identifier for Raiden KV transfer. |
| `TPU_RAIDEN_TRANSFER_PARALLELISM` | `8` | Number of parallel DMA transfer streams in `TpuRaidenConnector`. |
| `TPU_RAIDEN_RESHARD_IMPL` | `"controller"` | Selects `"controller"` or `"host"` KV cache resharding when Prefill and Decode topologies differ. |
| `TPU_KV_RESHARD_TRANSPORT` | `"zmq"` | Transport protocol (`"zmq"` or `"tcp"`) for host-side KV resharding. |
| `TPU_KV_TRANSFER_PORT` | `"9100"` | Port used by `TPUConnector` for KV cache transfer. |
| `TPU_SIDE_CHANNEL_PORT` | `"9600"` | Port used by `TPUConnector` for side-channel handshake metadata. |

---

## 6. Profiling & Developer Debugging

| Variable | Default | Description |
| :--- | :--- | :--- |
| `USE_PHASED_PROFILER` | `False` | Captures separate XProf traces for prefill and decode phases during a single benchmark run. See [Profiling](../developers_guide/profiling/phased.md). |
| `PYTHON_TRACER_LEVEL` | `1` | Python host call-stack tracing level (`0` = off, `1` = standard, `2` = verbose). |
| `HOST_TRACER_LEVEL` | `2` | XLA runtime host activity tracing level. |
| `DEVICE_TRACER_LEVEL` | `1` | TPU device TensorCore/SparseCore execution tracing level. |
| `TPU_TRACE_MODE` | `"TRACE_COMPUTE"` | SparseCore trace mode (`"TRACE_COMPUTE"`, `"TRACE_COMPUTE_AND_DMA"`, etc.). |
| `TPU_NUM_SPARSE_CORES_TO_TRACE` | `1` | Number of SparseCores to trace when SparseCore tracing is active. |
| `TPU_KERNEL_ITER_MODE` | `False` | Enables fast Pallas kernel hot-reloading without restarting the vLLM server (`TPU_KERNEL_RELOAD_MODULES`). |
