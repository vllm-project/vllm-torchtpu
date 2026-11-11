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
| `TPU_PRECOMPILE_PEER_REPLAY` | `False` | With `TPU_PARALLEL_PRECOMPILE`, each compile range is compiled by one rank of the host and replayed by the others from vLLM's compile cache. Requires the compile cache: startup fails with `VLLM_DISABLE_COMPILE_CACHE=1` (point `VLLM_CACHE_ROOT` at a temporary directory for a cold start with nothing kept across runs). A rank waits up to `TPU_PRECOMPILE_PEER_WAIT_S` for a range's owner to save it. An owner that compiles a range but cannot save it (a failed save, nothing serializable) leaves a `<key>.failed` marker in its compile cache dir, and the waiting ranks then compile the range at once, so the wait only runs out for an owner that died. Markers from an earlier run are ignored. A graph with a rooted or point-to-point collective, or a process group that neither `parallel_state` nor `c10d` resolves, gets fingerprint `None`, so every rank compiles it and logs one `Could not fingerprint graph` warning. |
| `TPU_PRECOMPILE_PEER_WAIT_S` | `600` | With `TPU_PRECOMPILE_PEER_REPLAY`, seconds a rank waits for a range's owner to save it before compiling the range itself (`0`: check once, then compile). Must be finite and `>= 0`. The default leaves room over the biggest Qwen3.5-397B range at DP8 (about 165 s with every rank of the host compiling); raise it if your biggest range compiles in longer, since a waiter that times out compiles concurrently with the owner. |
| `TPU_PRECOMPILE_TRACE_FIRST` | `False` | With `TPU_PARALLEL_PRECOMPILE` and shape-specialized backbone traces, traces every token-bucket group before compiling any bucket, then compiles all buckets across all traces in one pass (incompatible with `VLLM_USE_MEGA_AOT_ARTIFACT=1`). |
| `TPU_MULTIHOST_BACKEND` | `""` (`mp` default, or `"ray"`) | Selects the multi-host execution backend (`"ray"` uses `RayDistributedExecutorV2`; unset uses native multiprocessing `mp`). See [Multi-Host Serving](../user_guide/multihost.md). |
| `TORCH_TPU_BASE_PORT` | `8070` | Base TCP port for `torch_tpu` worker slice-builder communication. |
| `TORCH_TPU_MP_RENDEZVOUS_PORT` | `None` | Optional explicit TCP port for multi-host `mp` `TCPStore` rendezvous. |
| `TPU_ACCELERATOR_TYPE` | `None` | Explicit accelerator type override (e.g., `"tpu7x"`, `"v6e-8"`) when running offline compilation or CPU tests. |
| `TPU_LIBRARY_PATH` | `None` | `libtpu.so` loaded by the TPU runtime. `import libtpu` sets it to the pip package's own library. A custom path (any other file) takes precedence over an installed `libtpu` package: a fingerprint of that file replaces the package version in the compile and Raiden layout cache keys, and the minimum libtpu version check is skipped. |

---

## 2. Parallelism, Mesh & Scheduling

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DP_SCHED_ENABLED` | `False` | Enables TPU data-parallel request scheduling optimizations. |
| `VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS` | `8` | Consecutive `schedule()` steps a running prefill chunk may go without getting at least one KV block of tokens in `SrptPrefillScheduler` (held back or crowded out by shorter prefills) before it moves ahead of the other prefill chunks and is no longer held until it gets a block (`0` disables hold-back). |
| `VLLM_TPU_SRPT_ATTN_WORK_CAP` | `0` | In `SrptPrefillScheduler`, cap on a step's attention work (`sum(n * (c + n / 2))` over scheduled chunks with `n` new tokens after `c` computed ones); while this rank decodes, or another DP rank decodes under EP lockstep (`DP > 1` with expert parallel), prefill chunks are shrunk to stay under it (`0` disables). |
| `VLLM_TPU_SRPT_ATTN_TILE_TOKENS` | `32` | In `SrptPrefillScheduler`, a chunk shrunk by `VLLM_TPU_SRPT_ATTN_WORK_CAP` ends on a multiple of this many tokens, rounded down (the attention kernel's query tile). |
| `VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP` | `False` | Sets `SpeculativeConfig.disable_eagle_block_drop`. With EAGLE or MTP, keeps the trailing prefix-cache block vLLM otherwise drops from each hit and skips the matching one-block Mamba split back-off. This also turns off vLLM's lookahead block hashes, which otherwise keep the block themselves, so hashes stop covering the token after each block. Changes speculative acceptance, not target KV. |
| `TPU_PP_DYNAMIC_CHUNKS` | `True` | Dynamically sizes microbatch chunks in Pipeline Parallelism (`--pipeline-parallel-size`). |
| `TPU_PP_CHUNK_SLACK` | `0.1` | Slack ratio for dynamic pipeline-parallel chunk sizing. |
| `TPU_PP_HANDOFF` | `wave` | How pipeline stages hand activations on: `wave` launches one collective on every stage per forward; `ring` sends each forward's output point to point to the next stage as soon as it is done (needs libtpu 0.0.50 or later with torch_tpu 0.2.0 or later). |
| `TPU_PP_RING_DEPTH` | `2` | With `TPU_PP_HANDOFF=ring`, the forwards a stage may queue for sending before it waits for room; each device holds up to this many outgoing buffers and up to 7 landing buffers for receives posted ahead of their forwards, each `--max-num-batched-tokens` x hidden size in the model dtype, and that memory is reserved before the KV cache is sized. |
| `TPU_PCP_TOPOLOGY_AWARE_MESH` | `True` | Orders Prefill Context Parallel (`PCP`) ranks along adjacent ICI torus neighbors to minimize ring hop count. |
| `USE_TPU_PCP_MANAGER` | `False` | Runs `PCP` through the PCP manager. Only supports sparse MLA models with `DCP` = `PCP` x `TP`. See [PCP with PCP manager](../user_guide/parallelism.md#pcp-with-pcp-manager). |
| `TPU_PCP_KV_WORKSPACE_TOKENS` | `1048576` | With `USE_TPU_PCP_MANAGER=1`, the context tokens one forward's prefills may gather; a step that needs more runs as several forwards. Must be at least `--max-model-len`. |
| `TPU_PCP_KV_CHUNK_TOKENS` | `65536` | With `USE_TPU_PCP_MANAGER=1`, the context tokens each all-gather moves. |
| `TPU_DCP_MLA_TOPK_CAP` | `0` | Caps top-k selection per Decode Context Parallel (`DCP`) rank in Sparse MLA (`0` disables cap). |
| `TPU_DP_VOCAB_PARALLEL` | `False` | Opt-in. Shards the input embedding and lm_head over every DP x TP rank (the EP group) instead of each DP engine's TP group, cutting their HBM per device by DP x. Needs DP with expert parallelism; ignored, with a warning, under pipeline or context parallelism, speculative decoding, LoRA, pooling or multimodal models, or `TPU_GREEDY_TOP_TOKENS`. |
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
| `TPU_EXPERT_PLACEMENT` | `""` | Static expert placement for `EP`: a JSON table (a local path or a `gs://` URI) that sets, per MoE layer, which experts each EP rank holds, as a permutation. Applied at weight load to the experts and the router, so every MoE kernel serves it unchanged. With Ray DP engines, also list it in `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`: vLLM copies only `VLLM_` variables and the names listed there to the engine actors. |
| `TPU_MOE_HIERARCHICAL_EP` | `False` | Enables two-level intra-chip / inter-chip hierarchical Expert Parallelism on dual-core TPU v7x hosts. |
| `TPU_MOE_SEQUENCE_PARALLEL` | `False` | Enables sequence parallelism around MoE blocks. |
| `TPU_ATTN_MOE_OVERLAP` | `False` | Overlaps Qwen3.5 MoE post-attention norm and router gate with TP collectives on token buckets >= 8 MiB per rank. |
| `TPU_MOE_ROUTER_TOPK` | `"rowmax"` | MoE router top-k kernel implementation (`"rowmax"` or standard top-k). |
| `TPU_MOE_FP8_ALL_GATHER` | `False` | Performs MoE activation all-gather in `fp8` before expert computation. |
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
| `VLLM_TPU_BLOCK_MAJOR_KV` | `False` | Selects the block-major KV cache memory layout for Ragged Paged Attention. |
| `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL` | `None` | Enables the unified KV cache block pool across hybrid attention layers (e.g., GDN + Full Attention or MLA). Unset: on for the pooled GDN / KDA architectures, off otherwise. Off on a hybrid model selects the per-layer layout: attention KV blocks plus `max_num_reqs + 1` compact mamba slots (or the uniform layout, which pads every block to the mamba page, when a KV connector other than DecodeBench is configured). On the compact layout `--num-gpu-blocks-override` sets the attention block count and the mamba slots stay compact. |
| `TPU_SKIP_MAMBA_SCHEDULER_BLOCKS` | `True` | On the compact-mamba layout (`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0`) with prefix caching off, GDN layers take no blocks from the attention pool: the runner keeps their state in its own per-request slots. Applies only to GDN layers without speculative decoding or a KV connector; elsewhere it has no effect. Set `0` to restore vLLM's stock accounting (one block per GDN group per request) for debugging. |
| `TPU_LATENT_PROJ_INTRA_CHIP_TP` | `False` | Shards MLA latent projections across the 2 TensorCores of a single TPU v7x chip when `TP=1`. |
| `TPU_MLA_MASKED_DENSE_ENABLED` | `False` | Enables the masked-dense MLA attention path for short KV lengths. |
| `TPU_SPARSE_MLA_NOPE_LAYOUT` | `"sparsecore"` | Memory layout for Sparse MLA non-positional (`nope`) KV cache (`"sparsecore"` or `"tc"`). |
| `TPU_SPARSE_MLA_ROPE_LAYOUT` | `"sparsecore"` | Memory layout for Sparse MLA RoPE KV cache (`"sparsecore"` or `"tc"`). |
| `TPU_SPARSE_MLA_CHUNK_TOKENS` | `None` | Query-token chunk size for Sparse MLA attention (each chunk gathers its tokens' top-k KV rows into one buffer and attends over it; unset keeps the kernel default of `64`). |
| `USE_RPA_PIPELINED_DMA_STAGING` | `False` | Enables pipelined DMA staging in Ragged Paged Attention. |

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
