# Environment Variables Reference

All environment variables read by `vllm-torchtpu` are declared and validated in [`src/vllm_torchtpu/envs.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/envs.py) (enforced in CI by `tools/check_env_reads.py`). Boolean variables accept `"1"`, `"true"`, `"yes"`, `"on"` or `"0"`, `"false"`, `"no"`, `"off"` (case-insensitive); invalid values raise a `ValueError` at startup.

---

## 1. Platform, Runtime & Compilation Cache

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | Directory path (or `gs://` URI) for persistent XLA/PJRT compilation cache artifacts. See [Compilation & Warmup Cache](../user_guide/compilation_cache.md). |
| `VLLM_XLA_CHECK_RECOMPILATION` | `False` | Logs or raises when an unexpected XLA graph recompilation occurs after warmup. |
| `TPU_PARALLEL_PRECOMPILE` | `False` | Enables parallel compilation during model warmup. |
| `TPU_MULTIHOST_BACKEND` | `""` (`mp` default, or `"ray"`) | Selects the multi-host execution backend (`"ray"` uses `RayDistributedExecutor`; unset uses native multiprocessing `mp`). See [Multi-Host Serving](../user_guide/multihost.md). |
| `TORCH_TPU_BASE_PORT` | `8070` | Base TCP port for `torch_tpu` worker slice-builder communication. |
| `TORCH_TPU_MP_RENDEZVOUS_PORT` | `None` | Optional explicit TCP port for multi-host `mp` `TCPStore` rendezvous. |
| `TPU_ACCELERATOR_TYPE` | `None` | Explicit accelerator type override (e.g., `"tpu7x"`, `"v6e-8"`) when running offline compilation or CPU tests. |
| `TPU_EVICT_WEIGHTS_PAGE_CACHE` | `False` | With `DefaultModelLoader` (dense and MoE models), drops the host page cache of the checkpoint weight files (`posix_fadvise` `DONTNEED`) once the runner has loaded every model, the draft included, to free host memory (for example for a host KV cache tier). Every rank of this engine's world group first waits until the others have finished loading, then evicts. This covers one engine: it does not coordinate separate engines on a host, such as data-parallel replicas that run as separate engines. The wait is bounded by the CPU group's timeout (`--cpu-distributed-timeout-seconds`; 30 minutes when unset): if a rank never arrives, every rank fails the load within about that time. Every prefetch thread is stopped first: the EP range prefetch (the `prefetch` load strategy with the EP weight filter; stopped with or without this flag) and upstream's whole-file prefetch (the `prefetch` strategy without the filter, or auto-prefetch on NFS and Lustre), which this flag replaces with a stoppable copy. |

---

## 2. Parallelism, Mesh & Scheduling

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DP_SCHED_ENABLED` | `False` | Enables TPU data-parallel request scheduling optimizations. |
| `TPU_PP_DYNAMIC_CHUNKS` | `True` | Dynamically sizes microbatch chunks in Pipeline Parallelism (`--pipeline-parallel-size`). |
| `TPU_PP_CHUNK_SLACK` | `0.1` | Slack ratio for dynamic pipeline-parallel chunk sizing. |
| `TPU_PCP_TOPOLOGY_AWARE_MESH` | `True` | Orders Prefill Context Parallel (`PCP`) ranks along adjacent ICI torus neighbors to minimize ring hop count. |
| `TPU_DCP_MLA_TOPK_CAP` | `-1` (auto: `512` at DCP >= 8, else off) | Caps top-k selection per Decode Context Parallel (`DCP`) rank in Sparse MLA (`0` disables cap). |
| `TPU_DCP_INDEXER_LANE_SPLIT` | `False` | At TP > DCP, splits a large prefill bucket's indexer query tokens over the DCP lanes (which hold replicas of one KV shard) instead of scoring every token on every lane; a lane all-gather restores the top-k. Needs the hist or all-to-all indexer merge and `TPU_DCP_RANK_OPERAND`. |
| `TPU_DCP_INDEXER_LANE_SPLIT_MIN_TOKENS` | `512` | Smallest padded token bucket that takes `TPU_DCP_INDEXER_LANE_SPLIT`. |
| `TPU_DCP_INDEXER_LANE_SPLIT_CHUNK_GATHER` | `False` | With the lane split and the hist merge, all-gathers each top-k chunk's rows over the lanes instead of once at the end. Same result. |
| `TPU_DCP_MLA_TOKEN_SPLIT` | `False` | At TP > DCP, backbone prefill buckets split their sparse MLA query tokens over the DCP lanes instead of the heads (each lane attends its tokens with every head), so the per-token KV gather is not repeated on every lane. Needs the indexer lane split with the hist merge, `TPU_DCP_RANK_OPERAND` and query replication; exclusive with `TPU_DCP_MLA_OPROJ_ALL_REDUCE`. Costs one 64-head q_b, W_UK_T and W_UV per layer of HBM, resident next to the head-split ones (decode keeps those): on GLM-5.2 NVFP4 at TP8 DCP4, about 4.9 GB per rank, or 14% fewer KV cache tokens (2.62M -> 2.25M). |
| `TPU_DCP_MLA_TOKEN_SPLIT_MIN_TOKENS` | `1024` | Smallest padded token bucket that takes `TPU_DCP_MLA_TOKEN_SPLIT`. |
| `TPU_DCP_LSE_MERGE_A2A` | `False` | Uses all-to-all LSE merge for Decode Context Parallelism when token count is below `TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS` (`32`). |
| `TPU_DCP_MERGE_CHUNKED` | `0` | Splits the Decode Context Parallelism reduce-scatter LSE merge of backbone buckets of at least `TPU_DCP_MERGE_CHUNKED_MIN_TOKENS` (`1024`) tokens into this many token chunks, so each chunk's reduce-scatter overlaps the next chunk's compute (`0`/`1` disables). |
| `TPU_TOKEN_BUCKET_EXTRA` | `[]` | Comma-separated list of additional token padding bucket sizes to precompile. |
| `TPU_TOKEN_BUCKET_LINEAR` | `False` | Uses linearly spaced token compilation buckets instead of exponential buckets. |

---

## 3. Mixture-of-Experts (MoE) & Quantization

| Variable | Default | Description |
| :--- | :--- | :--- |
| `USE_MOE_SPARSE_CORE` | `True` | Enables TPU SparseCore offload for MoE token permutation and ragged gather/reduce. |
| `ONEHOT_MOE_PERMUTE_THRESHOLD` | `None` | Token threshold for switching between one-hot matmul and SparseCore sort/permute in MoE routing. |
| `USE_MOE_FUSED_EP_KERNEL` | `False` | Enables the fused Expert-Parallel (`EP`) Pallas MoE kernel. |
| `MOE_FUSED_EP_KERNEL_IMPL` | `None` | Selects a specific fused EP MoE kernel implementation (`"v1"` or `"v2"`). |
| `MOE_FUSED_EP_ENABLE_W4A8` | `False` | Enables 4-bit weight / 8-bit activation (`W4A8`) execution in the fused EP MoE kernel. |
| `TPU_MOE_HIERARCHICAL_EP` | `False` | Enables two-level intra-chip / inter-chip hierarchical Expert Parallelism on dual-core TPU v7x hosts. |
| `TPU_MOE_SEQUENCE_PARALLEL` | `False` | Enables sequence parallelism around MoE blocks. |
| `TPU_MOE_ROUTER_TOPK` | `"rowmax"` | MoE router top-k kernel implementation (`"rowmax"` or standard top-k). |
| `TPU_MOE_FP8_ALL_GATHER` | `False` | Performs MoE activation all-gather in `fp8` before expert computation. |
| `ENABLE_QUANTIZED_MATMUL_KERNEL` | `False` | Dispatches quantized linear layers to the custom Pallas quantized matmul kernel. |
| `REQUANTIZE_WEIGHT_DTYPE` | `"float8_e4m3fn"` | Target weight dtype for load-time linear weight requantization. |
| `REQUANTIZE_BLOCK_SIZE` | `None` | Target subchannel block size for load-time linear weight requantization. |
| `MOE_REQUANTIZE_WEIGHT_DTYPE` | `"float8_e4m3fn"` | Target weight dtype for load-time MoE expert weight requantization (e.g., `"fp4"`). |
| `MOE_REQUANTIZE_BLOCK_SIZE` | `None` | Target block size for load-time MoE expert weight requantization. |
| `VLLM_TPU_MXFP4_DEVICE_DEQUANT` | `True` | Performs MXFP4 weight dequantization on TPU device during load rather than on CPU. |

---

## 4. Attention & KV Cache Layouts

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VLLM_TPU_BLOCK_MAJOR_KV` | `False` | Selects the block-major KV cache memory layout for Ragged Paged Attention. |
| `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL` | `None` | Enables the unified KV cache block pool across hybrid attention layers (e.g., GDN + Full Attention or MLA). |
| `TPU_LATENT_PROJ_INTRA_CHIP_TP` | `False` | Shards MLA latent projections across the 2 TensorCores of a single TPU v7x chip when `TP=1`. |
| `TPU_MLA_MASKED_DENSE_ENABLED` | `False` | Enables the masked-dense MLA attention path for short KV lengths. |
| `TPU_SPARSE_MLA_NOPE_LAYOUT` | `"sparsecore"` | Memory layout for Sparse MLA non-positional (`nope`) KV cache (`"sparsecore"` or `"tc"`). |
| `TPU_SPARSE_MLA_ROPE_LAYOUT` | `"sparsecore"` | Memory layout for Sparse MLA RoPE KV cache (`"sparsecore"` or `"tc"`). |
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
| `USE_PHASED_PROFILER` | `False` | Captures separate XProf traces for prefill and decode phases during a single benchmark run. See [Profiling](../developers_guide/profiling.md). |
| `PYTHON_TRACER_LEVEL` | `1` | Python host call-stack tracing level (`0` = off, `1` = standard, `2` = verbose). |
| `HOST_TRACER_LEVEL` | `2` | XLA runtime host activity tracing level. |
| `DEVICE_TRACER_LEVEL` | `1` | TPU device TensorCore/SparseCore execution tracing level. |
| `TPU_TRACE_MODE` | `"TRACE_COMPUTE"` | SparseCore trace mode (`"TRACE_COMPUTE"`, `"TRACE_COMPUTE_AND_DMA"`, etc.). |
| `TPU_NUM_SPARSE_CORES_TO_TRACE` | `1` | Number of SparseCores to trace when SparseCore tracing is active. |
| `TPU_KERNEL_ITER_MODE` | `False` | Enables fast Pallas kernel hot-reloading without restarting the vLLM server (`TPU_KERNEL_RELOAD_MODULES`). |
