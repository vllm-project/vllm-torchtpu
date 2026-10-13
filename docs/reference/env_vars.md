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

---

## 2. Parallelism, Mesh & Scheduling

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DP_SCHED_ENABLED` | `False` | Enables TPU data-parallel request scheduling optimizations. |
| `VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS` | `8` | Maximum consecutive `schedule()` steps a running prefill chunk may be held back in `SrptPrefillScheduler` before it is guaranteed a step to advance (`0` disables hold-back). |
| `TPU_PP_DYNAMIC_CHUNKS` | `True` | Dynamically sizes microbatch chunks in Pipeline Parallelism (`--pipeline-parallel-size`). |
| `TPU_PP_CHUNK_SLACK` | `0.1` | Slack ratio for dynamic pipeline-parallel chunk sizing. |
| `TPU_PCP_TOPOLOGY_AWARE_MESH` | `True` | Orders Prefill Context Parallel (`PCP`) ranks along adjacent ICI torus neighbors to minimize ring hop count. |
| `TPU_DCP_MLA_TOPK_CAP` | `0` | Caps top-k selection per Decode Context Parallel (`DCP`) rank in Sparse MLA (`0` disables cap). |
| `TPU_DCP_LSE_MERGE_A2A` | `False` | Uses all-to-all LSE merge for Decode Context Parallelism when token count is below `TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS` (`64`). |
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
| `USE_PHASED_PROFILER` | `False` | Captures separate XProf traces for prefill and decode phases during a single benchmark run. See [Profiling](../developers_guide/profiling/phased.md). |
| `PYTHON_TRACER_LEVEL` | `1` | Python host call-stack tracing level (`0` = off, `1` = standard, `2` = verbose). |
| `HOST_TRACER_LEVEL` | `2` | XLA runtime host activity tracing level. |
| `DEVICE_TRACER_LEVEL` | `1` | TPU device TensorCore/SparseCore execution tracing level. |
| `TPU_TRACE_MODE` | `"TRACE_COMPUTE"` | SparseCore trace mode (`"TRACE_COMPUTE"`, `"TRACE_COMPUTE_AND_DMA"`, etc.). |
| `TPU_NUM_SPARSE_CORES_TO_TRACE` | `1` | Number of SparseCores to trace when SparseCore tracing is active. |
| `TPU_KERNEL_ITER_MODE` | `False` | Enables fast Pallas kernel hot-reloading without restarting the vLLM server (`TPU_KERNEL_RELOAD_MODULES`). |
