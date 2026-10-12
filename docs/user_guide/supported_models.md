# Supported Models & Features

`vllm-torchtpu` executes models through vLLM's native PyTorch model runner (`TPUModelRunner`) compiled with `torch.compile(backend="tpu")`, paired with custom JAX/Pallas kernels for attention, Mixture-of-Experts (MoE), and quantized matrix multiplication.

---

## Nightly CI Model Support Matrix

The tables below are generated directly from the repository's nightly CI support matrices (`support_matrices/nightly/`).

=== "TPU v7x (Ironwood)"

    {{ read_csv('support_matrices/nightly/v7x/model_support_matrix.csv') }}

=== "TPU v6e (Trillium)"

    {{ read_csv('support_matrices/nightly/v6e/model_support_matrix.csv') }}

---

## Validated Model Families & Architectures

In addition to the nightly matrix above, the following model families have first-class kernel, quantization, and parallelism support tested in CI and benchmarking suites. Checkpoints listed in `code` are the ones referenced by the repository's benchmark, integration, test, or recipe configurations (`scripts/vllm/benchmarking/configs/`, `scripts/vllm/integration/`, `tests/`, `tests_omni/`, `docs/deployment/`); other members of a family share the same model code and kernels.

| Category | Model Family / Checkpoints | Key TPU Optimizations |
| :--- | :--- | :--- |
| **Frontier MoE & Hybrid Linear Attention** | **Qwen3.5** (`Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-397B-A17B-FP8`, `Qwen/Qwen3.5-35B-A3B-FP8`), **Kimi-Linear-48B** | Gated DeltaNet (GDN) / Kimi Delta Attention (KDA) linear attention + Ragged Paged Attention, unified hybrid KV block pool with prefix caching and host KV offload, SparseCore/GMM Fused MoE, FP8 / W4A8 / MXFP4, TP / DP / EP + PCP / DCP, EAGLE-3 / MTP / DFlash speculative decoding, Raiden P/D disaggregation |
| **Coding & Dense/MoE LLMs** | **Qwen3 / Qwen3-Coder** (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, `nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4`, `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`, `nvidia/Qwen3-30B-A3B-NVFP4`, `Qwen/Qwen3-32B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-0.6B`) | Fused EP MoE kernels, FP8 static/block and NVFP4 quantization, EAGLE-3 & DFlash speculative decoding, structured outputs (`xgrammar`), prefill context parallelism (PCP) |
| **Multi-head Latent Attention (MLA)** | **DeepSeek-V2 / V3 / R1 / V4** (`deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V4-Pro-0813`, `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V2`), **Kimi-K2.5 / K2.6 / K3** (`moonshotai/Kimi-K3`, `moonshotai/Kimi-K2.6`), **GLM-5 / GLM-5.1 / GLM-5.2** (`zai-org/GLM-5.2-FP8`, `nvidia/GLM-5.2-NVFP4`), **Mistral-Large-3** | Pallas MLA v1/v2 & Sparse MLA kernels with SparseCore top-k selection and `sparsecore` / `tc` KV cache layouts, intra-chip latent projection TP, DCP + EP + TP sharding, `deepseek_v4_fp8` W4A8 MoE / MXFP4 / NVFP4 / W4A16, DSpark & DFlash speculative decoding, host KV offload and P/D disaggregation (DeepSeek-V4 and Kimi-K3 integration tests) |
| **Open-Weight Dense & MoE** | **Llama 3 / 3.1 / 3.2 / 3.3 / 4** (`meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`), **GPT-OSS** (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`), **Gemma 2 / 3 / 4** (`google/gemma-4-31B-it`, `RedHatAI/gemma-4-31B-it-FP8-Dynamic`, `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`), **Mixtral** | Ragged Paged Attention v3 (incl. sliding-window / hybrid attention masks), MXFP4 / FP8 / `compressed-tensors` W8A8 quantized matmul, Tensor, Data, Expert & Pipeline Parallelism |
| **Vision-Language & Multimodal** | **Qwen2.5-VL** (`Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`), **Qwen3-VL** (`Qwen/Qwen3-VL-4B`), **Kimi-K2.5 / Kimi-K3 VL**, **Gemma-3 / Gemma-4 VL** (`RedHatAI/gemma-4-31B-it-FP8-Dynamic`, `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`) | 3D M-RoPE, vision encoder FlashAttention, multimodal encoder manager with frame chunking (`MM_ENCODER_FRAME_CHUNK_PATCH_SIZE`), `--language-model-only` text-only serving of VLM checkpoints, CI MMMU-Pro accuracy baselines (Gemma 4) |
| **Embedding, Pooling & Safety Classification** | **Qwen3-VL-Embedding** (`Qwen/Qwen3-VL-Embedding-8B`), **Llama Guard 4** (`meta-llama/Llama-Guard-4-12B`) | TPU-native on-device pooling with request bucketing and static compilation (`examples/qwen3_vl_vllm_embedding.py`), safety classification example and accuracy test on the AILuminate prompt set (`examples/offline_safety_model_inference.py`) |
| **Diffusion & Omni (`vllm-omni`)** | **FLUX.1 / FLUX.2** (`FLUX.1-dev`, `FLUX.1-schnell`, `FLUX.1-Kontext-dev`, `FLUX.2-klein-4B`), **Qwen-Image / Qwen-Image-Edit**, **Wan2.1 / Wan2.2** (video), **LTX-2** (video) | `vllm_omni` TPU platform plugin (`vllm_torchtpu.omni.OmniTpuPlatform`), TPU SDPA diffusion attention backend, PR and nightly pixel-statistics regression tests (`tests_omni/`) |

See the [TPU v7x](../deployment/v7x_recipes.md) and [TPU v6e](../deployment/v6e_recipes.md) serving recipes for the exact slice topologies, parallelism layouts, and `vllm serve` flags used to validate these checkpoints.

---

## Core Feature Support Overview

**Supported** means the feature is merged in `main` and exercised by the repository's unit, integration, or nightly benchmark pipelines on TPU v6e and/or v7x.

| Feature | Status | Details |
| :--- | :--- | :--- |
| **Ragged Paged Attention (RPA v3)** | Supported | Custom Pallas paged attention for MHA/GQA with block-major and HND KV cache layouts (`bf16` and `fp8`), causal / sliding-window / hybrid attention masks, and pipelined DMA staging (`USE_RPA_PIPELINED_DMA_STAGING`). |
| **Multi-head Latent Attention (MLA)** | Supported | Dense MLA v1/v2 and Sparse MLA kernels with SparseCore gather, SparseCore top-k selection, KV compression, masked-dense MLA for short KV lengths, and `sparsecore` / `tc` KV cache layouts (`TPU_SPARSE_MLA_*_LAYOUT`). |
| **Hybrid Linear Attention (GDN / KDA)** | Supported | Pallas chunked-prefill and recurrent-decode kernels for Gated DeltaNet (Qwen3.5) and Kimi Delta Attention (Kimi-Linear, Kimi-K3), with a unified KV block pool shared by linear-attention state and full-attention KV (`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL`). |
| **Parallelism (TP, DP, PP, EP, PCP, DCP)** | Supported | Up to 6D composable parallelism across single-host and multi-host TPU slices, hierarchical EP on dual-core TPU v7x (`TPU_MOE_HIERARCHICAL_EP`), MoE sequence parallelism, and `mp` / Ray multi-host backends. See [Parallelism & Scaling](parallelism.md) and [Multi-Host Serving](multihost.md). |
| **Quantization** | Supported | Unquantized `bf16`, `fp8` (`e4m3fn`) incl. `online_fp8` load-time quantization, `compressed-tensors` (`W8A8`, `W4A8`, `W4A16`), `awq`, `gptq`, `mxfp4`, `nvfp4` (ModelOpt), `deepseek_v4_fp8`, `fp8` KV cache, and load-time MoE expert requantization (`MOE_REQUANTIZE_*`). See [Quantization](quantization.md). |
| **Automatic Prefix Caching (APC)** | Supported | `--enable-prefix-caching` block reuse for full-attention KV and hybrid linear-attention state, prefix-aware request scheduling (`PrefixAwareScheduler`), and prefix-cache correctness integration tests (Qwen3.5, Kimi-K3). |
| **Chunked Prefill & Async Scheduling** | Supported | Chunked prefill with padded token buckets (`--max-num-batched-tokens`, `TPU_TOKEN_BUCKET_*`), `--async-scheduling` to overlap host-side scheduling with TPU execution, and data-parallel request scheduling (`DP_SCHED_ENABLED`). |
| **KV Cache Offloading** | Supported | `TPURaidenOffloadingConnector` tiers KV blocks from HBM to a host DRAM pool and a remote store node, serving external prefix-cache hits; validated by Qwen3.5, DeepSeek-V4, and Kimi-K3 offload integration tests. |
| **Disaggregated Serving (P/D)** | Supported | High-bandwidth KV cache transfer via `TpuRaidenConnector` (`tpu_sync`) with dynamic P/D resharding across PCP / TP / DCP layouts; combinable with KV offloading through `TPUMultiConnector`. See [Disaggregated Serving](disaggregated_serving.md). |
| **Speculative Decoding** | Supported | `ngram`, `suffix`, `mtp` (Multi-Token Prediction), `eagle3`, `dflash`, and `dspark` (parallel drafting for DeepSeek-V4 and Kimi-K3). See [Speculative Decoding](speculative_decoding.md). |
| **Structured Outputs** | Supported | `--structured-outputs-config` with the `xgrammar` backend (JSON schema and grammar constraints) through a TPU `StructuredDecodingManager`; structured-output benchmark configs on `Qwen/Qwen3-0.6B` (`scripts/vllm/benchmarking/configs/qwen3-0.6b-structured-*.sh`). |
| **Multi-LoRA** | Supported | Dynamic LoRA adapter routing (`--enable-lora`). |
| **Multimodal Inputs** | Supported | Image and video inputs for vision-language models (`--limit-mm-per-prompt`), multimodal encoder manager with frame chunking, and `--language-model-only` to serve VLM checkpoints as text-only models. |
| **Embedding & Pooling Models** | Supported | Pooling runner (`PoolerConfig(task="embed")`) with TPU-native on-device pooling, request bucketing, and static compilation for embedding models. |
| **Compilation & Warmup Cache** | Supported | Persistent on-disk XLA (`VLLM_XLA_CACHE_PATH`) and `torch_tpu` warmup caches, explicit `--compilation-config '{"compile_sizes": [...]}'`, parallel precompilation (`TPU_PARALLEL_PRECOMPILE`), and a runtime recompilation detector (`VLLM_XLA_CHECK_RECOMPILATION`). See [Compilation Cache](compilation_cache.md). |
| **Weight Streaming & Loading** | Supported | Checkpoint streaming from `gs://` (`--load-format runai_streamer`), per-rank expert shard download for EP (`--enable-ep-weight-filter`), `--safetensors-load-strategy prefetch`, and host-memory-bounded loading (`--model-loader-extra-config '{"memory_limit": ...}'`). |
| **SparseCore Offload** | Supported | MoE token permutation and ragged gather/reduce on TPU SparseCores (`USE_MOE_SPARSE_CORE`, `ONEHOT_MOE_PERMUTE_THRESHOLD`), Sparse MLA top-k selection, and SparseCore-resident KV cache layouts. |
| **Profiling & Observability** | Supported | XProf device/host traces with phased prefill/decode capture (`USE_PHASED_PROFILER`), SparseCore tracing, Pallas kernel hot-reload for development (`TPU_KERNEL_ITER_MODE`), and standard vLLM Prometheus metrics including local and external prefix-cache hit counters. See [Profiling](../developers_guide/profiling.md). |
| **Diffusion (`vllm-omni`)** | Supported | `OmniTpuPlatform` plugin runs `vllm-omni` image and video diffusion pipelines on TPU with a TPU SDPA attention backend. |

See the [Environment Variables Reference](../reference/env_vars.md) for the full list of tuning knobs behind these features.
