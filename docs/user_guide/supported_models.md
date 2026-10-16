# Supported Models & Features

`vllm-torchtpu` executes models through vLLM's native PyTorch model runner (`TPUModelRunner`) compiled with `torch.compile(backend="tpu")`, paired with custom JAX/Pallas kernels for attention, Mixture-of-Experts (MoE), and quantized matrix multiplication.

This page has two sections:

- [Validated Models](#validated-models) — Models running on TPU, which checkpoints are tested, and in which way.
- [Feature Support](#feature-support) — Supported features, how to turn them on, and which models they have been validated with.

Validation levels used in both tables:

| Level | Meaning |
| :--- | :--- |
| ✅ **CI** | Runs on every pull request (performance / accuracy evaluation or integration pipeline on TPU v7x) |
| 🌙 **Nightly** | Runs in the scheduled nightly pipeline |
| 🧪 **Tests** | Covered by unit, kernel, or component tests only (no end-to-end serving run in CI) |
| 📝 **Example** | Example script, benchmark config, or recipe exists but is not scheduled in CI |

---

## Validated Models

Click a family to see its architecture-specific TPU support.

<div class="expandable-table" data-detail-column="Architecture-specific TPU support" markdown>

| Family | Validated checkpoints | Architecture-specific TPU support |
| :--- | :--- | :--- |
| **Qwen3.5**<br><span class="family-desc">hybrid GDN + full attention, MoE, MTP heads</span> | <ul><li>🌙 `Qwen/Qwen3.5-397B-A17B-FP8`</li><li>✅ `Qwen/Qwen3.5-35B-A3B-FP8`</li><li>✅ `Qwen/Qwen3.5-4B` (MTP tests)</li></ul> | Gated DeltaNet chunked-prefill and recurrent-decode Pallas kernels; unified KV block pool shared by GDN state and full-attention KV; native MTP drafting |
| **Kimi-Linear**<br><span class="family-desc">hybrid KDA + MLA</span> | <ul><li>🧪 `Kimi-Linear-48B`</li></ul> | Kimi Delta Attention kernels (fused conv1d + delta rule) with pooled KDA state |
| **Qwen3 / Qwen3-Coder**<br><span class="family-desc">dense and MoE</span> | <ul><li>🌙 `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`</li><li>✅ `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`</li><li>✅ `nvidia/Qwen3-30B-A3B-NVFP4`</li><li>✅ `Qwen/Qwen3-0.6B`</li><li>✅ `Qwen/Qwen3-4B` (DFlash / DSpark tests)</li><li>📝 `nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4`</li></ul> | Reference dense / MoE path: Ragged Paged Attention v3, fused EP MoE (GMM / SparseCore), FP8 and NVFP4 checkpoints |
| **DeepSeek-V2 / V4**<br><span class="family-desc">MLA, sparse attention, MoE</span> | <ul><li>🌙 `deepseek-ai/DeepSeek-V4-Pro-0813`</li><li>🌙 `deepseek-ai/DeepSeek-V4-Flash`</li><li>✅ `deepseek-ai/DeepSeek-V2`</li></ul> | Dedicated DeepSeek-V4 model implementation; Sparse MLA with SparseCore top-k selection and `sparsecore` / `tc` KV layouts; `deepseek_v4_fp8` FP8 + W4A8 MoE path; DSpark drafting |
| **Kimi-K2.6 / K3**<br><span class="family-desc">MLA; K3 adds KDA layers and the MoonViT vision tower</span> | <ul><li>✅ `moonshotai/Kimi-K3` (expert-pruned variant in PR CI, full model in on-demand v7x-32 sweeps)</li><li>🌙 `moonshotai/Kimi-K2.6`</li></ul> | Dedicated Kimi-K3 implementation (MLA + KDA layers, MoonViT encoder); W4A16 Kimi-K2.6; DSpark drafting |
| **GLM-5.2**<br><span class="family-desc">MLA, MoE</span> | <ul><li>🌙 `zai-org/GLM-5.2-FP8`</li><li>🌙 `nvidia/GLM-5.2-NVFP4`</li></ul> | Calibrated MLA kernel dispatch profile for the GLM-5.2 geometry; NVFP4 MoE with host KV offload |
| **Gemma 4**<br><span class="family-desc">dense 31B and MoE 26B-A4B, multimodal, hybrid sliding-window attention</span> | <ul><li>✅ `RedHatAI/gemma-4-31B-it-FP8-Dynamic`</li><li>✅ `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`</li><li>🧪 `google/gemma-4-31B-it`</li></ul> | Sliding-window / global hybrid attention page sizing; multimodal path with MMMU-Pro accuracy baselines |
| **Llama 3.x**<br><span class="family-desc">dense</span> | <ul><li>✅ `meta-llama/Llama-3.1-8B-Instruct` (EAGLE-3 tests)</li><li>📝 `meta-llama/Llama-3.2-1B-Instruct`</li></ul> | Reference dense path; EAGLE-3 drafting with `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |
| **Qwen2.5-VL / Qwen3-VL**<br><span class="family-desc">vision-language</span> | <ul><li>✅ `Qwen/Qwen2.5-VL-3B-Instruct`</li><li>📝 `Qwen/Qwen2.5-VL-7B-Instruct`</li><li>🧪 `Qwen/Qwen3-VL-4B`</li></ul> | 3D M-RoPE, vision-encoder attention patches, multimodal encoder manager with video frame chunking |
| **Embedding & safety classifiers** | <ul><li>📝 `Qwen/Qwen3-VL-Embedding-8B`</li><li>📝 `meta-llama/Llama-Guard-4-12B`</li></ul> | TPU-native on-device pooling with request bucketing and static compilation; safety classification on the AILuminate prompt set |

</div>

See the [TPU v7x](../deployment/v7x_recipes.md) and [TPU v6e](../deployment/v6e_recipes.md) serving recipes for the exact slice topologies, parallelism layouts, and `vllm serve` flags used for the CI checkpoints.

---

## Feature Support

All features below are merged in `main`. "Validated with" lists the models and validation level behind each feature; a model missing from a row means the combination has not been run in CI, not that it is unsupported. Click a feature to see how to enable it.

<div class="expandable-table" data-detail-column="How to enable" markdown>

| Feature | How to enable | Validated with | Guide |
| :--- | :--- | :--- | :--- |
| **Ragged Paged Attention (RPA v3)** | Default attention backend for MHA / GQA (`--attention-backend CUSTOM` in the recipes); `VLLM_TPU_BLOCK_MAJOR_KV=1` for the block-major KV layout, `USE_RPA_PIPELINED_DMA_STAGING=1` | <ul><li>Every text model in CI, with `bf16` and `fp8` KV cache</li></ul> | [Env vars](../reference/env_vars.md) |
| **Multi-head Latent Attention (MLA)** | Automatic for MLA architectures; `TPU_SPARSE_MLA_NOPE_LAYOUT` / `TPU_SPARSE_MLA_ROPE_LAYOUT`, `TPU_LATENT_PROJ_INTRA_CHIP_TP`, `TPU_MLA_MASKED_DENSE_ENABLED` | <ul><li>✅ DeepSeek-V2</li><li>🌙 DeepSeek-V4</li><li>✅ Kimi-K3</li><li>🌙 GLM-5.2</li></ul> | — |
| **Hybrid linear attention (GDN / KDA)** | Automatic for hybrid architectures; `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1` shares one block pool between linear-attention state and KV cache | <ul><li>✅ Qwen3.5-35B</li><li>🌙 Qwen3.5-397B</li><li>✅ Kimi-K3</li><li>🧪 Kimi-Linear</li></ul> | — |
| **Tensor / Data / Expert parallelism** | `--tensor-parallel-size`, `--data-parallel-size`, `--enable-expert-parallel`; `TPU_MOE_HIERARCHICAL_EP=1` on dual-core TPU v7x | <ul><li>Every CI recipe, from `TP=1, DP=8 + EP` to `TP=32 + EP`</li></ul> | [Parallelism](parallelism.md) |
| **Pipeline & context parallelism (PP, PCP, DCP)** | `--pipeline-parallel-size`, `--prefill-context-parallel-size`, `--decode-context-parallel-size`; `TPU_PP_DYNAMIC_CHUNKS`, `TPU_PCP_TOPOLOGY_AWARE_MESH`, `TPU_DCP_LSE_MERGE_A2A` | <ul><li>PCP + DCP:<ul><li>✅ Qwen3.5 (PCP4 / DP4 and PCP4-TP1 to TP4-DCP2 disaggregated correctness tests)</li></ul></li><li>🧪 PP</li></ul> | [Parallelism](parallelism.md) |
| **Multi-host serving** | `mp` backend (`--nnodes`, or multi-host `--data-parallel-size`) or `TPU_MULTIHOST_BACKEND=ray` | <ul><li>✅ Qwen3-0.6B (v7x-16 presubmit)</li><li>🌙 GLM-5.2 (v7x-16)</li><li>🌙 DeepSeek-V4-Pro (v7x-16)</li><li>Kimi-K3 (on-demand v7x-32)</li></ul> | [Multi-Host Serving](multihost.md) |
| **Quantization** | Auto-detected from the checkpoint, or `--quantization fp8`, `compressed-tensors`, `mxfp4`, `nvfp4`, `awq`, `gptq`, `deepseek_v4_fp8`, `online_fp8`; `--kv-cache-dtype fp8`; load-time requantization via `REQUANTIZE_*` / `MOE_REQUANTIZE_*` | <ul><li>FP8 weights + KV:<ul><li>✅ Qwen3-Coder-30B</li><li>🌙 Qwen3-Coder-480B</li><li>✅ Qwen3.5-35B</li><li>🌙 Qwen3.5-397B</li><li>🌙 GLM-5.2</li><li>✅ Kimi-K3</li><li>✅ Gemma 4</li></ul></li><li>NVFP4:<ul><li>✅ Qwen3-30B</li><li>🌙 GLM-5.2</li></ul></li><li>W4A16:<ul><li>🌙 Kimi-K2.6</li></ul></li><li>`deepseek_v4_fp8` (W4A8 MoE):<ul><li>🌙 DeepSeek-V4</li></ul></li></ul> | [Quantization](quantization.md) |
| **Automatic prefix caching** | `--enable-prefix-caching` (with prefix-aware request scheduling, `PrefixAwareScheduler`) | <ul><li>✅ Qwen3.5</li><li>✅ Kimi-K3 (including hybrid KDA state)</li></ul> | — |
| **Chunked prefill & async scheduling** | `--max-num-batched-tokens`, `--async-scheduling`; `TPU_TOKEN_BUCKET_EXTRA` / `TPU_TOKEN_BUCKET_LINEAR`, `DP_SCHED_ENABLED` | <ul><li>Every CI recipe</li></ul> | [v7x recipes](../deployment/v7x_recipes.md) |
| **KV cache offloading** | `--kv-transfer-config '{"kv_connector": "TPURaidenOffloadingConnector", "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector", "kv_role": "kv_both", ...}'` | <ul><li>✅ Qwen3.5 (host DRAM pool + store node)</li><li>✅ Kimi-K3</li><li>🌙 DeepSeek-V4-Flash</li><li>🌙 GLM-5.2-NVFP4</li></ul> | — |
| **Disaggregated serving (P/D)** | `--kv-transfer-config` with `TpuRaidenConnector` (or `TPUMultiConnector` to combine with offloading); `TPU_RAIDEN_*` | <ul><li>✅ Qwen3.5 (P4D2, PCP / TP / DCP resharding)</li><li>🌙 DeepSeek-V4-Flash</li><li>📝 Kimi-K3 1P1D</li></ul> | [Disaggregated Serving](disaggregated_serving.md) |
| **Speculative decoding** | `--speculative-config '{"method": "eagle3" / "mtp" / "dflash" / "dspark" / "ngram" / "suffix", ...}'` | <ul><li>EAGLE-3:<ul><li>✅ Llama-3.1-8B</li></ul></li><li>MTP:<ul><li>✅ Qwen3.5-4B / 35B</li></ul></li><li>DFlash:<ul><li>✅ Qwen3-4B</li><li>✅ Qwen3.5-35B</li></ul></li><li>DSpark:<ul><li>🌙 DeepSeek-V4-Flash</li><li>✅ Qwen3-4B</li></ul></li><li>✅ N-gram</li></ul> | [Speculative Decoding](speculative_decoding.md) |
| **Structured outputs** | `--structured-outputs-config '{"backend": "xgrammar"}'` plus `structured_outputs` in the request | <ul><li>🧪 Qwen3-0.6B (`tests/entrypoints/llm/test_structured_output.py`)</li><li>📝 Qwen3-0.6B (`qwen3-0.6b-structured-*` benchmark configs)</li></ul> | — |
| **Multi-LoRA** | `--enable-lora`, `--max-lora-rank` | <ul><li>📝 Qwen2.5-3B (`examples/offline_lora_inference.py`)</li></ul> | — |
| **Multimodal inputs** | `--limit-mm-per-prompt`; `--language-model-only` to serve VLM checkpoints text-only; `MM_ENCODER_FRAME_CHUNK_PATCH_SIZE` for video frame chunking | <ul><li>✅ Gemma 4 (MMMU-Pro)</li><li>✅ Qwen2.5-VL-3B (image and video chunking tests)</li></ul> | [Quickstart](../getting_started/quickstart.md) |
| **Embedding & pooling** | `--runner pooling` (or `PoolerConfig(task="embed")` in Python) | <ul><li>📝 Qwen3-VL-Embedding-8B</li></ul> | — |
| **Compilation & warmup cache** | `VLLM_XLA_CACHE_PATH`, `--compilation-config '{"compile_sizes": [...]}'`, `TPU_PARALLEL_PRECOMPILE=1`, `VLLM_XLA_CHECK_RECOMPILATION=1` | <ul><li>✅ Qwen3-0.6B (compilation-cache end-to-end test)</li><li>Every recipe</li></ul> | [Compilation Cache](compilation_cache.md) |
| **Weight streaming & loading** | `--load-format runai_streamer` (from `gs://`), `--enable-ep-weight-filter`, `--safetensors-load-strategy prefetch`, `--model-loader-extra-config '{"memory_limit": ...}'` | <ul><li>🌙 GLM-5.2</li><li>🌙 DeepSeek-V4-Pro</li><li>Kimi-K3 (multi-host recipes)</li></ul> | [Multi-Host Serving](multihost.md) |
| **SparseCore offload** | `USE_MOE_SPARSE_CORE=1` (default), `ONEHOT_MOE_PERMUTE_THRESHOLD` | <ul><li>Every MoE recipe</li><li>🌙 Sparse MLA top-k on DeepSeek-V4</li></ul> | [Parallelism](parallelism.md) |
| **Profiling & observability** | `--profiler-config.profiler torch` with `--profile-result-dir`; `USE_PHASED_PROFILER=1`; Python / host / device tracer levels; vLLM Prometheus metrics (local and external prefix-cache hit counters) | <ul><li>📝 `examples/tpu_profiling.py`</li><li>📝 `examples/run_profiling_suite.sh`</li></ul> | [Profiling](../developers_guide/profiling/index.md) |

</div>

See the [Environment Variables Reference](../reference/env_vars.md) for the full list of tuning knobs behind these features.
