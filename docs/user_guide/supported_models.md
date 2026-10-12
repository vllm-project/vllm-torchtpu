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

In addition to the nightly matrix above, the following checkpoints are exercised by the repository's tests, CI pipelines, or benchmark configs (`scripts/vllm/benchmarking/configs/`):

| Category | Model Family / Checkpoints | Key TPU Optimizations |
| :--- | :--- | :--- |
| **Frontier MoE & Hybrid Attention** | **Qwen3.5** (`Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-35B-A3B-FP8`) | Gated DeltaNet (GDN) linear attention + Ragged Paged Attention, SparseCore/GMM Fused MoE, FP8 / W4A8 / MXFP4, PCP, Raiden P/D disaggregation |
| **Coding & Dense/MoE LLMs** | **Qwen3 / Qwen3-Coder** (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`, `nvidia/Qwen3-30B-A3B-NVFP4`, `Qwen/Qwen3-0.6B`) | Fused EP MoE kernels, FP8 static/block quantization, EAGLE-3 & DFlash speculative decoding |
| **Multi-head Latent Attention (MLA)** | **DeepSeek-V2 / V4** (`deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Pro`), **Kimi-K2.6 / K3** (`moonshotai/Kimi-K3`), **GLM-5.2** (`zai-org/GLM-5.2-FP8`) | Pallas MLA v1/v2 & Sparse MLA kernels, Kimi-K3 intra-chip latent-MoE projection TP, DCP + EP + TP sharding, MXFP4 / W4A8 MoE |
| **Open-Weight Dense & MoE** | **Llama 3.1** (`NousResearch/Meta-Llama-3.1-8B-Instruct`, EAGLE-3 tests), **Gemma 4** (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`, `RedHatAI/gemma-4-31B-it-FP8-Dynamic`) | Ragged Paged Attention v3, FP8, Tensor & Data Parallelism |
| **Vision-Language & Multimodal** | **Qwen2.5-VL** (`Qwen/Qwen2.5-VL-3B-Instruct`), **Qwen3-VL** (including `Qwen/Qwen3-VL-Embedding-8B`), **Kimi-K3** (vision), **Gemma 4** (multimodal) | 3D M-RoPE, vision encoder FlashAttention, multimodal token chunking |
| **Diffusion & Omni (`vllm-omni`)** | **FLUX.1** (`FLUX.1-dev`, `FLUX.1-schnell`), **FLUX.2** (`FLUX.2-klein-4B`) | `vllm_omni` TPU platform plugin (`tests_omni/`) |

---

## Core Feature Support Overview

| Feature | Status | Details |
| :--- | :--- | :--- |
| **Ragged Paged Attention (RPA v3)** | Supported | Custom Pallas paged attention for MHA/GQA with block-major and HND KV cache layouts (`bf16` and `fp8`). |
| **Multi-head Latent Attention (MLA)** | Supported | Dense MLA v1/v2 and Sparse MLA kernels with SparseCore gather and KV compression. |
| **Gated DeltaNet (GDN) Hybrid Attention** | Supported | Pallas chunked prefill and recurrent decode kernels for hybrid linear + full attention models (e.g., Qwen3.5). |
| **Parallelism (TP, DP, PP, EP, PCP, DCP)** | Supported | Across single-host and multi-host TPU slices. Not every combination is allowed (for example, PP requires TP=1); see [Parallelism & Scaling](parallelism.md). |
| **Quantization** | Supported | Unquantized `bf16`, `fp8`, `mxfp4`, `modelopt_fp4` (NVFP4), `compressed-tensors`, DeepSeek-V4 FP8, and `fp8` KV cache. See [Quantization](quantization.md). |
| **Disaggregated Serving (P/D)** | Supported | High-bandwidth KV cache transfer via `TPURaidenConnector` (`tpu_sync`) with dynamic P/D resharding. |
| **Speculative Decoding** | Supported | `ngram`, `mtp` (Multi-Token Prediction), `eagle3`, `dflash`, and `dspark`. |
| **Multi-LoRA** | Supported | Dynamic LoRA adapter routing (`--enable-lora`). |
| **Compilation Cache** | Supported | Persistent on-disk TorchTPU compilation cache (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`). |
