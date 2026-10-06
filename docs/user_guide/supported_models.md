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

In addition to the nightly matrix above, the following model families have first-class kernel, quantization, and parallelism support tested in CI and benchmarking suites:

| Category | Model Family / Checkpoints | Key TPU Optimizations |
| :--- | :--- | :--- |
| **Frontier MoE & Hybrid Attention** | **Qwen3.5** (`Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-35B-A3B-FP8`) | Gated DeltaNet (GDN) linear attention + Ragged Paged Attention, SparseCore/GMM Fused MoE, FP8 / W4A8 / MXFP4, PCP, Raiden P/D disaggregation |
| **Coding & Dense/MoE LLMs** | **Qwen3 / Qwen3-Coder** (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`, `Qwen/Qwen3-32B`, `Qwen/Qwen3-8B`) | Fused EP MoE kernels, FP8 static/block quantization, EAGLE-3 & DFlash speculative decoding |
| **Multi-head Latent Attention (MLA)** | **DeepSeek-V3 / R1 / V4** (`deepseek-ai/DeepSeek-R1`), **Kimi-K2.5 / K3**, **GLM-5 / GLM-5.1**, **Mistral-Large-3** | Pallas MLA v1/v2 & Sparse MLA kernels, intra-chip latent projection TP, DCP + EP + TP sharding, MXFP4 / W4A8 MoE |
| **Open-Weight Dense & MoE** | **Llama 3 / 3.1 / 3.2 / 3.3 / 4**, **GPT-OSS** (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`), **Gemma 2 / 3 / 4**, **Mixtral** | Ragged Paged Attention v3, MXFP4 / FP8 quantized matmul, Tensor & Pipeline Parallelism |
| **Vision-Language & Multimodal** | **Qwen2.5-VL**, **Qwen3-VL** (including embedding), **Kimi-K2.5 VL**, **Gemma-3 / Gemma-4 VL** | 3D M-RoPE, vision encoder FlashAttention, multimodal token chunking |
| **Diffusion & Omni (`vllm-omni`)** | **FLUX.1** (`FLUX.1-dev`, `FLUX.1-schnell`, `FLUX.1-Kontext-dev`), **Wan2.2** | `vllm_omni` TPU platform plugin (`tests_omni/`) |

---

## Core Feature Support Overview

| Feature | Status | Details |
| :--- | :--- | :--- |
| **Ragged Paged Attention (RPA v3)** | Supported | Custom Pallas paged attention for MHA/GQA with block-major and HND KV cache layouts (`bf16` and `fp8`). |
| **Multi-head Latent Attention (MLA)** | Supported | Dense MLA v1/v2 and Sparse MLA kernels with SparseCore gather and KV compression. |
| **Gated DeltaNet (GDN) Hybrid Attention** | Supported | Pallas chunked prefill and recurrent decode kernels for hybrid linear + full attention models (e.g., Qwen3.5). |
| **Parallelism (TP, DP, PP, EP, PCP, DCP)** | Supported | Up to 6D composable parallelism across single-host and multi-host TPU slices. |
| **Quantization** | Supported | Unquantized `bf16`, `fp8` (`e4m3fn`), `compressed-tensors` (`W8A8`, `W4A8`, `W4A16`), `awq`, `gptq`, `mxfp4`, and `fp8` KV cache. |
| **Disaggregated Serving (P/D)** | Supported | High-bandwidth KV cache transfer via `TpuRaidenConnector` (`tpu_sync`) with dynamic P/D resharding. |
| **Speculative Decoding** | Supported | `ngram`, `suffix`, `mtp` (Multi-Token Prediction), `eagle3`, and `dflash`. |
| **Multi-LoRA** | Supported | Dynamic LoRA adapter routing (`--enable-lora`). |
| **Compilation & Warmup Cache** | Supported | Persistent on-disk XLA (`VLLM_XLA_CACHE_PATH`) and `torch_tpu` warmup caches. |
