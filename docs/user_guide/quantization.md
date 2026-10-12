# Quantization

`vllm-torchtpu` implements TPU-native weight, activation, and KV cache quantization under `src/vllm_torchtpu/layers/adapter/quantization/`. Quantized weights are reshaped and packed once at model load time (`process_weights_after_loading`) so Pallas matmul, attention, and fused MoE kernels execute directly on TPU MXU native low-precision paths (`fp8` and `int4`/`fp4` -> `fp8`) without runtime layout overhead.

---

## 1. Supported Quantization Formats

| Method (`--quantization`) | Weight Dtype | Activation Dtype | Supported Layers | Notes |
| :--- | :--- | :--- | :--- | :--- |
| *(auto from checkpoint)* / `fp8` | `float8_e4m3fn` | `float8_e4m3fn` or `bf16` | Linear, Fused MoE, Attention KV | Supports static per-tensor, per-channel, and 2D block-wise FP8 checkpoints (e.g., `Qwen3.5-*-FP8`, `Qwen3-Coder-*-FP8`). |
| `compressed-tensors` | `fp8` (Linear and MoE); `int4` (`W4A16` / `W4A8`) and `mxfp4` (MoE only) | `fp8` or `bf16` | Linear (FP8 only), Fused MoE | Automatically selected for Hugging Face checkpoints with `quantization_config.quant_method == "compressed-tensors"`. KV-cache quantization from these checkpoints is not supported. |
| `mxfp4` | `fp4` (`e2m1`) + `e8m0` block scales (block size 32) | `bf16` / `fp8` | Linear, Fused MoE | Used by `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and MXFP4 MoE checkpoints. |
| `modelopt_fp4` | `fp4` (`e2m1`) + `fp8` block scales | `bf16` / `fp8` | Linear, Fused MoE | ModelOpt NVFP4 checkpoints (e.g., `nvidia/Qwen3-30B-A3B-NVFP4`, `nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4`). |
| `deepseek_v4_fp8` | `float8_e4m3fn` (with optional `fp4` MoE requantization) | `float8_e4m3fn` | Linear, MLA, Fused MoE | Specialized FP8 + W4A8 MoE path for `DeepSeek-V4`. |
| `additional_config.tpu_online_fp8` (not a `--quantization` value) | selected `bf16` Linear weights -> `float8_e4m3fn` at load time | N/A | Selected unquantized Linear layers | Applies on top of an `fp8` or `modelopt_fp4` checkpoint config. |

> [!TIP]
> **Self-Describing Checkpoints**: When a Hugging Face checkpoint's `config.json` already contains a `quantization_config` section (such as `compressed-tensors`, `fp8`, or `mxfp4`), vLLM detects the format automatically. You do not need to pass `--quantization` unless overriding the detected method.

---

## 2. FP8 Weight & KV Cache Quantization

Combining FP8 model weights with an FP8 KV cache (`--kv-cache-dtype fp8`) halves both weight and KV cache HBM footprints (on TPU v6e, `--kv-cache-dtype fp8` maps to `fp8_e5m2`):

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95
```

---

## 3. Load-Time Weight Requantization

For checkpoints whose native block granularity or weight dtype is not optimal for TPU MXU tiling, `vllm-torchtpu` supports load-time requantization controlled via environment variables:

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `ENABLE_QUANTIZED_MATMUL_KERNEL` | `0` | Enables the dedicated Pallas quantized matmul kernel for linear layers. |
| `REQUANTIZE_WEIGHT_DTYPE` | `"float8_e4m3fn"` | Target weight dtype when requantizing linear weights at load time. |
| `REQUANTIZE_BLOCK_SIZE` | `None` | Target subchannel block size when requantizing linear weights at load time. |
| `MOE_REQUANTIZE_WEIGHT_DTYPE` | `"float8_e4m3fn"` | Target weight dtype for MoE expert weights (set to `fp4` with `MOE_FUSED_EP_ENABLE_W4A8=1` for W4A8 MoE execution). |
| `MOE_REQUANTIZE_BLOCK_SIZE` | `None` | Target block size (e.g., `512`) for load-time MoE weight requantization. |
| `QUANTIZE_ON_LOAD_PREFIXES` | `""` | Comma-separated, dot-bounded module-name prefixes (e.g., `self_attn,shared_experts`). Matching unquantized Linear and embedding weights, including `embed_tokens` and `lm_head`, are quantized to FP8 at load time. `kv_b_proj` is excluded. |

### Example: Requantizing MoE Experts to W4A8 at Load Time

This `TP=1 × DP=16` layout targets a two-host TPU v7x-16 slice: run the command on each host with `--data-parallel-size-local 8` and the multi-host DP flags from [Multi-Host Serving](multihost.md#mode-b-multi-host-data-parallelism-data-parallel-size).

```bash
export USE_MOE_FUSED_EP_KERNEL=1
export MOE_FUSED_EP_ENABLE_W4A8=1
export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4
export MOE_REQUANTIZE_BLOCK_SIZE=512

vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --enable-expert-parallel \
  --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8_e4m3
```
