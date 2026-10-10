# Quantization

`vllm-torchtpu` implements TPU-native weight, activation, and KV cache quantization under `src/vllm_torchtpu/layers/adapter/quantization/`. Quantized weights are reshaped and packed once at model load time (`process_weights_after_loading`) so Pallas matmul, attention, and fused MoE kernels execute directly on TPU MXU native low-precision paths (`fp8` and `int4`/`fp4` -> `fp8`) without runtime layout overhead.

---

## 1. Supported Quantization Formats

| Method (`--quantization`) | Weight Dtype | Activation Dtype | Supported Layers | Notes |
| :--- | :--- | :--- | :--- | :--- |
| *(auto from checkpoint)* / `fp8` | `float8_e4m3fn` | `float8_e4m3fn` or `bf16` | Linear, Fused MoE, Attention KV | Supports static per-tensor, per-channel, and 2D block-wise FP8 checkpoints (e.g., `Qwen3.5-*-FP8`, `Qwen3-Coder-*-FP8`). |
| `compressed-tensors` | `fp8` (`W8A8`), `int4` (`W4A8`, `W4A16`), `mxfp4` | `fp8` or `bf16` | Linear, Fused MoE | Automatically selected for Hugging Face checkpoints with `quantization_config.quant_method == "compressed-tensors"`. |
| `mxfp4` | `fp4` (`e2m1`) + `e8m0` block scales (block size 32) | `bf16` / `fp8` | Linear, Fused MoE | Used by `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and MXFP4 MoE checkpoints. |
| `nvfp4` | `fp4` (`e2m1`) + `fp8` block scales | `bf16` / `fp8` | Linear, Fused MoE | Supported for NVFP4 checkpoints (e.g., `Qwen3-30B-A3B-NVFP4`, `Qwen3-Coder-480B-A35B-NVFP4`). |
| `awq` | `int4` (packed) | `bf16` | Linear | Weight-only 4-bit AWQ checkpoints. |
| `gptq` | `int4` / `int8` | `bf16` | Linear | Weight-only GPTQ checkpoints. |
| `deepseek_v4_fp8` | `float8_e4m3fn` (with optional `fp4` MoE requantization) | `float8_e4m3fn` | Linear, MLA, Fused MoE | Specialized FP8 + W4A8 MoE path for `DeepSeek-V4`. |
| `online_fp8` | `bf16` checkpoint -> quantized to `float8_e4m3fn` at load time | `float8_e4m3fn` | Linear, Fused MoE | Quantizes unquantized `bf16` checkpoints to FP8 on the fly during weight loading. |

> [!TIP]
> **Self-Describing Checkpoints**: When a Hugging Face checkpoint's `config.json` already contains a `quantization_config` section (such as `compressed-tensors`, `fp8`, or `mxfp4`), vLLM detects the format automatically—you do not need to pass `--quantization` unless overriding or enabling online load-time quantization.

---

## 2. FP8 Weight & KV Cache Quantization

Combining FP8 model weights with an FP8 KV cache (`--kv-cache-dtype fp8`) halves both weight and KV cache HBM footprints while doubling MXU matrix throughput on TPU v6e and TPU v7x:

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

### Example: Requantizing MoE Experts to W4A8 at Load Time

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
