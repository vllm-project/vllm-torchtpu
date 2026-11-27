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
| `TPU_FP8_LINEAR_W8A16_MAX_ROWS` | `0` | Opt-in. FP8 linear steps whose padded token bucket (from `compile_sizes`) has at most this many rows skip activation quantization and run bf16 activations against the FP8 weights (W8A16). Speeds up small decode steps on v7x; applies to the per-channel path with FP8 weights only, so the blockwise kernel path and `REQUANTIZE_WEIGHT_DTYPE=int8` are unaffected. |
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

---

## 4. Selective Load-Time FP8 Quantization

To quantize layers that a checkpoint stores in `bfloat16` (an unquantized checkpoint or layers left unquantized via `ignored_layers` / `exclude_modules`) to FP8 at load time, `vllm-torchtpu` provides two mechanisms:

| Aspect | `--additional-config '{"tpu_online_fp8": ...}'` | `QUANTIZE_ON_LOAD_PREFIXES` |
| :--- | :--- | :--- |
| **Checkpoint formats** | `fp8`, `nvfp4` (`modelopt_fp4`), `modelopt_mixed` | Unquantized BF16, `compressed-tensors`, `modelopt_mixed`, `nvfp4`, `fp8`, `mxfp4`, `deepseek_v4_fp8` |
| **Supported layers** | Unquantized `LinearBase` only | Unquantized `LinearBase` (always skips MLA `kv_b_proj`), unquantized `RoutedExperts` (without expert bias), and `VocabParallelEmbedding` / `ParallelLMHead` (`unquantized`, `compressed-tensors`, `modelopt_mixed` only) |
| **Selection syntax** | `targets` / `exclude` (`fnmatch` globs or exact names; `hf_to_vllm_mapper` applied; rejects partial fused-layer matches) | Comma-separated dot-bounded prefix tokens (e.g., `"self_attn,shared_experts,layers.0.mlp"`) |
| **Format knobs** | Always `float8_e4m3fn`; `weight_scheme`: `per_channel` (default), `per_tensor`, or `per_block` (`weight_block_size: [N_out, K_in]`, `K_in % 128 == 0`, dividing each TP-sharded `[N, K]` weight) | Fixed per-channel `float8_e4m3fn` for `LinearBase` and embeddings (ignores `REQUANTIZE_*`); `MOE_REQUANTIZE_WEIGHT_DTYPE` / `MOE_REQUANTIZE_BLOCK_SIZE` (default per-channel `float8_e4m3fn`) for `RoutedExperts` |
| **Load & validation** | Layerwise conversion during sharded weight load (lower peak load HBM) + post-load audit (`validate_online_fp8`) | Converts in `process_weights_after_loading` without `validate_online_fp8` or layerwise sharded conversion |

> [!WARNING]
> Do not target the same `LinearBase` layer with both `tpu_online_fp8` and `QUANTIZE_ON_LOAD_PREFIXES`: doing so raises a `ValueError` during quantization dispatch. Use only one mechanism per prefix.

```bash
# Layerwise online FP8 for unquantized Linear layers on NVFP4 / FP8 checkpoints
vllm serve nvidia/Qwen3-30B-A3B-NVFP4 \
  --additional-config '{"tpu_online_fp8": {"enabled": true, "targets": ["*self_attn.*_proj"], "weight_scheme": "per_channel"}}'

# Env-var prefix selection (covers excluded LinearBase and RoutedExperts)
QUANTIZE_ON_LOAD_PREFIXES="self_attn,experts" vllm serve nvidia/Qwen3-30B-A3B-NVFP4
```
