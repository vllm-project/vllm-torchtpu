# TPU v6e (Trillium) Serving Recipes

**TPU v6e (Trillium)** provides 1 TensorCore per physical chip with `32 GB` of high-bandwidth memory (HBM) per chip. Single-host TPU v6e VMs are available in `1-chip` (`v6e-1`, `32 GB` HBM), `4-chip` (`v6e-4`, `128 GB` HBM), and `8-chip` (`v6e-8`, `256 GB` HBM) topologies, with multi-host pod slices scaling to `v6e-16` (`512 GB` HBM) and beyond.

---

## Recommended Topologies by Model Size

| Model Checkpoint | Precision | Recommended v6e Slice | Parallelism Layout |
| :--- | :--- | :--- | :--- |
| `meta-llama/Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Qwen/Qwen3-8B` | `bf16` or `fp8` | `v6e-1` (1 chip, 32 GB) | `TP=1` |
| `Qwen/Qwen2.5-VL-7B-Instruct`, `google/gemma-3-12b-it` | `bf16` or `fp8` | `v6e-1` or `v6e-4` | `TP=1` or `TP=4` |
| `Qwen/Qwen3-32B`, `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`, `Qwen/Qwen3.5-35B-A3B-FP8` | `fp8` (`bf16` on `v6e-4`/`v6e-8`) | `v6e-4` (4 chips, 128 GB) | `TP=4` (+ `--enable-expert-parallel` for MoE) |
| `meta-llama/Llama-3.3-70B-Instruct`, `openai/gpt-oss-120b` | `fp8` / `mxfp4` (`bf16` on `v6e-8`) | `v6e-8` (8 chips, 256 GB) | `TP=8` (+ `--enable-expert-parallel` for MoE) |
| `Qwen/Qwen3.5-397B-A17B-FP8` | `fp8` | `v6e-16` (2 hosts, 512 GB) | `TP=16` or `DP=2, TP=8` + EP |

---

## 1. `v6e-1` (Single Chip, 32 GB HBM)

Ideal for 1B–8B dense models and development testing:

```bash
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --port 8000
```

---

## 2. `v6e-4` (4 Chips, 128 GB HBM)

Recommended for 30B–35B dense and MoE checkpoints:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --port 8000
```

---

## 3. `v6e-8` (8 Chips, 256 GB HBM)

Recommended for 70B–120B models on a single host without multi-host networking overhead:

### `meta-llama/Llama-3.3-70B-Instruct` (`TP=8`)

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --port 8000
```

### `openai/gpt-oss-120b` (`TP=8 + EP`, MXFP4)

```bash
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --port 8000
```

> [!TIP]
> **Maximizing KV Cache Capacity on `v6e` (`32 GB` HBM/chip)**: Always pass `--kv-cache-dtype fp8` on TPU v6e to double available KV cache blocks and prevent preemption under long-context or high-concurrency workloads.
