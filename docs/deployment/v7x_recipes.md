# TPU v7x (Ironwood) Serving Recipes

**TPU v7x (Ironwood)** is Google Cloud's 7th-generation TPU architecture. Each physical TPU v7x chip exposes **2 logical TensorCore devices** to PyTorch and JAX (`192 GB` HBM per chip / `96 GB` per logical device), so a single 4-chip TPU v7x host (`tpu7x-8`) provides **8 logical TPU devices** and `768 GB` of total HBM.

The recipes below are extracted directly from the repository's active PR and nightly CI performance benchmark configurations in [`scripts/vllm/benchmarking/configs/`](https://github.com/vllm-project/vllm-torchtpu/tree/main/scripts/vllm/benchmarking/configs) (scheduled via [`.buildkite/pipeline_perf.yml`](https://github.com/vllm-project/vllm-torchtpu/blob/main/.buildkite/pipeline_perf.yml)).

---

## Summary of Validated TPU v7x Topologies

| Model Checkpoint | Slice / Devices | Parallelism Layout | Quantization & KV Cache | CI Benchmark Config |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-397B-A17B-FP8` | `v7x-8` (1 host, 8 devices) | `TP=1, DP=8` + EP | `fp8` weights, `fp8` KV | [`qwen3.5-397b-fp8-tp1-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh) |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | `v7x-8` (1 host, 8 devices) | `TP=8, DP=1` + EP | `fp8` weights, `fp8` KV | [`qwen3-coder-480b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-480b-fp8-tp8-ep.sh) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `v7x-8` (1 host, 8 devices) | `TP=8, DP=1` + EP | `fp8` weights, `fp8` KV | [`qwen3-coder-30b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-tp8-ep.sh) |
| `Qwen/Qwen3.5-35B-A3B-FP8` | `v7x-8` (1 host, 8 devices) | `TP=2, DP=4` + EP | `fp8` weights, `fp8` KV | [`qwen3.5-35b-fp8-dp4-tp2-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-35b-fp8-dp4-tp2-ep.sh) |
| `RedHatAI/gemma-4-31B-it-FP8-Dynamic` | `v7x-8` (1 host, 8 devices) | `TP=2, DP=4` | `compressed-tensors` FP8 | [`gemma-4-31b-fp8-dp4-tp2-multimodal.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/gemma-4-31b-fp8-dp4-tp2-multimodal.sh) |
| `zai-org/GLM-5.2-FP8` | `v7x-16` (2 hosts, 16 devices) | `TP=1, DP=16` + EP | `fp8` weights, `fp8` KV | [`glm-5.2-fp8-dp16-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh) |
| `deepseek-ai/DeepSeek-V4-Pro-0813` | `v7x-16` (2 hosts, 16 devices) | `TP=1, DP=16` + EP | `deepseek_v4_fp8` (W4A8 MoE), `fp8_e4m3` KV | [`deepseek-v4-pro-dp16-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v4-pro-dp16-ep.sh) |
| `moonshotai/Kimi-K3` | `v7x-32` (4 hosts, 32 devices) | `TP=32, DP=1` + EP | `compressed-tensors` FP8, `fp8` KV | [`kimi-k3-tp32-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/kimi-k3-tp32-ep.sh) |

---

## 1. Single-Host Recipes (`v7x-8` — 8 Logical Devices)

### `Qwen/Qwen3.5-397B-A17B-FP8` (`TP=1, DP=8 + EP`)

Golden throughput recipe on a single TPU v7x host ([`scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh)):

```bash
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 10240 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --block-size 256 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --port 8000
```

> [!TIP]
> **Alternative Parallelism Layouts for `Qwen3.5-397B-A17B-FP8`**: For lower per-request latency at smaller concurrencies, use `--tensor-parallel-size 2 --data-parallel-size 4` ([`qwen3.5-397b-fp8-tp2-dp4-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp2-dp4-ep.sh)) or `--tensor-parallel-size 8 --data-parallel-size 1` ([`qwen3.5-397b-fp8-tp8-dp1-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp8-dp1-ep.sh)).

### `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` (`TP=8 + EP`)

Validated nightly in [`scripts/vllm/benchmarking/configs/qwen3-coder-480b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-480b-fp8-tp8-ep.sh):

```bash
vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --port 8000
```

### `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (`TP=8 + EP`)

Used as the PR-time performance guard gate ([`scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-tp8-ep.sh)):

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --port 8000
```

---

## 2. Multi-Host Recipes (`v7x-16` & `v7x-32`)

On multi-host TPU v7x slices without large local disks, stream checkpoints directly from Google Cloud Storage (`gs://`) using `--load-format runai_streamer` and `--enable-ep-weight-filter` so each rank downloads only its own MoE expert shard.

### `zai-org/GLM-5.2-FP8` on `v7x-16` (`2 Hosts, TP=1, DP=16 + EP`)

Validated nightly in [`scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh):

```bash
vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --safetensors-load-strategy prefetch \
  --kv-cache-dtype fp8 \
  --max-model-len 9216 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --port 8000
```

### `moonshotai/Kimi-K3` on `v7x-32` (`4 Hosts, TP=32 + EP`)

Validated nightly in [`scripts/vllm/benchmarking/configs/kimi-k3-tp32-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/kimi-k3-tp32-ep.sh):

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 32 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --trust-remote-code \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --model-loader-extra-config '{"memory_limit":17179869184}' \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 10240 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.82 \
  --compilation-config '{"compile_sizes":[1,8,8192]}' \
  --async-scheduling \
  --port 8000
```
