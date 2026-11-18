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
| `Qwen/Qwen3.5-35B-A3B-FP8` | `v7x-8` (1 host, 8 devices) | `TP=8, DP=1` + EP | `fp8` weights, `fp8` KV | [`qwen3.5-35b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-35b-fp8-tp8-ep.sh) |
| `RedHatAI/gemma-4-31B-it-FP8-Dynamic` | `v7x-8` (1 host, 8 devices) | `TP=2, DP=4` | `compressed-tensors` FP8 | [`gemma-4-31b-fp8-dp4-tp2-multimodal.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/gemma-4-31b-fp8-dp4-tp2-multimodal.sh) |
| `deepseek-ai/DeepSeek-V4-Flash` | `v7x-8` (1 host, 8 devices) | `TP=1, DP=8` + EP | `deepseek_v4_fp8` (W4A8 MoE), `fp8_e4m3` KV | [`deepseek-v4-flash-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v4-flash-dp8-ep.sh) |
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

### `deepseek-ai/DeepSeek-V4-Flash` (`TP=1, DP=8 + EP`)

Validated nightly on a single TPU v7x host (`v7x-8`) with W4A8 load-time MoE requantization ([`scripts/vllm/benchmarking/configs/deepseek-v4-flash-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v4-flash-dp8-ep.sh)):

**Server:**

```bash
export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4
export MOE_REQUANTIZE_BLOCK_SIZE=512
export TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_MOE_HASH_TABLE_ROW_MAJOR=1

vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --max-model-len 9216 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 256 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --async-scheduling \
  --generation-config="vllm" \
  --override-generation-config='{"do_sample": false, "temperature": 0.0, "model": "deepseek-ai/DeepSeek-V4-Flash"}' \
  --port 8000
```

**Benchmark Client (`8k` input / `1k` output):**

```bash
vllm bench serve \
  --backend vllm \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --num-prompts 2048 \
  --ignore-eos \
  --skip-chat-template
```

---

## 2. Multi-Host Recipes (`v7x-16` & `v7x-32`)

On multi-host TPU v7x slices without large local disks, stream checkpoints directly from Google Cloud Storage (`gs://`) using `--load-format runai_streamer` and `--enable-ep-weight-filter` so each rank downloads only its own MoE expert shard.

### `deepseek-ai/DeepSeek-V4-Pro-0813` on `v7x-16` (`2 Hosts, TP=1, DP=16 + EP`)

Validated nightly across 2 TPU v7x hosts (`2x2x2` topology, 16 logical devices) with fused EP MoE v2 and W4A8 MoE weight requantization ([`scripts/vllm/benchmarking/configs/deepseek-v4-pro-dp16-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v4-pro-dp16-ep.sh)):

**Server:**

```bash
export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4
export MOE_REQUANTIZE_BLOCK_SIZE=512
export TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_MOE_HASH_TABLE_ROW_MAJOR=1
export USE_MOE_FUSED_EP_KERNEL=1
export MOE_FUSED_EP_ENABLE_W4A8=1
export MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE=2
export TPU_MOE_SKIP_PADDED_TOKENS=1
export RAGGED_GATHER_REDUCE_VERSION=v3
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"

vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --safetensors-load-strategy prefetch \
  --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --max-model-len 9216 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 256 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --generation-config="vllm" \
  --override-generation-config='{"do_sample": false, "temperature": 0.0, "model": "deepseek-ai/DeepSeek-V4-Pro-0813"}' \
  --port 8000
```

> [!IMPORTANT]
> **Ray Multi-Host Environment Propagation**: When launching across a 2-host Ray cluster (`TPU_MULTIHOST_BACKEND=ray`), set `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` so non-`VLLM_`-prefixed model and kernel variables are forwarded to every Ray DP engine actor:
>
> ```bash
> export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="MOE_REQUANTIZE_WEIGHT_DTYPE,MOE_REQUANTIZE_BLOCK_SIZE,TPU_ROPE_CACHE_ROW_MAJOR,TPU_MOE_HASH_TABLE_ROW_MAJOR,USE_MOE_FUSED_EP_KERNEL,MOE_FUSED_EP_ENABLE_W4A8,MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE,TPU_MOE_SKIP_PADDED_TOKENS,RAGGED_GATHER_REDUCE_VERSION,LIBTPU_INIT_ARGS"
> ```
>
> When streaming the 805 GiB checkpoint from GCS via `--load-format runai_streamer`, also pass `--model-loader-extra-config '{"memory_limit":4294967296,"concurrency":8}'` to cap per-rank host streamer buffers. See [Multi-Host Serving](../user_guide/multihost.md) for full `mp` and `ray` cluster setup details.

**Benchmark Client (`8k` input / `1k` output):**

```bash
vllm bench serve \
  --backend vllm \
  --model deepseek-ai/DeepSeek-V4-Pro-0813 \
  --tokenizer deepseek-ai/DeepSeek-V4-Pro-0813 \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --num-prompts 2048 \
  --random-range-ratio 0 \
  --ignore-eos \
  --max-concurrency 1024
```

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
