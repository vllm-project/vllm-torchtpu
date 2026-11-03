# TPU v7x (Ironwood) Serving Recipes

**TPU v7x (Ironwood)** is Google Cloud's 7th-generation TPU architecture. Each physical TPU v7x chip exposes **2 logical TensorCore devices** to PyTorch and JAX (`192 GB` HBM per chip / `96 GB` per logical device), so a single 4-chip TPU v7x host (`tpu7x-8`) provides **8 logical TPU devices** and `768 GB` of total HBM.

Most recipes below are extracted directly from the repository's active PR and nightly CI performance benchmark configurations in [`scripts/vllm/benchmarking/configs/`](https://github.com/vllm-project/vllm-torchtpu/tree/main/scripts/vllm/benchmarking/configs) (scheduled via [`.buildkite/pipeline_perf.yml`](https://github.com/vllm-project/vllm-torchtpu/blob/main/.buildkite/pipeline_perf.yml)). A recipe validated outside CI says so in its section.

---

## Summary of Validated TPU v7x Topologies

| Model Checkpoint | Slice / Devices | Parallelism Layout | Quantization & KV Cache | CI Benchmark Config |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-397B-A17B-FP8` | `v7x-8` (1 host, 8 devices) | `TP=1, DP=8` + EP | `fp8` weights, `fp8` KV | [`qwen3.5-397b-fp8-tp1-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp1-dp8-ep.sh) |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | `v7x-8` (1 host, 8 devices) | `TP=8, DP=1` + EP | `fp8` weights, `fp8` KV | [`qwen3-coder-480b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-480b-fp8-tp8-ep.sh) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `v7x-8` (1 host, 8 devices) | `TP=8, DP=1` + EP | `fp8` weights, `fp8` KV | [`qwen3-coder-30b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3-coder-30b-fp8-tp8-ep.sh) |
| `Qwen/Qwen3.5-35B-A3B-FP8` | `v7x-8` (1 host, 8 devices) | `TP=8, DP=1` + EP | `fp8` weights, `fp8` KV | [`qwen3.5-35b-fp8-tp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/qwen3.5-35b-fp8-tp8-ep.sh) |
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

### `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` (`TP=8 + EP`, long prompts, context-parallel prefill)

Validated on a long-prompt trace (prompts of 10K-160K tokens, mean 87K, short answers, up to 64 concurrent requests); not a CI benchmark config:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600   # 19 context-parallel programs compile at startup (~30 min cold)
export TPU_FP8_ATTENTION=1 TPU_RPA_SKIP_EMPTY_LAUNCH=1 TPU_RPA_FENCE_OUTPUT=1 \
  TPU_RPA_CAP_SLIDING_DECODE_BLOCK=1 TPU_NORM_WEIGHTS_FP32=1 TPU_REUSE_STAGED_BLOCK_TABLES=1 \
  TPU_GEMMA4_TOPK_SOFTMAX_ROUTING=1 TPU_MOE_GATHER_GMM=1
export VLLM_TPU_SOLO_PREFILL_OVER_TOKENS=8192 TPU_GEMMA4_CP_MIN_TOKENS=16384
export VLLM_KV_CACHE_LAYOUT=HND

vllm serve RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' \
  --language-model-only \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' \
  --attention-backend CUSTOM \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 163840 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 163840 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"compile_sizes": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072, 139264, 147456, 155648, 163840]}' \
  --api-server-count 4 \
  --port 8000
```

- Every prompt above 8192 tokens is prefilled whole in a step of its own (`VLLM_TPU_SOLO_PREFILL_OVER_TOKENS`), as a context-parallel program over the 8 ranks for each compile size from 16384 up (`TPU_GEMMA4_CP_MIN_TOKENS`). A step pads its prompt to the next compile size, so the sizes step by 8192 above 16384.
- At TP 8 every layer keeps one KV head per rank; `VLLM_KV_CACHE_LAYOUT=HND` stores it without padding, twice the KV cache of the default layout, which is what lets all 64 long requests run at once. `TPU_GEMMA4_FUSED_ATTN_PROLOGUE` does not support HND; leave it unset.
- One API-server process falls behind on 64 streams of long prompts; `--api-server-count 4` spreads them.
- `LIBTPU_INIT_ARGS=--xla_tpu_dvfs_p_state=7` (the highest DVFS performance state) adds about 5% at 4 or more concurrent requests.

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
