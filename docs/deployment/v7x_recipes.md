# TPU v7x (Ironwood) Serving Recipes

**TPU v7x (Ironwood)** is Google Cloud's 7th-generation TPU architecture. Each physical TPU v7x chip exposes **2 logical TensorCore devices** to PyTorch and JAX (`192 GB` HBM per chip / `96 GB` per logical device), so a single 4-chip TPU v7x host (`tpu7x-8`) provides **8 logical TPU devices** and `768 GB` of total HBM.

The recipes below are extracted directly from the repository's active PR and nightly CI performance benchmark configurations in [`scripts/vllm/benchmarking/configs/`](https://github.com/vllm-project/vllm-torchtpu/tree/main/scripts/vllm/benchmarking/configs) (scheduled via [`.buildkite/pipeline_perf.yml`](https://github.com/vllm-project/vllm-torchtpu/blob/main/.buildkite/pipeline_perf.yml)).

---

## Summary of Validated TPU v7x Topologies

| Model Checkpoint | Slice / Devices | Parallelism Layout | Quantization & KV Cache | CI Benchmark Config |
| :--- | :--- | :--- | :--- | :--- |
| `deepseek-ai/DeepSeek-V4.1-Flash` | `v7x-8` (1 host, 8 devices) | `TP=1, DP=8` + EP | `deepseek_v4_fp8` (W4A8 MoE), `fp8` KV | [`deepseek-v41-flash-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep.sh), [`deepseek-v41-flash-dp8-ep-128k-prefill.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep-128k-prefill.sh), [`deepseek-v41-flash-dp8-ep-dspark-decode.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep-dspark-decode.sh) |
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

### `deepseek-ai/DeepSeek-V4.1-Flash` (`TP=1, DP=8 + EP`)

Single-host recipes on `TPU v7x-8` across three configurations:

- [`deepseek-v41-flash-dp8-ep.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep.sh): Nightly CI smoke test (`ISL=128, OSL=16, Concurrency=8`, `BENCHMARK_WARMUP_RUNS=1` to amortize first-wave XLA/host dispatch warmup before timing: `542.64 tok/s` total, `60.29 tok/s` output, `33.30 ms` median TPOT), `lm_eval` GSM8K 5-shot `--limit 100` accuracy gate (`LM_EVAL_TASKS=gsm8k`, `EVAL_TOLERANCE=0.03`, checked in at `scripts/vllm/benchmarking/baselines/eval/deepseek-v41-flash-dp8-ep.gsm8k.baseline.json` with `limit=100`: `0.98 ± 0.0141` `flexible-extract`, `0.92 ± 0.0273` `strict-match`), manual mixed serving (`ISL=4096, OSL=1024, Concurrency=8`: `1038.64 tok/s` total, `207.73 tok/s` output, `33.10 ms` median TPOT), and steady-state high-concurrency non-speculative decode (`ISL=8000, OSL=6000, Concurrency=384` with `--prime-prefix-cache`, `MAX_MODEL_LEN=131072`, `MAX_NUM_SEQS=48`, `KV_CACHE_MEMORY_BYTES=21474836480`, `COMPILE_SIZES=[64,288,2048]`: `7065.60 tok/s` peak-window / `6693.79 tok/s` full-overlap output throughput, `53.39 ms` peak-active TPOT P50, `50.33 ms` peak-window TPOT P50). Script defaults use `MAX_MODEL_LEN=8192`, `KV_CACHE_MEMORY_BYTES=4294967296` (`4 GiB` per rank), and `COMPILE_SIZES=[8,64,2048]` to bound XLA compilation time and HBM allocation while providing headroom for `4096 + 1024 = 5120` tokens.
- [`deepseek-v41-flash-dp8-ep-128k-prefill.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep-128k-prefill.sh): Manual `128k` pure prefill benchmark (`ISL=129024, OSL=1, Concurrency=1`, `MAX_MODEL_LEN=131072`, `KV_CACHE_MEMORY_BYTES=21474836480`, `ENABLE_PREFIX_CACHING=false`; measured on `TPU v7x-8`: `3581.05 tok/s` total throughput, `36028.76 ms` mean and median TTFT; with `DEEPSEEK_V41_CONCURRENCY_OPTIONS=8`: `24830.05 tok/s` total throughput, `38729.00 ms` mean TTFT, `38911.85 ms` median TTFT; at `ISL=100000, OSL=1, Concurrency=8`: `24768.63 tok/s` total throughput, `30300.48 ms` median TTFT).
- [`deepseek-v41-flash-dp8-ep-dspark-decode.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep-dspark-decode.sh): Manual `128k` decode config (script defaults: `ISL=129024, OSL=2048, Concurrency=64`, `MAX_MODEL_LEN=131072`, `MAX_NUM_BATCHED_TOKENS=2048`, `MAX_NUM_SEQS=48`, `KV_CACHE_MEMORY_BYTES=21474836480`, `COMPILE_SIZES=[64,288,2048]`, `DSpark K=5` speculative decoding) with `--enable-ep-weight-filter`, `--safetensors-load-strategy=prefetch`, and `--safetensors-prefetch-num-threads=8` enabled across all 8 EP ranks (`48/384` target routed experts and `16/128` `DSpark` draft routed experts per rank). Measured on `TPU v7x-8` with overrides `MAX_NUM_SEQS=8`, `COMPILE_SIZES=[64,2048]`, and `temperature=0`: periodic decode (`ISL=8000, OSL=1024, Concurrency=8`) achieves `98.50%` acceptance (`6816/6920` draft tokens across `1384` draft steps, `5.925` mean acceptance length, `901.27 tok/s` full-overlap output throughput from `rounds.csv` as the `8.99 s` decode span finishes inside the `10 s` default sliding window, `578.41 tok/s` end-to-end throughput, `9.19 ms` mean / `8.88 ms` median request TPOT, `5.07 s` median TTFT); single-request greedy decode (`ISL=256, OSL=18, Concurrency=1`) achieves `100.0%` acceptance (`15/15` draft tokens, `6.000` mean acceptance length); post-fix `GSM8K` 5-shot sanity scores `9/9` (`100%` exact match, `61.32%` acceptance, `4.066` mean acceptance length), and the baseline `lm_eval` `GSM8K` 5-shot eval run (`limit=100`, `max_gen_toks=1024`, `--fewshot_as_multiturn true`) scores `0.99 ± 0.0100` `flexible-extract` (`0.95 ± 0.0219` `strict-match`; `check_regression.py` `PASS` against `deepseek-v41-flash-dp8-ep.gsm8k.baseline.json`); high-concurrency `Concurrency=384` speculative vs. non-speculative throughput comparison remains open pending post-fix re-measurement.

```bash
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export USE_MOE_FUSED_EP_KERNEL=1
export MOE_FUSED_EP_ENABLE_W4A8=1
export MOE_FUSED_EP_V2_SHARDED_PLAN=1
export MOE_REQUANTIZE_BLOCK_SIZE=256
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"

vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --safetensors-load-strategy prefetch \
  --safetensors-prefetch-num-threads 8 \
  --kv-cache-dtype fp8 \
  --attention-backend CUSTOM \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --block-size 16 \
  --kv-cache-memory-bytes 4294967296 \
  --compilation-config '{"compile_sizes":[8,64,2048]}' \
  --async-scheduling \
  --enable-prefix-caching \
  --language-model-only \
  --port 8000
```

> [!NOTE]
> **Verification & Manual Benchmark Commands**:
>
> - **Nightly Smoke + GSM8K Accuracy Gate**: `./scripts/vllm/benchmarking/run_eval_flow.sh --config deepseek-v41-flash-dp8-ep --run-lm-eval`
> - **128k Pure Prefill (Manual Only)**: `./scripts/vllm/benchmarking/run_benchmarks.sh --config deepseek-v41-flash-dp8-ep-128k-prefill`
> - **DSpark Decode Benchmark (Manual Only)**: `./scripts/vllm/benchmarking/run_benchmarks.sh --config deepseek-v41-flash-dp8-ep-dspark-decode`
> - **Mixed Serving (Manual Only)**: `DEEPSEEK_V41_ISL_OSL_CONFIGS=4096:1024 DEEPSEEK_V41_CONCURRENCY_OPTIONS=8 ./scripts/vllm/benchmarking/run_benchmarks.sh --config deepseek-v41-flash-dp8-ep` (the `DEEPSEEK_V41_*` prefix overrides `run_benchmarks.sh` pre-source harness defaults for `ISL_OSL_CONFIGS`, `CONCURRENCY_OPTIONS`, `NUM_PROMPTS`, `BENCHMARK_WARMUP_RUNS`, and `LM_EVAL_TASKS`).

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
