# Qwen3 480B Benchmark Recipe

This matches the 480B nightly perf config used by `run_benchmarks.sh`.
Use one terminal for the server and another for the benchmark client.

## Server

```bash
vllm serve --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --tensor-parallel-size=8 --data-parallel-size=1 \
  --max-model-len=10240 --max-num-batched-tokens=8192 --max-num-seqs=512 \
  --port 8000 --async-scheduling --no-enable-prefix-caching \
  --gpu-memory-utilization=0.95 --kv-cache-dtype=fp8 \
  --enable-expert-parallel --quantization fp8 \
  --attention-backend CUSTOM
```

## Client

Run `vllm bench serve` against the running server. The config sets `RANDOM_RANGE_RATIO=0.8` with `RANGE_RATIO_STYLE=min` (lengths in `[0.8*len, len]`); `run_benchmarks.sh` translates that to the symmetric lengths and ratio below.

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --trust-remote-code \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input-len 922 \
  --random-output-len 922 \
  --random-range-ratio 0.111111 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  --temperature 0 \
  --save-result \
  --result-filename isl1024_osl1024_c64.json \
  --seed 42
```

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --trust-remote-code \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input-len 922 \
  --random-output-len 7373 \
  --random-range-ratio 0.111111 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  --temperature 0 \
  --save-result \
  --result-filename isl1024_osl8192_c64.json \
  --seed 42
```

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --trust-remote-code \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input-len 7373 \
  --random-output-len 922 \
  --random-range-ratio 0.111111 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  --temperature 0 \
  --save-result \
  --result-filename isl8192_osl1024_c64.json \
  --seed 42
```
