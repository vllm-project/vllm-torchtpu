# Qwen3 480B Benchmark Recipe

This matches the 480B nightly perf config used by `run_benchmarks.sh`.
Use one terminal for the server and another for the benchmark client.

## Server

```bash
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random

vllm serve --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --tensor-parallel-size=8 --data-parallel-size=1 \
  --max-model-len=16384 --max-num-batched-tokens=8192 --max-num-seqs=512 \
  --port 8000 --async-scheduling --no-enable-prefix-caching \
  --gpu-memory-utilization=0.95 --kv-cache-dtype=fp8 \
  --enable-expert-parallel --quantization fp8 \
  --attention-backend CUSTOM
```

## Client

Run `vllm bench serve` against the running server.

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-filename isl1024_osl1024_c64.json \
  --seed 42
```

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 8192 \
  --random-range-ratio 0.8 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-filename isl1024_osl8192_c64.json \
  --seed 42
```

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-filename isl8192_osl1024_c64.json \
  --seed 42
```
