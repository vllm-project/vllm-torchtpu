# Offline profiling with `examples/tpu_profiling.py`

[`examples/tpu_profiling.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/examples/tpu_profiling.py) drives the offline
`LLM` API with a synthetic batch, warms up, and then captures a trace of
`--num-iters` generate calls. Use it when you want a clean trace of one shape
with no scheduler noise.

```bash
python3 examples/tpu_profiling.py --model <model> [OPTIONS]
```

Key arguments:

* `--model` — model name or path.
* `--input-len` — prompt tokens per request.
* `--output-len` — tokens generated per request.
* `--batch-size` — number of requests.
* `--num-iters-warmup` / `--num-iters` — warmup iterations (not traced) and
  profiled iterations. Warmup matters: the first iterations pay XLA compilation.
* `--profile-result-dir` — where the trace lands.
* All standard vLLM `EngineArgs` are accepted (`--tensor-parallel-size`,
  `--max-model-len`, `--enable-expert-parallel`, ...).

**Profile a prefill.** Long prompt, one output token, small batch:

```bash
python3 examples/tpu_profiling.py \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --input-len 1024 \
  --output-len 1 \
  --batch-size 1 \
  --profile-result-dir profiles/prefill_1k
```

**Profile a decode.** Single-token requests, large batch:

```bash
python3 examples/tpu_profiling.py \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --input-len 1 \
  --output-len 1 \
  --batch-size 256 \
  --profile-result-dir profiles/decode_bs256
```

[`examples/run_profiling_suite.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/main/examples/run_profiling_suite.sh) sweeps a
matrix of these; uncomment the experiments you want and run it.
