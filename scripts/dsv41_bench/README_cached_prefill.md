# DP cached-prefill reproduction

This recipe runs DeepSeek-V4.1 v2 with TP1/DP8/EP8 on four physical v7x
chips (eight devices). It requires the repository-pinned TPU runtime and a
local model checkpoint. It does not modify the attention top-k kernels.

## Server

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
TP=1 DP=8 CHUNK=2048 MAX_SEQS=1 PREFILL_ADMISSION_MS=100 \
  bash scripts/dsv41_bench/serve_cached_prefill.sh
```

The launcher enables CED, SP, local queries, Engram CPU offload, fused EP
MoE, FP8 attention/indexer KV and DSpark with five draft tokens. Each DP
engine uses a 2048-token budget, one sequence and 4 GiB of KV cache.
The maximum sequence length is 82944; compilation buckets are 128, 256
and 2048. The global token budget is eight times the per-engine budget.
The optional 100 ms admission window aligns fresh cache-affine bursts;
set `PREFILL_ADMISSION_MS=off` for a control or latency-sensitive C1.
The server enables development endpoints for cache reset and profiling:
keep it bound to localhost or an isolated benchmark network.

## Client

Install `requests`, `prometheus-client` and `transformers` in the client
Python environment, then run:

```bash
python scripts/dsv41_bench/bench_cached_prefill.py \
  --tokenizer "$MODEL_PATH" --dataset /tmp/longbench-v2.json \
  --out /tmp/dp8-cached-prefill --concurrency 8 --dp-size 8 --rounds 10
python scripts/dsv41_bench/validate_cached_prefill.py /tmp/dp8-cached-prefill
```

The client downloads and verifies a pinned LongBench-v2 dataset when the
file is absent. Eight distinct prompts are pinned to ranks 0 through 7
with `X-data-parallel-rank`. Each round resets the cache, primes a 65536-token
prefix, then extends it by 16384 tokens. Requests use greedy sampling,
streaming and one output token. One warmup round precedes measured rounds.
The output includes tokenized inputs, input hashes, responses and metrics.
The validator requires at least five measured rounds, checks each rank's
65536 cache hits, one completion and zero preemptions, and reports prefill,
TTFT and throughput of newly computed input tokens. Prime, compilation and
profile time must not be included in the reported extension timings.

For active DSpark and idle-rank checks after the prefill run:

```bash
python scripts/dsv41_bench/check_long_context_decode.py \
  --input /tmp/dp8-cached-prefill/input.json --out /tmp/dp8-decode-check
```

## Accuracy and limits

One-token completion and cache-counter validation are not accuracy tests.
Run GSM8K separately, for example with `lm-eval[api]`:

```bash
lm_eval --tasks gsm8k --seed '0,1234,None,1234' \
  --model local-chat-completions \
  --model_args 'model=deepseek-v41,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=8,max_retries=0,timeout=3600' \
  --apply_chat_template --fewshot_as_multiturn true \
  --gen_kwargs '{"chat_template_kwargs":{"thinking":false},"max_gen_toks":1024,"temperature":0}' \
  --log_samples --output_path /tmp/dp8-gsm8k
```

The supported DP envelope is PP1/TP1 with EP and CPU-offloaded Engram,
`dp_shared_memory=false`, and one node-local Engram group covering all DP
replicas. Multi-node DP and TP greater than one combined with DP are rejected.
Repeated greedy runs can still differ in generated answers; do not infer
bitwise model determinism from fixed inputs or a one-token benchmark.
