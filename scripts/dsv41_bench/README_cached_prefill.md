# DP cached-prefill benchmark

Run on four physical v7x chips (eight devices) with the repository-pinned
runtime and a local DeepSeek-V4.1-Flash checkpoint:

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
TP=1 DP=8 CHUNK=2048 MAX_SEQS=1 \
  bash scripts/dsv41_bench/serve_cached_prefill.sh
python scripts/dsv41_bench/bench_cached_prefill.py \
  --tokenizer "$MODEL_PATH" --dataset /tmp/longbench-v2.json \
  --out /tmp/dp8-prefill --concurrency 8 --dp-size 8 --rounds 10
python scripts/dsv41_bench/validate_cached_prefill.py /tmp/dp8-prefill
```

The client requires `requests`, `prometheus-client` and `transformers`.
It verifies a pinned LongBench-v2 dataset, routes eight distinct prefixes to
ranks 0 through 7, and runs reset/prime/extend waves. Each request reuses
65536 tokens and computes 16384 new tokens, with one greedy output token.
One warmup precedes measured waves. Inputs, hashes, responses and metrics
are saved; the validator checks per-rank hits, completion and preemptions.
Prime, warmup and profile times are excluded from extension measurements.

The server enables EP, CED, Engram CPU offload, FP8 KV and DSpark with five
draft tokens. The 2048-token budget and one-sequence limit are per DP engine.
It uses ordinary scheduler admission; this PR does not add an admission
window. Keep development/cache-reset endpoints on an isolated network.
One-token timing is not accuracy validation or a multi-step decode benchmark.

Supported DP requires PP1/TP1, EP, CPU Engram offload and one node-local
Engram group covering all DP ranks, with `dp_shared_memory=false`.
