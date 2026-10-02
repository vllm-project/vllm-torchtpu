# GLM-5.3-Flash

| | |
|---|---|
| Hugging Face repo | `zai-org/GLM-5.3-Flash` |
| Architecture class | `Glm5NextForConditionalGeneration`, `Glm5NextForCausalLM` (`model_type: glm5_next`) |
| Code | `src/vllm_torchtpu/models/vllm/glm5_next/`, `src/vllm_torchtpu/layers/adapter/custom_ops/glm53_*` |
| Status | Experimental |
| Last validated | 2026-09-29, on `main` @ `250958522` |

## Model summary

- 45 decoder layers: 34 Kimi Delta Attention (KDA, linear attention with a
  fixed-size recurrent state and a short convolution) and 11 NoPE multi-head
  latent attention (MLA) layers that read the paged KV cache.
- Manifold-constrained hyper-connections (mHC) around every attention and MLP
  block.
- Mixture-of-experts MLP.
- Checkpoint `max_position_embeddings`: 1,048,576. Validated up to
  `--max-model-len 131072`.

Because only the 11 MLA layers keep per-token KV, decode cost is nearly
independent of context length (see [Long context](#long-context)).

## What was added

- **Model** (`models/vllm/glm5_next/model.py`): decoder with mHC, KDA / MLA
  layer selection, MoE through `FusedMoEFactory` with the SwiGLU clamp,
  checkpoint-name mapping, `packed_modules_mapping`, and config registration
  for `glm5_next` / `glm5_next_text`.
- **KDA backends** (`GLM53_KDA_BACKEND`):
  - `kimi` (default): the Kimi pooled / dispatched KDA ops. With the unified
    block pool (default) the KDA state lives in the same block pool as the
    MLA KV cache. The conv weight is folded after weight loading.
  - `legacy`: the earlier Pallas KDA prefill/decode kernels with per-layer
    state caches (set `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0`).
- **mHC** as a fused VPU fp32 op; RPA v3 default block sizes for this model;
  GDN pool layout and platform hooks for the hybrid KV / state pool.

## How to run

```bash
vllm serve zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 10240 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

- `GLM53_KDA_BACKEND=kimi` is the default and needs no setting.
- For long context, raise `--max-model-len` (validated at 131072).
- Prefix caching: add `--enable-prefix-caching --mamba-cache-mode align`.

## Test environment

| Item | Value |
|---|---|
| TPU | TPU v7x, one host, 2x2x1 (4 chips, 8 TensorCores) |
| Parallelism | TP=8 |
| vllm / vllm_torchtpu | 0.29.0 / this branch |
| torch / torch_tpu | 2.13.0 / 0.1.1.dev20260912075415 |
| jax / libtpu | 0.10.2 / 0.0.47 |
| lm_eval / transformers | 0.5.0.dev1 / 5.14.1 |
| KV cache capacity | 549,326 tokens at `--max-model-len 10240` (53.6 max-length requests); 564,111 tokens at 131072 (4.3 max-length requests) |

## Tests

| Suite | Result (TPU v7x) |
|---|---|
| `tests/kernels/test_glm53_kimi_kda.py` | pass; the 2 `tp1` cases are strict xfails (TP=1 is unsupported, see Known limitations) |
| `tests/models/vllm/test_glm53_kimi_kda_layer.py` | pass |
| `tests/models/vllm/test_glm53_kda_batching.py` | pass |
| `tests/models/vllm/test_glm53_flash.py` | pass |
| `tests/kernels/test_rpa_v3_default_block_sizes.py` | pass |

CI: all files except `test_glm53_kimi_kda.py` are marked `cpu_test` and run on
the CPU agents. In `test_glm53_kimi_kda.py` only the layout and argument checks
are `cpu_test`; its Pallas kernel tests run on the TPU agents and skip on
other backends. The `tp1` xfails are `strict`, so they fail CI if the kernel
starts fitting TP=1.

## Accuracy

| Task | Setting | Score |
|---|---|---|
| MMLU (`mmlu_llama`) | 5-shot, 100 questions per subject, prefix caching off | **0.8818 ± 0.0041** |
| MMLU (`mmlu_llama`) | same, prefix caching on (align mode) | **0.8818 ± 0.0041** |
| MMLU (`mmlu_llama`) | same, earlier revision with the Pallas KDA kernels (before the Kimi ops) | 0.881 |

Groups (prefix caching off): humanities 0.8823, other 0.8723, social sciences
0.9150, STEM 0.8674.

A vendor-reported reference score on the same harness is not recorded yet.

```bash
lm_eval --tasks mmlu_llama --limit 100 --seed 0,1234,None,1234 --log_samples \
  --model local-chat-completions \
  --model_args "model=zai-org/GLM-5.3-Flash,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,timeout=3600" \
  --apply_chat_template --fewshot_as_multiturn true \
  --gen_kwargs '{"continue_final_message": true, "add_generation_prompt": false, "chat_template_kwargs": {"enable_thinking": false}}'
```

## Performance

All numbers are measured, one run per cell, with:

```bash
vllm bench serve --backend vllm --endpoint /v1/completions \
  --model zai-org/GLM-5.3-Flash --dataset-name random \
  --random-input-len <isl> --random-output-len <osl> --random-range-ratio 0 \
  --num-prompts <2 x concurrency> --max-concurrency <concurrency> \
  --ignore-eos --seed 0
```

Each cell is preceded by a warm-up at the same concurrency (osl 64 or 16).
Requests are sent all at once, so the first wave's prefills queue behind each
other: P99 TTFT is roughly concurrency x single-request prefill time, and mean
TTFT is the better summary.

### 8k in / 1k out

Server: `--max-model-len 10240`, prefix caching off.

| Concurrency | ok / n | Mean TTFT (s) | P99 TTFT (s) | Median TPOT (ms) | Median ITL (ms) | Output tok/s | Total tok/s |
|---|---|---|---|---|---|---|---|
| 2 | 4/4 | 1.04 | 1.46 | 105.2 | 103.5 | 18.8 | 169 |
| 4 | 8/8 | 4.16 | 21.68 | 104.6 | 102.2 | 34.5 | 310 |
| 8 | 16/16 | 3.66 | 22.92 | 108.0 | 105.3 | 65.0 | 585 |
| 16 | 32/32 | 4.30 | 21.10 | 107.5 | 101.7 | 131.3 | 1182 |
| 32 | 64/64 | 6.45 | 22.55 | 112.0 | 102.0 | 267.1 | 2404 |
| 34 | 68/68 | 6.81 | 29.31 | 115.1 | 105.9 | 269.4 | 2425 |

Against the earlier revision with the Pallas KDA kernels (before the Kimi ops;
same hardware shape and load): median TPOT 150.7-170.1 ms and output
13.1-181.2 tok/s, so the Kimi-ops backend decodes 29-37% faster with 33-54%
more output throughput.

### High concurrency

Server: `--max-model-len 10240 --max-num-seqs 128`, prefix caching off.

| ISL / OSL | Concurrency | ok / n | Mean TTFT (s) | Median TPOT (ms) | Output tok/s |
|---|---|---|---|---|---|
| 1k / 1k | 64 | 128/128 | 1.89 | 117.8 | 533 |
| 1k / 1k | 128 | 256/256 | 4.98 | 119.8 | 1011 |
| 8k / 1k | 64 | 128/128 | 14.71 | 131.5 | 333 |

- Median TPOT stays within 2% from 64 to 128 concurrent 1k/1k requests, so
  output throughput almost doubles.
- 8k / 1k at 64 needs about 590k KV tokens, more than the 549k in the pool.
  The scheduler preempted 8 requests and recomputed them; all requests
  finished, and the server log has no errors.

### Long context

Server: `--max-model-len 131072`, prefix caching off, output 1024 tokens.
"126k" is 129,024 input tokens.

| ISL | Concurrency | ok / n | Mean TTFT (s) | P99 TTFT (s) | Median TPOT (ms) | Output tok/s | Total tok/s |
|---|---|---|---|---|---|---|---|
| 16k | 1 | 2/2 | 1.36 | 1.39 | 101.9 | 9.7 | 165 |
| 16k | 2 | 4/4 | 1.82 | 2.66 | 101.3 | 19.4 | 329 |
| 16k | 4 | 8/8 | 10.17 | 24.89 | 118.9 | 31.7 | 538 |
| 16k | 8 | 16/16 | 7.05 | 34.44 | 137.2 | 56.1 | 954 |
| 16k | 16 | 32/32 | 9.04 | 43.64 | 113.3 | 118.2 | 2009 |
| 32k | 1 | 2/2 | 2.75 | 2.75 | 101.4 | 9.6 | 317 |
| 32k | 2 | 4/4 | 3.54 | 5.37 | 103.6 | 18.7 | 616 |
| 32k | 4 | 8/8 | 4.95 | 10.58 | 105.7 | 36.0 | 1187 |
| 32k | 8 | 16/16 | 7.80 | 21.20 | 110.7 | 66.8 | 2204 |
| 32k | 16 | 32/32 | 13.35 | 43.22 | 119.1 | 119.8 | 3954 |
| 64k | 1 | 2/2 | 5.83 | 5.84 | 102.6 | 9.2 | 601 |
| 64k | 2 | 4/4 | 7.42 | 11.39 | 101.8 | 18.2 | 1182 |
| 64k | 4 | 8/8 | 10.40 | 22.64 | 107.0 | 33.7 | 2194 |
| 64k | 8 | 16/16 | 16.43 | 45.95 | 118.1 | 59.1 | 3839 |
| 126k | 1 | 2/2 | 12.90 | 12.90 | 104.0 | 8.6 | 1090 |
| 126k | 2 | 4/4 | 16.36 | 25.18 | 108.3 | 16.1 | 2043 |
| 126k | 4 | 8/8 | 23.07 | 50.61 | 119.4 | 28.0 | 3561 |

- Single-request TPOT grows only 2% from 16k to 126k.
- Single-request prefill runs at 10-12k tokens/s at every length.
- At 126k the KV pool fits about 4 concurrent requests.
- Re-checked on the final tree (`main` @ `250958522`): 64k at concurrency 8
  gives median TPOT 122.8 ms and 52.8 output tok/s, and 126k at concurrency 4
  gives 122.1 ms and 25.6 tok/s. All requests succeeded, with no preemptions
  or server errors.

### Prefix caching

Server: `--enable-prefix-caching --mamba-cache-mode align`, `--max-model-len 10240`.
Load: `--dataset-name prefix_repetition`, 6144-token shared prefix,
1024-token unique suffix, 512 output tokens, concurrency 8, 32 requests.

| Setting | Mean TTFT (s) | Median TTFT (s) | Median TPOT (ms) | Output tok/s |
|---|---|---|---|---|
| Caching off | 1.45 | 0.98 | 104.2 | 74.4 |
| Caching on | 2.70 | 0.60 | 106.3 | 68.2 |

- Prefix-cache hit rate over the cached run (`vllm:prefix_cache_hits_total /
  queries_total`): 79%, with 0 preemptions.
- MMLU with caching on matches caching off (0.8818 both).
- End-to-end exact-output check: the same ~4k-token prompt sent four times at
  temperature 0 (the first request misses, the other three hit the cached
  prefix) gives identical text, with a max logprob difference of 0.0.
- The higher mean TTFT with caching on comes from a few slow requests in a
  32-request run and was not investigated further.

## Known limitations

- **Speculative decoding** is not supported with pooled KDA (the default). The
  pooled path keeps one KDA state per request (the state after the last
  scheduled token). A verify step needs a state checkpoint for each draft
  position so it can roll back rejected tokens, and there is nowhere to keep
  them. So the layer fails at the first speculative step with
  `NotImplementedError`, instead of producing wrong tokens.
  With `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0` (dedicated conv / recurrent state
  caches), ngram speculative decoding works. The server starts, a greedy
  prompt gives the same text as the pooled server without speculation, and
  1k/256 at concurrency 8 finished 16/16 with 60% draft acceptance. It was
  slower than the pooled path without speculation (median TPOT 186 ms), so
  it is a working fallback, not a tuned configuration. Pooled-KDA
  speculative decoding needs per-position state checkpoints in the pool;
  that is future work.
- **TP=1** does not fit: 64 KDA heads per shard exceed v7x VMEM in the fused
  kernel. Use TP=8.
- Validated with `--enforce-eager` only.

## Reproduction checklist

- [x] Server command and environment variables
- [x] Test commands
- [x] `lm_eval` command
- [x] `vllm bench serve` commands
- [ ] Raw result JSON location (not shared)
