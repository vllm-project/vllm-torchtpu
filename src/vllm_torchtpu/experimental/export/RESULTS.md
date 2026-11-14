# Results: exported models served by rvllm vs `vllm serve`

These numbers compare two ways of serving the same checkpoint on TPU v7x:

- an artifact exported with `vllm_torchtpu.experimental.export` and served by
  [rvllm](../rust/README.md), the Rust runtime for these artifacts;
- the same model served by `vllm serve` with vllm-torchtpu (labelled "vLLM" below).

Both sides use the same benchmark configuration from
`scripts/vllm/benchmarking/configs/<config>.sh`, with the same parallelism, quantization,
`max_num_seqs` and serve arguments. The rvllm artifact is exported with those same engine
arguments.

## Nightly benchmark (2026-11-06)

Workload settings:

- `vllm bench serve` on the random dataset;
- about 1024 input and 128 output tokens per request, with range ratio 0.8;
- temperature 0;
- 4 × concurrency prompts at each level (8 × at concurrency 8).

The configs' own input and output lengths are overridden to 1024/128 so that every model runs
the same workload.

Throughput is output tokens per second. TTFT and TPOT are means in milliseconds.

### Summary at concurrency 64

| config | rvllm tok/s | vLLM tok/s | rvllm / vLLM |
|---|---:|---:|---:|
| `qwen3-0.6b-tp1` | 10631 | 2358 | 4.51x |
| `qwen3.5-35b-fp8-tp4-ep` | 4033 | 2806 | 1.44x |
| `qwen3.5-397b-fp8-tp8-dp1-ep` | 1817 | 1610 | 1.13x |
| `qwen3-coder-480b-fp8-tp8-ep` | 1261 | 1280 | 0.99x |

The gap is widest on small models, where host-side scheduling dominates. On the 397B and 480B
MoE models, each step is dominated by device time, so the runtimes are close.

### Qwen3-0.6B, TP 1 (`qwen3-0.6b-tp1`, `MAX_NUM_SEQS=256`)

| concurrency | rvllm tok/s | vLLM tok/s | rvllm / vLLM | rvllm TTFT | vLLM TTFT | rvllm TPOT | vLLM TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4569.5 | 1185.4 | 3.85x | 32.6 | 418.9 | 1.50 | 3.46 |
| 64 | 10631.2 | 2357.8 | 4.51x | 114.5 | 1794.9 | 5.04 | 12.22 |
| 256 | 12555.2 | 5672.8 | 2.21x | 475.9 | 1789.4 | 15.99 | 28.62 |

### Qwen3.5-35B-A3B-FP8, TP 4 + EP (`qwen3.5-35b-fp8-tp4-ep`)

| concurrency | rvllm tok/s | vLLM tok/s | rvllm / vLLM | rvllm TTFT | vLLM TTFT | rvllm TPOT | vLLM TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4032.7 | 2806.2 | 1.44x | 291.0 | 680.2 | 13.47 | 16.73 |

### Qwen3.5-397B-A17B-FP8, TP 8 + EP (`qwen3.5-397b-fp8-tp8-dp1-ep`)

| concurrency | rvllm tok/s | vLLM tok/s | rvllm / vLLM | rvllm TTFT | vLLM TTFT | rvllm TPOT | vLLM TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1817.2 | 1609.5 | 1.13x | 889.8 | 1172.0 | 28.33 | 29.82 |
| 256 | 2654.8 | 2612.4 | 1.02x | 1808.3 | 1951.6 | 82.12 | 82.16 |
| 512 | 2923.3 | 2874.9 | 1.02x | 3082.7 | 3305.5 | 150.01 | 150.95 |

The golden configuration for this model is TP 1 × DP 8. It runs here at TP 8 because the exporter
does not support DP > 1 yet.

### Qwen3-Coder-480B-A35B-Instruct-FP8, TP 8 + EP (`qwen3-coder-480b-fp8-tp8-ep`)

| concurrency | rvllm tok/s | vLLM tok/s | rvllm / vLLM | rvllm TTFT | vLLM TTFT | rvllm TPOT | vLLM TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1261.4 | 1280.4 | 0.99x | 1326.3 | 1413.9 | 40.50 | 38.31 |

## Nightly accuracy (2026-11-06)

lm-eval runs against each server's OpenAI API on subsets: 10 questions per MMLU subject and per
MMLU-Pro category, and 10 problems each from HumanEval+ and MBPP+. The subsets are small, so a
difference of a few points is within noise. Each value is checked against the config's baseline
and tolerance (`EVAL_TOLERANCE`), and every value below passes.

| config | engine | MMLU (570) | MMLU-Pro (140) | HumanEval+ (10) | MBPP+ (10) |
|---|---|---:|---:|---:|---:|
| `qwen3.5-397b-fp8-tp8-dp1-ep` | rvllm | 0.907 | 0.871 | 1.00 | 0.90 |
| | vLLM | 0.911 | 0.850 | 1.00 | 1.00 |
| `qwen3-coder-480b-fp8-tp8-ep` | rvllm | 0.889 | 0.793 | 1.00 | 0.90 |
| | vLLM | 0.882 | 0.814 | 1.00 | 0.90 |

MMLU is `mmlu_llama` strict match, MMLU-Pro is custom-extract exact match, and the code tasks are
pass@1.

## Muse Spark Open 816B-A42B NVFP4, TP 8 (2026-11-04)

This is a separate run, not part of the nightly benchmark:

- 8192 input and 1024 output tokens per request, with range ratio 0.8;
- `max_num_seqs` 32, `max_num_batched_tokens` 8192, `max_model_len` 10240;
- each concurrency level uses its own seed, so prompts never repeat and the prefix cache never
  hits.

46 of the model's 62 layers use sliding-window attention (window 2048). The artifact declares KV
cache groups, so rvllm frees a sliding layer's blocks once they leave the window, as vLLM's hybrid
KV cache manager does. Without groups rvllm runs out of KV memory at concurrency 32: it preempts,
recomputes about 15% of the prompt tokens, and reaches 323.5 tok/s.

| concurrency | rvllm tok/s | vLLM tok/s | rvllm / vLLM | rvllm TPOT (median) | vLLM TPOT (median) | rvllm TTFT (median) | vLLM TTFT (median) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71.75 | 72.87 | 0.98x | 12.66 | 12.44 | 1171 | 1167 |
| 2 | 119.38 | 121.24 | 0.98x | 15.40 | 15.15 | 1186 | 1187 |
| 4 | 167.46 | 165.33 | 1.01x | 22.55 | 22.70 | 1195 | 1196 |
| 8 | 219.66 | 217.65 | 1.01x | 34.68 | 34.87 | 1218 | 1193 |
| 16 | 284.68 | 286.75 | 0.99x | 52.82 | 52.06 | 1218 | 1215 |
| 32 | 364.98 | 359.69 | 1.01x | 82.00 | 83.02 | 1242 | 1244 |

GSM8K uses the full set: 1319 questions, 5-shot, 256 tokens, temperature 0.

| engine | strict-match | flexible-extract |
|---|---:|---:|
| rvllm | 0.9553 ± 0.0057 | 0.9568 ± 0.0056 |
| vLLM | 0.9530 ± 0.0058 | 0.9538 ± 0.0058 |

Open items for this model:

- Concurrency 1 is about 1% slower: a decode step takes 12.70 ms against 12.58 ms, probably
  because there are more page tables per step.
- Concurrency above 32 needs a re-export with a larger `max_num_seqs`.
- Prefix caching, MTP and prefill/decode disaggregation are not yet supported for artifacts with
  sliding-window KV groups.
