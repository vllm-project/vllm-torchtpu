# Model Onboarding Report Template

Copy this file to `docs/models/<model_name>.md` when a new model is onboarded,
fill in every section, and add the page to the `Supported Models` section of
`mkdocs.yml`. Keep the report next to the code change that onboards the model
and update it whenever a change moves accuracy or performance.

Guidelines:

- Report **measured** numbers only. If a number is an estimate, say so.
- State the exact software versions, hardware, and serving flags, so anyone can
  reproduce the result.
- Use the same load generator settings across models (`vllm bench serve`,
  random dataset, `--ignore-eos`, `num_prompts = 2 x concurrency`) unless the
  model needs something different; if so, explain why.
- Do not include internal cluster, project, or node names, internal links, or
  unreleased XLA/libtpu flags.

---

## <Model name>

| | |
|---|---|
| Hugging Face repo | `<org/model>` |
| Architecture class | `<ArchitectureForCausalLM>` |
| Code | `src/vllm_torchtpu/models/vllm/<dir>/` |
| Status | Experimental / Supported |
| Last validated | `<YYYY-MM-DD>`, branch/commit `<ref>` |

### Model summary

Parameter count (total / active), layer types and counts, attention variant,
MoE layout, context length, and anything that needed TPU-specific work.

### What was added or changed

Bullet list of the model code, kernels, and platform hooks added for this
model, plus environment variables that select code paths.

### How to run

```bash
vllm serve <org/model> \
  --tensor-parallel-size <tp> \
  --max-model-len <len> \
  ...
```

Required environment variables and their defaults.

### Test environment

| Item | Value |
|---|---|
| TPU | `<generation, topology, chips, TensorCores>` |
| TP / DP / EP | |
| vllm / vllm_torchtpu | |
| torch / torch_tpu | |
| jax / libtpu | |
| lm_eval / transformers | |
| Serving flags | |
| KV cache capacity (from server log) | |

### Tests

| Suite | Result |
|---|---|
| `tests/...` | passed / xfail (reason) |

### Accuracy

| Task | Setting | Score | Reference | Notes |
|---|---|---|---|---|
| MMLU (`mmlu_llama`) | 5-shot, `--limit 100`, chat template | | vendor / GPU reference | |

Exact `lm_eval` command:

```bash
lm_eval --tasks mmlu_llama ...
```

### Performance

`vllm bench serve` command:

```bash
vllm bench serve --backend vllm --dataset-name random \
  --random-input-len <isl> --random-output-len <osl> --random-range-ratio 0 \
  --num-prompts <2 x conc> --max-concurrency <conc> --ignore-eos --seed 0
```

#### Short context (for example 8k in / 1k out)

| Concurrency | Mean TTFT (s) | Median TPOT (ms) | Median ITL (ms) | Output tok/s | Total tok/s |
|---|---|---|---|---|---|

#### Long context

| ISL | Concurrency | Mean TTFT (s) | Median TPOT (ms) | Output tok/s |
|---|---|---|---|---|

#### Prefix caching (if supported)

Hit rate, TTFT with and without caching, and accuracy with caching on.

### Known limitations

Unsupported features (speculative decoding, quantization modes, TP sizes),
known performance gaps, and open issues.

### Reproduction checklist

- [ ] Server command and environment variables
- [ ] Test commands
- [ ] `lm_eval` command
- [ ] `vllm bench serve` commands
- [ ] Raw result JSON location (if shared)
