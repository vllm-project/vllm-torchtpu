# Static expert placement tables

Example tables for `TPU_EXPERT_PLACEMENT` (see `src/vllm_torchtpu/expert_placement.py`
and the "Static expert placement" section of `docs/user_guide/parallelism.md`).

| File | Model | Deployment | Prefill chunk (tokens) | sha256 |
| :--- | :--- | :--- | :--- | :--- |
| `glm52_ep16_c32_4k.json` | `zai-org/GLM-5.2-FP8` | flat EP16 (16 ranks x 16 experts), layers 3-77 | 4096 | `524dbf2c8d2bc6df86fc3cf411a13980adf1f0048a9363c6c3b3ff44059fc93b` |
| `glm52_ep16_c32_8k.json` | `zai-org/GLM-5.2-FP8` | flat EP16 (16 ranks x 16 experts), layers 3-77 | 8192 | `718b592b75dbdf98b86c9910f0c55990794218e8b2c7fadda78c7e6d85b2451a` |

Both tables were solved from one routing profile: per-layer expert hit counts of an
AgentX trace-replay serving run at concurrency 32 (MTP 3, 4096-token prefill chunks,
2,508 s window, 1,836 requests, decode and prefill steps). The objective is the
time-weighted busiest-rank MoE cost of decode steps and prefill chunks; each file's
`meta` records the weights and cost model.

## Format

```json
{
  "experts": 256, "shards": 16, "slots_per_core": 16,
  "layers": {"3": {"cores": [[27, 225, "...16 ids"], "...16 ranks"]}, "...": {}}
}
```

`layers["<n>"].cores[r]` lists the global expert ids rank `r` holds, in slot order;
every layer is a permutation of `0..experts-1`. Only routed decoder layers may be
listed (GLM-5.2: 3-77). The loader logs the table's sha256 on every worker:
`[eplb] placement installed ep_rank=<r> layers=<n> table_sha256=<first 16 hex>`.

## Use

```bash
TPU_EXPERT_PLACEMENT=/workspace/vllm-torchtpu/examples/expert_placement/glm52_ep16_c32_4k.json \
  vllm serve zai-org/GLM-5.2-FP8 --tensor-parallel-size 16 --enable-expert-parallel ...
```

The table needs flat expert parallelism (`TPU_MOE_EP_TP_SIZE=1`, no
`TPU_MOE_HIERARCHICAL_EP`) and `--expert-placement-strategy linear` (the default).
A table helps only when the busiest rank's expert load sets the MoE time; at low
token counts per rank every rank streams most of its experts and a permutation
changes little.
