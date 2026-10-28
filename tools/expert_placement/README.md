# Expert placement tools

These tools build a `TPU_EXPERT_PLACEMENT` table (see "Static Expert Placement" in
`docs/user_guide/parallelism.md`) from the routing of your own workload, in five steps:
profile, aggregate, solve, validate, install. Run them from the repository root inside the
vllm-torchtpu image (they need numpy and scipy; the counter and the validator also need
torch and vllm_torchtpu).

| Step | Command | Output |
| :--- | :--- | :--- |
| 1. Profile | serve with the routing counter on | one `ep_routing_counts_rank<r>_pid<p>.npz` per worker process |
| 2. Aggregate | `aggregate_counts.py` | `counts.npz`: per-layer expert hits of decode and prefill steps |
| 3. Solve | `solve_placement.py` | `table.json`, `table.json.sha256`, `table.json.report.json` |
| 4. Validate | `validate_placement.py` | `OK ...` line with the sha256 the loader will log |
| 5. Install | `TPU_EXPERT_PLACEMENT=table.json vllm serve ...` | one `[eplb] placement installed` line per EP rank |

## 1. Profile

The counter is a `sitecustomize` drop-in: nothing in `vllm_torchtpu` imports it. It wraps
`moe_tp_decode_forward` (layers that `moe_tp_decode` armed route inside its op) and
`moe_routing.route` (all other layers), and adds each forward's top-k expert ids into one
int32 device buffer inside the compiled program. Routing outputs are not changed.

```bash
export PYTHONPATH=$PWD/tools/expert_placement/routing_counter:$PYTHONPATH
export EP_ROUTING_COUNTER=1 EP_ROUTING_COUNTER_DIR=/tmp/ep_counts
export VLLM_CACHE_ROOT=/tmp/vllm-cache-counter   # fresh: a cache built without the counter has no counter
vllm serve zai-org/GLM-5.2-FP8 --tensor-parallel-size 16 --enable-expert-parallel ...
# run the workload, wait for the idle dump (below), then stop the server
```

- Set the variables before `ray start` on every host so the workers inherit them. Each
  host writes to its own `EP_ROUTING_COUNTER_DIR`; copy every host's directory under one
  root for step 2.
- The counter needs `data_parallel_size` 1 and `prefill_context_parallel_size` 1; otherwise
  it logs `NOT counting` and records nothing.
- Each worker rewrites its cumulative dump between steps, at most every
  `EP_ROUTING_COUNTER_INTERVAL_S` seconds (default 60), once more after the server has been
  idle for that long, and at exit. Workers that are killed skip the exit dump, so wait for
  the idle dump after the last request before stopping the server.
- Check delivery in the worker logs: one `[ep_routing_counter] counting <n> MoE layers ...`
  line per worker after weight loading, then periodic `rank <r>: <hits> hits ... (decode
  <d>, prefill <p>, padded <x>, unknown <u>)` lines. While requests are served, decode hits
  must grow and unknown hits must stay at 0.
- To drop warm-up from the profile, wait for the idle dump after warm-up, copy
  `/tmp/ep_counts` aside (for example to `/tmp/ep_counts_before`) before the workload
  starts, and pass it to `--subtract`.

## 2. Aggregate

```bash
python tools/expert_placement/aggregate_counts.py /tmp/ep_counts -o counts.npz \
  --subtract /tmp/ep_counts_before   # optional
```

It prints the layers found, decode and prefill hits and step counts, and checks:
`unknown_hits` must be 0 and `step_counters_equal_across_layers` true; with `--subtract`,
`pad_mismatch_rows` must be 0 too (warm-up forwards add some by construction). Draft (MTP) layers
are left out: the loader never places them.

## 3. Solve

```bash
python tools/expert_placement/solve_placement.py --counts counts.npz --ranks 16 \
  --out table.json --layers 3-77 --procs 8 --model zai-org/GLM-5.2-FP8 [--cost cost.json]
```

Per layer it minimizes the expected MoE time of the busiest rank over the decode and prefill
steps of the profile (model in `placement_model.py`), starting from two greedy placements and
applying pairwise swaps; a layer is never worse than vLLM's linear placement on that
objective. It prints the objective saved versus linear. Expect seconds to about a minute
per 256-expert, 16-rank layer on one core; `--layers` defaults to every layer with data.

`--cost` takes a JSON object with any of these fields:

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `decode_us`, `decode_us_per_active_expert` | 31.9, 11.3 | decode time per layer and rank: a + b x (experts with at least one token) |
| `prefill_us`, `prefill_ns_per_pair` | 200.0, 53.0 | prefill time per layer and rank: a + b x (token, expert) pairs |
| `decode_tokens` | from the counts | `{"<tokens per decode step>": relative frequency, ...}` |
| `decode_weight` | from the counts | decode steps, on the same scale as the prefill weights |
| `prefill_steps` | from the counts | `[[tokens per step, steps], ...]`; give it together with `decode_weight` |
| `decode_min_tokens` | 0 | decode steps below this many tokens count as placement-independent |

The cost defaults were measured for GLM-5.2-FP8 on a 16-chip v7x slice with flat EP16;
measure them for another model or slice. Only their ratios affect the search. The step-mix
defaults come from the counter: one decode bin at the mean tokens per decode step and one
prefill type at the mean tokens per prefill step, weighted by step counts.

## 4. Validate

```bash
python tools/expert_placement/validate_placement.py table.json --ranks 16 \
  --hf-config /path/to/GLM-5.2-FP8/config.json
```

It runs the loader's own checks (`expert_placement.parse`, `check_layers` and
`check_routing`): every listed layer a permutation, the deployment's experts, ranks and
slots, only routed decoder layers, no grouped top-k or hash routing. Without a config, pass
`--experts 256 --first-moe-layer 3 --num-layers 78`. Exit 0 prints
`OK ... sha256 <64 hex>; expect table_sha256=<16 hex> ...`; exit 1 prints why it refused.

## 5. Install

```bash
TPU_EXPERT_PLACEMENT=$PWD/table.json \
  vllm serve zai-org/GLM-5.2-FP8 --tensor-parallel-size 16 --enable-expert-parallel ...
```

The table must exist at that path on every host (or use a `gs://` URI). Check that every
expert-parallel worker logs
`[eplb] placement installed ep_rank=<r> layers=<n> table_sha256=<16 hex>` with the 16 hex
digits from step 4. The loader's requirements (flat expert parallelism, linear
`--expert-placement-strategy`, and so on) are listed in `docs/user_guide/parallelism.md`.

## Limits

- The fused EP kernel (`USE_MOE_FUSED_EP_KERNEL`) routes inside its own program; its layers
  are not counted (the counter logs a warning). Any other path that routes without
  `moe_tp_decode_forward` or `moe_routing.route` is missed too; the step 1 check (decode hits
  grow while serving) catches this.
- A prefill chunk of at most 1 + `num_speculative_tokens` tokens looks like a decode request
  in the attention metadata and is counted as decode.
- The counter adds a top-k and a scatter-add per MoE layer and forward: do not use a profiling
  run's timings as a baseline.
- Tables from a profile at one concurrency or prompt mix may help less at another; profile
  the workload you serve.
