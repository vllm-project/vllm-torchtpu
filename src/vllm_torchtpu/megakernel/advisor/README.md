# TPU Megakernel Fusion-Scope Advisor

The `vllm_torchtpu.megakernel.advisor` package evaluates whether a model and serving workload benefit from fusing operators across a layer, layer group, or full step on TPU, using hardware specifications, model geometry, and optional trace calibration.

## Two-Step Workflow

### 1. Prior Estimate (`estimate`)

Evaluate HBM fit, the ideal-kernel VMEM token ceiling, roofline step floor, and uncertainty bands without a TPU or `jax` installation:

```bash
python -m vllm_torchtpu.megakernel.advisor estimate \
  --model qwen3.5-397b --chip tpu7x --phase decode --batch 16 --tp 8 \
  --explain --unresolved
```

### 2. Trace-Measured Evaluation (`--measurements`)

Pass an `advisor.measurements.v1` JSON file containing trace-measured launch gaps, collective hop latencies, step coverage, or DCN bandwidth to promote `ASSUMED` inputs to `MEASURED` and settle `tie(measure: <input>)` verdicts. You may also pass `--measurements bundled` (or `--measurements auto`) to load bundled public chip priors (such as `tpu7x_public.json`):

```json
{
  "schema": "advisor.measurements.v1",
  "source": {"kind": "xplane", "file": "profile.xplane.pb"},
  "launch_gap_us": {"p50": 3.5, "p90": 5.0, "n": 40},
  "collectives": [
    {"op": "all_reduce", "group_size": 8, "bytes": 32768, "latency_us": 12.0, "n": 20}
  ],
  "step_coverage": {"fused_eligible_fraction": 0.88, "basis": "trace"}
}
```

```bash
python -m vllm_torchtpu.megakernel.advisor estimate \
  --model qwen3.5-397b --chip tpu7x --measurements measurements.json \
  --json report.json
```

## Provenance Tags

Hardware specifications (`vmem_capacity_bytes`, `hbm_bw_bytes_per_s`, `bf16_flops_per_s`) and model geometry constants are displayed in the Sourced Hardware & Model Architecture Specifications header card. Every computed metric in `report.metrics` carries an uncertainty provenance tag, unit, source string, and `[low, high]` band:

| Tag | Meaning |
| --- | --- |
| `DERIVED` | Closed-form function of hardware specs, model geometry, or `MEASURED` quantities (`tokens_per_core`, `onchip_ceiling`, `step_floor_us`). |
| `ASSUMED` | Uncalibrated prior bracket (`launch_gap_us`, `collective_latency_us`, `step_coverage`, `dcn_bw_gbps`) with explicit `[low, high]` bounds. |
| `MEASURED` | Quantity populated from `--measurements` (`advisor.measurements.v1`). |

## Key CLI Flags

| Flag | Description |
| --- | --- |
| `--model` / `--config` | Registered preset name, HuggingFace repo ID, local directory, or `config.json` path. |
| `--chip` / `--chip-spec` | Built-in TPU target (`tpu7x`, `v6e`, `auto`) or path to a custom JSON chip spec. |
| `--measurements` | Path to an `advisor.measurements.v1` JSON file, or `bundled`/`auto` for packaged chip priors. |
| `--objective` | `step` (minimize step latency), `throughput` (maximize `tok/s/chip`), or `both` (flag objective splits). |
| `--max-rows-per-call` | Most decode or verify rows the megakernel serves in one call. A larger batch is priced as separate calls that each stream their weights again. `0` (default) means no declared limit. |
| `--explain` / `--unresolved` | Print per-metric provenance tags or list unresolved `ASSUMED` inputs ranked by step-time sensitivity. |
| `--json [out.json]` | Write JSON output to stdout (when path omitted) or to `out.json`. |
| `--html out.html` | Write a self-contained inline-SVG HTML regime report to `out.html`. |

## Estimate vs. Strict Floor and Support Contract

`step_floor_us` (`ScopeEstimate.step_bound_us`) is a uniform-routing roofline estimate, paired with `step_floor_strict_us` (strict token-proportional routing lower bound) and `step_achievable_us` / `step_achievable_upper_us` (derated achievable band):

- **Support Classes (Default-Deny Guard)**:
  - `baseline`: `per_op` execution (always runnable).
  - `within_core_limits`: `whole_step` within `CORE_LIMITS` (`max_batch_rows = 128`, `verified_ranks = (1, 2, 4, 8, 16)`, `chip_kinds = ("TPU7x",)`, `ep_dp == 1`, recognized `model_type`, and all layer families in `SUPPORTED_ONCHIP_FAMILIES`).
  - `declared`: explicitly declared in `data/presets/deployments.json` under `"megakernel"`.
  - `hypothetical`: any fused scope outside `CORE_LIMITS` without a preset declaration; `recommended_scope` stays `per_op` while `fastest_scope` and `hypothetical_best` report the physics winner and its `blockers`.
- **Extension Contract**: update `CORE_LIMITS` in `support.py` when `megakernel/core/` expands its verified grid, or add a `"megakernel"` entry to `data/presets/deployments.json` once a workload is validated end-to-end.
- **Rows per Call**: `t_vmem_max` for `whole_step` is the VMEM ceiling of an ideal kernel (aliased scratch, minimum ring). It is an upper bound on rows per call, not a served limit. `--max-rows-per-call`, or `max_rows` in a `"megakernel"` preset entry, caps it for decode and verify. Rows above the cap are priced as full calls plus a remainder, split in whole requests, and the report carries the `microbatch_rerun` flag. The megakernel core's default buckets stop at 32 decode rows and 64 verify rows; a larger call carries the `above_default_buckets` flag.
- **Collectives**: a layer requests its routed-expert weights after its router runs, so collectives add to the routed-expert stream instead of overlapping it. All-reduce transfer is priced on one ICI link, as a ring or dimension-ordered all-reduce uses it; `step_floor_strict_us` keeps the egress sum of all links.

- Optimistic floor (`band[0]`): peak hardware FLOP/s (FP8 peak on native-FP8 chips when weights are FP8) and lower-bound collective latency.
- Conservative upper bound (`band[1]` / `step_upper_us`): BF16 attention dot throughput on prefill plus upper-bound launch gap and collective hop latency.
- Every scope is priced at each corner of the same input box (launch gap, collective hop latency, prefill attention dot dtype); the all-low corner is `step_bound_us` and the all-high corner is `step_upper_us`. `verdict` reports `tie(measure: <input>)` only when the fastest scope changes between corners, naming the assumed input that settles it; `rationale` names the competing scope and corner.
- `--measurements` replaces the launch gap and collective latency bands with trace values, which narrows the bands and the verdict.
