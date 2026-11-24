# TPU Megakernel Fusion-Scope Advisor

The `vllm_torchtpu.megakernel.advisor` package evaluates whether a model and serving workload benefit from fusing operators across a layer, layer group, or full step on TPU, using hardware specifications, model geometry, and optional trace calibration.

## Two-Step Workflow

### 1. Prior Estimate (`estimate`)

Evaluate HBM fit, the ideal-kernel VMEM token ceiling, roofline step floor, and uncertainty bands on CPU without an attached TPU or PyTorch runtime:

```bash
python -m vllm_torchtpu.megakernel.advisor estimate \
  --model qwen3.5-397b --chip tpu7x --phase decode --batch 16 --tp 8 \
  --explain --unresolved
```

### 2. Trace-Measured Evaluation (`--measurements`)

Pass an `advisor.measurements.v1` JSON file containing trace-measured launch gaps, all-reduce call latencies, step coverage, or DCN bandwidth to promote `ASSUMED` inputs to `MEASURED` and settle `tie(measure: <input>)` verdicts. You may also pass `--measurements bundled` (or `--measurements auto`) to load bundled `architectural_prior` chip defaults (`tpu7x_public.json` on `tpu7x`; errors on `v6e`), which leave step bounds unchanged until a real trace replaces priors:

```json
{
  "schema": "advisor.measurements.v1",
  "source": {"kind": "xplane", "file": "profile.xplane.pb"},
  "launch_gap_us": {"p50": 3.5, "p90": 5.0, "n": 40},
  "collectives": [
    {
      "op": "all-reduce",
      "path": "xla",
      "group_size": 8,
      "bytes": 32768,
      "latency_us": 12.0,
      "mean_latency_us": 13.5,
      "n": 20
    }
  ],
  "step_coverage": {"fused_eligible_fraction": 0.88, "basis": "trace"}
}
```

```bash
python -m vllm_torchtpu.megakernel.advisor estimate \
  --model qwen3.5-397b --chip tpu7x --measurements measurements.json \
  --json report.json
```

Extract a measurements file directly from a trace:

```bash
python -m vllm_torchtpu.megakernel.advisor calibrate \
  --trace <trace.xplane.pb> --out measurements.json
```

The `calibrate` command matches collectives by event-name substring across device-plane lines without recording the source line. It emits `latency_us` (median) and `mean_latency_us` (mean) per bucket; when explicit `Steps` markers are absent, it trims the primary `XLA Modules` program window on each device rank to `[first_op_start, last_op_end]` and emits the pre-first-op host/dispatch gap as `host_head_us`. Collective entries written by `calibrate` omit `path` and load as `xla`; an `in_kernel` entry is written by hand from an in-kernel measurement. Calibrated `all-reduce` entries (sources other than `architectural_prior`) apply only when `path` (`xla` for `per_op`, `in_kernel` for fused scopes) and `group_size` match the workload's tensor-parallel rank count, charging `mean_latency_us` per call when present and `latency_us` otherwise in `step_achievable_us` while keeping hardware floors (`step_floor_strict_us` and `step_bound_us`) strictly analytical. `collective_latency_us` remains the chip's per-hop prior (`ASSUMED`), while `collective_call_us_xla` and `collective_call_us_in_kernel` report the single-call all-reduce cost on each path (`MEASURED` when matched, `ASSUMED` otherwise, or `DERIVED` on one rank).

## Provenance Tags

Hardware specifications (`vmem_capacity_bytes`, `hbm_bw_bytes_per_s`, `bf16_flops_per_s`) and model geometry constants are displayed in the Sourced Hardware & Model Architecture Specifications header card. Every computed metric in `report.metrics` carries an uncertainty provenance tag, unit, source string, and `[low, high]` band:

| Tag | Meaning |
| --- | --- |
| `DERIVED` | Closed-form function of hardware specs, model geometry, or `MEASURED` quantities (`tokens_per_core`, `onchip_ceiling`, `step_floor_us`). |
| `ASSUMED` | Uncalibrated prior bracket (`launch_gap_us`, `collective_latency_us`, `step_coverage`, `dcn_bw_gbps`) with explicit `[low, high]` bounds. |
| `FACT` | Hardware or model specification value read from chip info or a model config snapshot. |
| `CHOSEN` | Explicit user CLI/workload selection (parallelism layout, batch, context, or dtype override). |
| `MEASURED` | Quantity populated from `--measurements` (`advisor.measurements.v1`). |

## Key CLI Flags

| Flag | Description |
| --- | --- |
| `--model` / `--config` | Registered preset name, HuggingFace repo ID, local directory, or `config.json` path. |
| `--list-models` | Print every shipped preset grouped by megakernel support tier, chip core-budget coverage, dtype basis, and unmodeled caveats. |
| `--chip` / `--chip-spec` | Built-in TPU target (`tpu7x`, `v6e`, `auto`) or path to a custom JSON chip spec. |
| `--measurements` | Path to an `advisor.measurements.v1` JSON file, or `bundled`/`auto` for packaged `architectural_prior` defaults (`tpu7x` only; step bounds stay unchanged until a real trace is supplied). |
| `--decoding-seqs` / `--batch` | Active sequences per step (`output_tok_s * mean_tpot_s`), not open client concurrency. Alternatively pass `--output-tok-s <rate> --mean-tpot-ms <ms>` to derive `B` via Little's Law. |
| `--host-head-us` | Optional per-step host dispatch head `per_op=<us>,fused=<us>` (default `0,0`). When `per_op` wins pure device physics, reports show the break-even host-head delta. |
| `--per-op-step-us` / `--step-coverage` | Measured baseline `per_op` E2E step latency (`us`) and fused-eligible step fraction (`(0, 1]`), promoting `gain_basis` from `fused_scope_device_estimate` to `calibrated_whole_server_e2e` when both are `MEASURED`. |
| `--objective` | `step` (minimize step latency), `throughput` (maximize `tok/s/chip`), or `both` (flag objective splits). |
| `--max-rows-per-call` | Most decode or verify rows the megakernel serves in one call. A larger batch is priced as separate calls that each stream their weights again. `0` (default) means no declared limit. |
| `--explain` / `--unresolved` | Print per-metric provenance tags or list unresolved `ASSUMED` inputs ranked by step-time sensitivity. |
| `--json [out.json]` | Write JSON output to stdout (when path omitted) or to `out.json`. |
| `--html out.html` | Write a self-contained inline-SVG HTML regime report to `out.html`. |

## Estimate vs. Strict Floor and Support Contract

`step_floor_us` (`ScopeEstimate.step_bound_us`) is a uniform-routing roofline estimate, paired with `step_floor_strict_us` (strict token-proportional routing lower bound) and `step_achievable_us` / `step_achievable_upper_us` (derated achievable band):

| Report label | Field | Meaning |
| --- | --- | --- |
| `Peak Floor` | `step_bound_us` | Step time at peak chip rates (`eta = 1.0`) under uniform expert routing, with the lower launch-gap prior. A lower bound when routing is no more concentrated than uniform. |
| `Peak Upper` | `step_upper_us` | The same work at peak rates under the upper priors for launch gaps, collective hops, and the matrix path. |
| `Derated Achievable` | `step_achievable_us` to `step_achievable_upper_us` | An estimate: the same work with the efficiency factors `eta_hbm`, `eta_dense`, `eta_moe_grouped`, and `eta_ici` applied. Not a measurement. It can exceed `Peak Upper`. |
| strict floor | `step_floor_strict_us` | A lower bound for any expert routing. |

The per-row ledger in the HTML report uses the same labels.

- **Support Classes (Default-Deny Guard)**:
  - `baseline`: `per_op` execution (always runnable).
  - `within_core_limits`: `whole_step` within `CORE_LIMITS` (`max_batch_rows = 128`, `verified_ranks = (1, 2, 4, 8, 16)`, `chip_kinds = ("TPU7x",)`, `max_ar_slots_per_layer = 2`, `ep_dp == 1`, recognized `model_type`, periodic layer schedule, all layer families in `CORE_MODULE_FAMILIES`, and weight and expert dtypes in `CORE_SUPPORTED_DTYPES`).
  - `declared`: explicitly declared in `data/presets/deployments.json` under `"megakernel"` or `"megakernel_layouts"` for the workload's rank count and serving layout.
  - `hypothetical`: any fused scope outside `CORE_LIMITS` without a preset declaration; `recommended_scope` stays `per_op` while `fastest_scope` and `hypothetical_best` report the physics winner and its `blockers`.
- **Layout Matching & Declared Collectives**: pass the 6 serving layout flags (`tp`, `ep`, `dp`, `pp`, `dcp`, `pcp`) that the server runs with. Declarations match both `ranks` and all 6 serving axes (`layout_not_declared` on mismatch), and may specify `collectives` (`all_reduces` plus `routed_transport`, `split_kv`, and `dcp_gather` steps) and `expert_mode_by_rows` (`tp` vs `ep` row thresholds). A kernel that reshards experts inside the fused call is declared on its serving layout, so passing its internal expert mode as the layout yields `layout_not_declared`.
- **Extension Contract**: update `CORE_LIMITS` in `constants.py` and `CORE_MODULE_FAMILIES` / `CORE_SUPPORTED_DTYPES` in `specs.py` when `megakernel/core/` expands its verified grid, or add a `"megakernel"` entry (on `decode`, `prefill`, `saturation`, or `megakernel_layouts`) to `data/presets/deployments.json` once a workload is validated end-to-end.
- **Rows per Call**: `t_vmem_max` for `whole_step` is the VMEM ceiling of an ideal kernel (aliased scratch, minimum ring). It is an upper bound on rows per call, not a served limit. `--max-rows-per-call`, or `max_rows` in a `"megakernel"` preset entry, caps it for decode and verify. Rows above the cap are priced as full calls plus a remainder, split in whole requests, and the report carries the `microbatch_rerun` flag. The megakernel core's default buckets stop at 32 decode rows and 64 verify rows; a larger call carries the `above_default_buckets` flag. An undeclared `whole_step` call above that bucket top without `--max-rows-per-call` is marked `hypothetical` with `row_limit_not_stated` and lists `max_rows_per_call` as an unresolved input (`affects: "support"`).
- **Collectives and Achievable Composition**: on single-pass (`n_splits == 1`) fused `whole_step` and `layer_group` scopes, `step_bound_us` and `step_upper_us` overlap sublayer collectives behind the weight stream once each layer's router runs (`max(stream, chain + collectives)`), while `per_op`, `single_layer`, and row-tiled (`n_splits > 1`) fused scopes add collectives to `max(stream, chain)`. On `step_achievable_us` and `step_achievable_upper_us`, synchronous sublayer barriers add to `max(stream, chain)`, routed-expert stream and compute scale to the busiest expert owner group (`expert_owner_groups` in `deployments.json`, the disjoint expert sets each shared by a group of tensor-parallel ranks, defaulting to one group per `ep` shard when unset), and every scope derates inter-call cold-slot fills by `eta_hbm` and boundary sync hops by `eta_ici` (with `per_op` counting unfused operator kernel launches per layer). All-reduce transfer is priced on one ICI link, as a ring or dimension-ordered all-reduce uses it; `step_floor_strict_us` keeps the egress sum of all links.

- Optimistic floor (`band[0]`): peak hardware FLOP/s (FP8 peak on native-FP8 chips when weights are FP8) and lower-bound collective latency.
- Conservative upper bound (`band[1]` / `step_upper_us`): BF16 attention dot throughput on prefill plus upper-bound launch gap, collective hop latency, boundary fill/drain (`BOUNDARY_SYNC_HOP_FACTOR` + cold first-slot HBM), and per-op collective call factor (`PER_OP_COLLECTIVE_CALL_FACTOR`).
- Every scope is priced at each corner of the same 5-axis input box (`launch_gap_us`, `collective_latency_us`, `prefill_attn_dot_dtype`, `boundary_sync_hop_factor`, `per_op_collective_call_factor`); the all-low corner is `step_bound_us` and the all-high corner is `step_upper_us`. `verdict` reports `tie(measure: <input>)` only when the fastest scope changes between corners, naming the assumed input that settles it; `rationale` names the competing scope and corner.
- `--measurements` replaces launch-gap priors in the corner box (narrowing launch-gap bands and verdict) and applies measured per-path all-reduce call latency in `step_achievable_us` while keeping `step_floor_strict_us` and `step_bound_us` strictly analytical.

## Model Support Tiers

Run `--list-models` to print every shipped preset grouped by tier, its dtype basis, the sublayer meshes of presets that have a deployment entry, and its unmodeled caveats. On chips without core budgets (`v6e`), all fused scopes are gated by `chip_without_core_budget` and priced from hardware rates only.

| Tier | Meaning |
| --- | --- |
| `declared` | A megakernel is declared for at least one serving layout. |
| `priced_only` | Priced by the advisor. A layer family or weight dtype has no megakernel/core counterpart. |

## Adding a Model

Only validated presets (`qwen3.5-397b`, `kimi-k3`, `glm-5.2`) are accepted; any other config or model fails closed with `UnsupportedArchitectureError` so unmodeled layer semantics or serving quirks are never silently priced.

To onboard a new model:

1. Save the full authentic `config.json` (untrimmed, wrapped with `repo_id` and `revision`) under `tests/megakernel/data/advisor/hf_configs/`.
2. Add the verified preset and aliases to `data/presets/models.json` with `dtype_basis: "config"`.
3. Add a deployment entry in `data/presets/deployments.json` (and a `megakernel` block with a verifiable `source_ref` if a fused runtime exists).
4. Record a serving or device-trace validation point and run the advisor test suite (which checks every shipped preset against its full authentic snapshot and verifies unsupported configs fail closed).

## Config Files

`--config` reads quantization skip lists (`ignore`, `modules_to_not_convert`) and nested `dsa_config` blocks, and matches the parsed geometry and config dtypes against the shipped presets. A config matching a shipped preset adopts that preset's name, layouts, and declarations and prints one confirmation line on stderr. Any config that does not match a validated preset raises `UnsupportedArchitectureError` listing the supported repositories and pointing to the onboarding section above. Passing a preset name with CLI dtype overrides keeps that preset's declarations.

## Drift Gates

- Snapshot tests compare `data/presets/models.json` fields against the checked-in `hf_configs` snapshots.
- `verify_core_limits_sync()` imports `megakernel/core` and cross-checks core limits, implemented core module packages, and default bucket tops against `CORE_LIMITS`.
- Built-in chip facts and sublane row tiles are read directly from JAX's `jax.experimental.pallas.tpu.get_tpu_info_for_chip()` in `chips.py`, and tests verify that every non-JAX field is explicitly enumerated.
- Open request to the core owners: the accepted weight dtypes (`CORE_SUPPORTED_DTYPES`) have no exported symbol name in `megakernel/core/`, so they are not automatically checked by `verify_core_limits_sync()` (the per-layer all-reduce slot limit `max_ar_slots_per_layer` is verified behaviorally against `planner.plan` and `ring_scratch`).
