# PR 2821 MoE overlay reproduction

This standalone distributed UT reproduces `overlayer-overhead` spans in the
MoE path of [PR #2821](https://github.com/vllm-project/vllm-torchtpu/pull/2821).
The branch is based on PR head `3db01e3b0270986075109d650dbda90f77ed2ff6`.
It imports the branch's existing kernels without modifying them. No model
checkpoint, inference server, or external kernel package is needed.

## Observed issue

In a warmed-up full MoE invocation, the first GMM has a long tail near its
`ep_finalize` region. Fine TraceMe identifies a roughly **15.7 microsecond**
`overlayer-overhead` span inside GMM1. For example, a captured rank0 TensorCore
invocation had **115.33 microseconds** total GMM1 time and **15.66 microseconds**
in this single overlay span.

The kernel names in that capture were:

```text
gmm_v2-g_16-m_384-k_8192-act_silu-n_4096-tm_128-tk_8192-tn_2048.2
gmm_v2-g_16-m_384-k_2048-act_None-n_8192-tm_256-tk_2048-tn_8192.2
```

A separate capture without fine pipeline scopes measured roughly **445.10 us**
for the full MoE invocation, with **20.58 us** of `overlayer-overhead` across
that invocation (mean across the profiled TensorCore planes and the final three of five
calls). These are observations, not portable performance thresholds.

The marker alone does **not** establish that all of this time is instruction
loading from HBM to IMEM, nor that subtracting it predicts the speed of a
modified program. The purpose of the reproducer is to expose the interval for
compiler/runtime investigation. Inspect sync-wait tracks and LLO before
attributing a cause. Profiler settings can also change generated code and
execution time; use the separate timing mode for unprofiled measurements.

## Workload and environment

- Four TPU7x hosts, 2 x 2 x 4 chip topology; 32 logical ranks, eight per host.
- Logical mesh `(dp=8, tp=4)`, consecutive groups of four ranks share an input.
- Compile capacity: 128 global requests, MTP K=3, hence 512 global token rows.
- Active workload: 96 global requests, hence 384 valid token rows.
- Each DP shard: BF16 `[64, 8192]` input, 48 valid rows and 16 padded rows.
  Padding remains nonzero in the input; its routing weights are masked to zero.
  The PR still includes those rows' expert IDs in dispatch capacity accounting.
- 512 routed experts, top-10, 16 experts per logical rank; FP4 packed weights,
  FP32 scales, SiLU activation. Local GMM1 logical weight shape is
  `[16, 8192, 4096]`, GMM2 is `[16, 2048, 8192]`.
- Shared expert: FP8 weights, two TP-sharded matmuls plus sigmoid gate.
- PR ragged all-to-all capacity factor 2.0; sparse-core gather/reduce v2;
  one-hot permutation threshold 3840; dense-expert and owner-output paths off.
- Deterministic seeds: router 271; shared expert `731 + rank % 4`;
  routed weights `991 + rank`; inputs `1000 + dp_rank`.
- The original capture used torch `2.13.0+cpu`, torch-tpu
  `0.1.1.dev20260928182719`, JAX/jaxlib `0.10.2`, libtpu `0.0.47`.
  A different compiler/libtpu can change or remove the phenomenon.

`adapter.py` supplies routing and the shared expert around the PR's
`ragged_a2a_moe_local`. It does not instantiate an entire vLLM model. The UT
checks that capacity does not overflow, compares active outputs against the
same PR's full-capacity path (`capacity_factor=0`, rtol=0.03, atol=0.001), then
warms the measured path for 20 calls. The reference is a capacity-path check,
not an independent mathematical MoE correctness oracle.

## Run on four hosts

Use a working torch-tpu/vLLM environment on a reserved four-host slice. Check
out this branch at the same absolute path on every host. Retain the runtime's
normal TPU device visibility and topology settings. No server may occupy the
same devices. The launcher uses the current Python executable and puts this
checkout's `src/` first on `PYTHONPATH`.

`--worker-env-json` accepts the eight worker environments for the **local**
host from a known-working torch-tpu launcher, in this format:

```json
[
  {"env": {"RANK": "0", "LOCAL_RANK": "0"}},
  {"env": {"RANK": "1", "LOCAL_RANK": "1"}}
]
```

The illustration is abbreviated: provide all eight workers, including your
runtime's per-worker TPU visibility/topology variables. Global rank ranges
must be 0-7, 8-15, 16-23, and 24-31 on the four hosts respectively. Inherited
host environment is retained; the launcher overrides rendezvous, mesh worker
addresses, selectors, and profiling flags. Do not distribute credentials or
unrelated service environment variables with a reproduction report.

Run this concurrently on **all four hosts**, substituting the same four ordered
IP addresses and session ID, but each host's local worker JSON:

```bash
python tools/overlayer_repro/launch.py \
  --hosts HOST0_IP,HOST1_IP,HOST2_IP,HOST3_IP \
  --worker-env-json /path/to/local-worker-envs.json \
  --output /tmp/pr2821-overlay-profile \
  --session-id 1793618000000000081 \
  --profile
```

Use a fresh output directory and unique shared session ID on each run. Default
rendezvous port is 29718; slice-builder ports start at 19440. They can be
changed with `--master-port` and `--slice-port-base`. Check `exit-codes.json`
and all eight local `rank*/done` files on **every** host.

The launcher sets these tracing flags before worker imports:

```text
--xla_enable_transpose_trace=true
--xla_tpu_profile_traceme_level=10
--xla_tpu_show_overlay_details_in_profiler=true
--xla_tpu_show_overlay_overhead_in_profiler=true
--xla_tpu_show_overlay_waits_in_profiler=true
--xla_tpu_show_overlay_waits_in_xprof=true
--xla_xprof_register_llo_debug_info=true
--xla_xprof_enable_custom_call_tracing=false
```

These flags are libtpu-version-specific. With the recorded libtpu, ragged A2A
also needs `--xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false`;
the launcher sets this explicitly. It does not disable all sparse-core work.
The effective launch flags are saved in `flags-rank*.json`.

The profiler uses `TRACE_COMPUTE_AND_SYNC`, host/device tracer levels 1 and
Python tracer level 0. It captures exactly **five** calls after compilation
and warm-up, synchronizes the TPU, and closes the capture on every rank.
Rank0 additionally gets text HLO dumps under `rank0/xla_dump/`.

## Verify and open the profile

On the rank0 host, using the same environment and checkout:

```bash
PYTHONPATH=src python tools/overlayer_repro/analyze.py \
  /tmp/pr2821-overlay-profile/rank0/profile \
  --output /tmp/pr2821-overlay-profile/rank0/overlay-analysis.json
```

The analyzer verifies `ep_copy_in`, `ep_wait_in`, `ep_run_kernel`,
`bm64_first_last`, and `ep_finalize`. It reports each TensorCore separately,
drops the first two calls, and reports the last three calls' full module time,
individual `overlayer-overhead` spans, and their interval union. It does not
add nested `overlayer-get-or-add` or `overlayer-cut-and-add` spans to the total.
Absence of overlay on a different runtime is a result, not an analyzer error.

Copy **one rank's entire `profile/` directory**, preserving its directory
structure, to the viewing machine. With XProf installed:

```bash
xprof --logdir /path/to/rank0/profile --port 18923
```

Open `http://localhost:18923`, select `pr_capacity_host100/<timestamp>`, and
open Trace Viewer. The `host100` suffix is a legacy case label: profiling
contains only five invocations. Expand **XLA Ops**, **XLA TraceMe**, and
**Tensor Core Sync Flag**. Search for GMM1, `ep_finalize`, and
`overlayer-overhead`; inspect the end of GMM1 and the transition into GMM2.
The presence of ordinary `ep_*` scopes establishes that fine tracing is active.
Raw trace-event counts include multiple cores and must not be interpreted as
MoE invocation counts.

For a small report, share rank0's `profile/`, `overlay-analysis.json`,
`capacity.json`, `results.json`, `topology.json`, effective flags, package
versions, and the checkout commit. Full HLO dumps are optional and larger.

## Timing without a profiler session

Run the same launcher with a new output/session ID and **omit `--profile`**.
It disables fine TraceMe/transpose tracing and does not create a profiler
session. Overlay visibility flags remain set for consistency with the recorded
experiment. After 20 warm-ups, it measures ten rounds of 100 host-issued calls,
synchronizes at each round's end, and gathers the maximum elapsed time across
all 32 ranks. `rank*/timing.json` contains per-round microseconds per call and
their mean. This includes host dispatch effects; it is not a device-side loop
measurement and must not be conflated with a sum of trace scopes.

## Standalone harness verification

The standalone harness was rerun on the four-host slice on 2026-11-06.
Rank0's last three calls on each of its two TensorCore planes showed:

| TensorCore | Full invocation (us) | Overlay union (us) | Longest span (us) |
| --- | --- | --- | --- |
| 0 | 442.980 | 20.455 | 15.671 |
| 0 | 444.972 | 20.685 | 15.886 |
| 0 | 445.161 | 20.482 | 15.691 |
| 1 | 443.082 | 20.579 | 15.671 |
| 1 | 444.934 | 20.658 | 15.766 |
| 1 | 445.178 | 20.710 | 15.880 |

All required fine TraceMe scopes were present on both planes. Rank0's
capacity diagnostic reported 288 token rows and 384 route slots, no overflow,
151 received token rows and 176 routes. Its active outputs matched the PR's
full-capacity reference exactly for this synthetic input. These figures are
from the standalone harness, rather than a serving benchmark.
