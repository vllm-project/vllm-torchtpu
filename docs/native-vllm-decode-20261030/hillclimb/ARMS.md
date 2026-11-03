<!-- SPDX-License-Identifier: Apache-2.0 -->

# Arms

Run in this order; each arm is one server launch on all four hosts plus one `bench_sweep.py`
run. Record the `SUMMARY` line and the four server logs of every arm under
`results/<arm>/`. Stop and report if an arm fails to start; do not retune it.

Reference: the 2026-10-30 baseline measured 879.9 tok/s at 109.1 ms steady TPOT.

| # | arm | LAYOUT | GLOBAL_SEQS | MTP_K | other env | isolates |
|---|---|---|---|---|---|---|
| 1 | `base-dp8tp4-s96-k3` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776` | reproduces the baseline on this build (sanity) |
| 2 | `base-dp8tp4-s96-k0` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776` | cost of the 3-token draft with dummy weights (should be the floor for every K > 0 arm) |
| 3 | `tp32dcp8-s96-k3` | tp32dcp8 | 96 | 3 | | one wide engine: no dense-weight replication, no DP lockstep, 8-way page stripes |
| 4 | `tp32dcp8-s96-k0` | tp32dcp8 | 96 | 0 | | same without the draft |
| 5 | `dp4tp8dcp2-s96-k3` | dp4tp8dcp2 | 96 | 3 | | one engine per host: all TP collectives inside a host, 2-way page stripes |
| 6 | `tp32dcp8-s192-k3` | tp32dcp8 | 192 | 3 | | batch amortization: the per-layer fixed costs are paid once per step whatever the batch; the roofline predicts +60% tok/s for +25% TPOT |
| 7 | `dp4tp8dcp2-s160-k3` | dp4tp8dcp2 | 160 | 3 | | same lever at 40 per engine (capacity about 44) |
| 8 | `tp32dcp8-s96-k3-fusedep` | tp32dcp8 | 96 | 3 | `FUSED_EP=v2` | fused EP MoE program replacing the per-layer all-gather + reduce-scatter |
| 9 | `tp32dcp8-s96-k3-sync` | tp32dcp8 | 96 | 3 | `ASYNC_SCHED=0` | async scheduling on/off |
| 10 | `dp2tp16dcp4-s96-k3` | dp2tp16dcp4 | 96 | 3 | | experimental launch form; skip if it does not start within 30 minutes |

| 11 | `base-dp8tp4-s96-k3-fusedep` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776 FUSED_EP=v2` | fused EP on the baseline layout |
| 12 | `base-dp8tp4-s96-k3-w4a8` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776 FUSED_EP=v2 W4A8=1` | 4-bit experts fed to the MXU as fp8 instead of bf16 |
| 13 | `base-dp8tp4-s96-k3-uniform` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776 HOT_SET=uniform` | simulated balanced routing (control for 14 and 15) |
| 14 | `base-dp8tp4-s96-k3-hot` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776 HOT_SET=16:0.3` | skewed routing: 30% of tokens into 16 experts on one rank |
| 15 | `base-dp8tp4-s96-k3-hotspread` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776 HOT_SET=16:0.3:spread` | same skew, hot experts spread over ranks |
| 16 | `base-dp8tp4-s144-k3` | dp8tp4 | 144 | 3 | `KV_BYTES=auto` | batch within the baseline's own capacity (18 per engine) |
| 17 | `tp32dcp8-s96-k3-fuseddraft` | tp32dcp8 | 96 | 3 | `FUSED_DRAFT=1` | draft steps 1-2 as one program (vs arm 3) |
| 18 | `base-dp8tp4-s96-k3-fastsubst` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776 FAST_SUBST=1` | host fast path for token substitution (vs arm 1) |

Arms 1 to 5 answer the layout question; 6 and 7 the batch question; 8 and 9 the two cheapest
engine knobs; 11 and 12 the expert path; 13 to 15 how much routing skew costs (dummy weights route nearly uniformly, so arms 1 to 12 cannot show it); 16 the batch lever on the baseline itself; 17 and 18 the per-step program and host costs. If arm 3 beats arm 1, run 6 and 8 on top of it; otherwise run 7 on top of arm 5.

## What to report per arm

From `SUMMARY`: `steady.tokens_per_second`, `steady.mean_request_tpot_ms`,
`steady.p95_request_tpot_ms`, `ttft_s`, `cached_tokens_hist`, `elapsed_seconds`,
`spec_decode_counter_deltas` (if non-empty). From the server logs: the `HILLCLIMB` line, the
KV-pool size line (`GPU KV cache size` or the hybrid manager's per-group block counts), any
`Fused EP MoE` line, and the compile-size warm-up lines. Report any request error or
preemption as a failed arm.

## Expected direction (roofline model, to be confirmed or refuted by the runs)

At 96 requests and 131K context every layout has the same HBM floor of about 20 ms per
step on this slice; the measured 109 ms is dominated by per-layer fixed costs (collective
latency, small kernels, host work). So:

- arm 2 vs 1: the draft removal should cut TPOT by the draft passes only (about 5 to 10 ms);
- arms 3 and 5 vs 1: within about +/-10% TPOT; the layouts differ mainly in KV capacity;
- arms 6 and 7: the first large win, if the fixed costs are as large as modeled;
- arm 8: a few percent on the EP exchange (measured +7.8% tok/s on another MoE at DP8 + EP).
