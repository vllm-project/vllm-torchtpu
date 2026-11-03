<!-- SPDX-License-Identifier: Apache-2.0 -->

# Arms

Speculative decoding is out of scope for this round: every optimization arm runs without draft tokens (`MTP_K=0`), so
draft costs do not mask the target model's step. Arm 1 keeps the 2026-10-30 configuration (3 draft tokens) only to tie
this build to the recorded number.

Run in this order; each arm is one server launch on all four hosts plus one `bench_sweep.py` run. Record the `SUMMARY`
line and the four server logs of every arm under `results/<arm>/`. Stop and report if an arm fails to start; do not
retune it.

Reference: the 2026-10-30 baseline measured 879.9 tok/s at 109.1 ms steady TPOT with 3 draft tokens.

| # | arm | LAYOUT | GLOBAL_SEQS | MTP_K | other env | isolates |
|---|---|---|---|---|---|---|
| 1 | `base-dp8tp4-s96-k3` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776` | reproduces the recorded baseline on this build |
| 2 | `base-dp8tp4-s96-k0` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776` | **the baseline for every arm below** |
| 3 | `tp32dcp8-s96-k0` | tp32dcp8 | 96 | 0 | | one wide engine: no dense-weight replication, no DP lockstep, 8-way page stripes |
| 4 | `dp4tp8dcp2-s96-k0` | dp4tp8dcp2 | 96 | 0 | | one engine per host: TP collectives inside a host, 2-way page stripes |
| 5 | `base-dp8tp4-s96-k0-fusedep` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776 FUSED_EP=v2` | fused EP MoE: dispatch, expert matmuls and combine in one program per layer |
| 6 | `base-dp8tp4-s96-k0-w4a8` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776 FUSED_EP=v2 W4A8=1` | 4-bit experts fed to the MXU as fp8 instead of being dequantized to bf16 |
| 7 | `base-dp8tp4-s96-k0-fastsubst` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776 FAST_SUBST=1` | host fast path for token substitution |
| 8 | `base-dp8tp4-s144-k0` | dp8tp4 | 144 | 0 | `KV_BYTES=auto` | batch within the baseline's own capacity (18 per engine) |
| 9 | `dp4tp8dcp2-s176-k0` | dp4tp8dcp2 | 176 | 0 | | batch at the wider layout's capacity (44 per engine) |
| 10 | `tp32dcp8-s192-k0` | tp32dcp8 | 192 | 0 | | batch at the widest layout's capacity |
| 11 | `base-dp8tp4-s96-k0-uniform` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776 HOT_SET=uniform` | simulated balanced routing (control for 12 and 13) |
| 12 | `base-dp8tp4-s96-k0-hot` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776 HOT_SET=16:0.3` | skewed routing: 30% of tokens into 16 experts on one rank |
| 13 | `base-dp8tp4-s96-k0-hotspread` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776 HOT_SET=16:0.3:spread` | same skew, hot experts spread over ranks |
| 14 | `tp32dcp8-s96-k0-fusedep` | tp32dcp8 | 96 | 0 | `FUSED_EP=v2` | fused EP on the single-engine layout |
| 15 | `tp32dcp8-s96-k0-sync` | tp32dcp8 | 96 | 0 | `ASYNC_SCHED=0` | async scheduling on/off |
| 17b | `base-dp8tp4-s96-k3-blk256` | dp8tp4 | 96 | 3 | `KV_BYTES=25769803776 EXPERT_BLOCK=256` | block 256 with the recorded 3-draft configuration (vs arm 1); measured -23..-29% on one host |
| 17 | `base-dp8tp4-s96-k0-blk256` | dp8tp4 | 96 | 0 | `KV_BYTES=25769803776 EXPERT_BLOCK=256` | expert scales after the matmul (measured 2x on the expert kernel at decode batch) |
| 16 | `dp2tp16dcp4-s96-k0` | dp2tp16dcp4 | 96 | 0 | | experimental launch form; skip if it does not start within 30 minutes |

Run arm 17 right after arm 2: it is the largest expected gain (measured -32% on one host). Skip arms 5, 6 and 14: on one host the fused EP kernel engages for 4-bit experts only with W4A8 and then makes the step 13% slower. Arms 2-4 answer the layout question at fixed batch; 5-7 the expert path and host costs; 8-10 the batch lever on each
layout; 11-13 how much routing skew costs (dummy weights route nearly uniformly, so the other arms cannot show it);
14-16 combinations and the experimental layout. Arms 4 and 9 (24 and 44 requests per engine) and 8 (18 per engine) need
the GDN tile fix on this branch (commit 7e0b6bc22): without it warm-up fails with a broadcasting error.

## What to report per arm

From `SUMMARY`: `steady.tokens_per_second`, `steady.mean_request_tpot_ms`, `steady.p95_request_tpot_ms`, `ttft_s`,
`cached_tokens_hist`, `elapsed_seconds`. From the server logs: the `HILLCLIMB` line, the KV-pool size line (`GPU KV cache
size` or the hybrid manager's per-group block counts), any `Fused EP MoE` engaged/not-engaged line, the routing-simulation
warning for arms 11-13, and the compile-size warm-up lines. Report any request error or preemption as a failed arm.
