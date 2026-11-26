# MoE layer optimization: results log

## Final summary (2026-11-27)

**MoE layer: 218.2 us -> 164.8 us (-53.4 us, -24.5%)**, 32-chip Qwen3.8 decode, mean of layer slots
0,1,3-9 from the TPU:0 device trace (method in "Metric" below). Best run: j-b3769b4e-8720-4172-a6d4.

Best build: branch `gxd/q38-moe-sc-dispatch` (worktree /mnt/ssd/vllm-torchtpu-wt/q38-sc-dispatch, HEAD
24ceba849a, not pushed), run with
`TPU_MOE_A2A_FP8_DISPATCH=1 TPU_MOE_ROUTE_PLAN_KERNEL=1 TPU_MOE_A2A_SC_DISPATCH=1` and libtpu
`--xla_mosaic_unsafe_allow_multicore_remote_dma=true` (production also used
`--xla_enable_custom_call_region_trace=true` for in-kernel regions). Every kept step is bitwise identical
to the fp8-dispatch path in tests (fp8 dispatch itself is the one numerics change, < 5% rel. err).

| step (production job) | us/layer |
|---|---:|
| baseline PR #2821 port (j-bea4600e) | 218.2 |
| round 1: HBM gather result, whole-group sizes, fp8 dispatch, 16-row bucket (j-cfe2ad36) | 186.7 |
| deferred barrier, first weights after first rows, in-kernel dest copies, gather index (j-d3401ae2) | 182.3 |
| SC side-channel dispatch + route-plan kernel + wave-mode expert kernel, DMA-queueing fixes (j-6ad82ac4) | 175.3 |
| Pallas send plan (j-33b6b4e2) | 167.4 |
| fused router top-k (j-ab4fdd5e) | 165.1 |
| owner-tail sum in spent rows (j-b3769b4e) | **164.8** |

Branch history on top of round 1 (`gxd/q38-moe-from-2821` @ 94834e118c):
daac504c20 route-plan kernel; 3c556611d2/b8fdd84d89 SC dispatch (source rows, send plan, dispatch,
drain); d4264dcf65..03e1ff71d4 expert-kernel wave mode; 4271bf8ce7 route scales; 30d768c844 wiring;
e988996493 route-plan outputs to HBM in-kernel; 54bd8e5db4 early row copies; bd07030a1b no exchange_plan
on the SC path; cd3b66a685 + b5340b0fdb routed-prefix SMEM copies; ec72b78956 first rows before first
weights; 61d06bd5b4 first gate/up part at entry; a2c9caaac8 Pallas send plan; 6f5e6b5996 fused top-k;
39ee218a22 owner-tail sum; 89ec792b1f reverted by 24ceba849a (no gain).

Where the 164.8 us goes: before the expert kernel ~34 (router + top-k ~6, routing all-gather ~11, send
plan / source rows / glue); expert kernel ~128 (start ~11.6 before the first matmul, matmul 86 = near
its structural floor, in-kernel conversion 3.8, row sends 5.6, owner tail ~10, final wait for other
members' rows ~7, mostly skew); post 1.8.

Tried and not kept: TC DMA all-to-all (interconnect-bound, 210.6), requant block 512 (no gain),
fp8 combine (+1 us), W4A8 in the kernel (+4.8), num_parts 1/4/8, select instead of conditional,
in-kernel TC dispatch (iteration 2), sequence-parallel residual / no TP replication (study: 5-9 us worse),
replicated shared expert (estimated net loss), per-token tail waits (no gain).

Remaining levers (not done): kernel start (~11.6 us before the first matmul; each reordering so far
moved the wait), pre-kernel glue on the critical path, final-wait skew, and outside the MoE window the
20 us attention TP4 all-reduce (in-kernel all-reduce fused into the out-projection).

## Summary (as of 2026-11-26)

**MoE layer: 218.2 us -> 186.7 us (-31.5 us, -14.4%)**, 32-chip Qwen3.8 decode, mean of layer slots
0,1,3-9 from the TPU:0 device trace (method below). Branch `gxd/q38-moe-from-2821`:

| commit | change | numerics | measured (job) | layer us |
|---|---|---|---|---:|
| (baseline) | PR #2821 port | | j-bea4600e | 218.2 |
| `13e7c1b4aa` | keep the row gather's result in HBM (no VMEM->HBM copy) | bitwise | j-336fa10a | 214.5 |
| `46a75793af` | whole-DP-group capacity sizes: no overflow, no `lax.cond` fallback | bitwise | j-ca44bcf0 | 209.5 |
| `78c3ee7183` | fp8 dispatch (scale in the routing all-gather, rows as uint32 words), opt-in `TPU_MOE_A2A_FP8_DISPATCH=1` | fp8 rounding of expert inputs (< 5% rel. err in tests) | j-a0ca63ee | 197.2 |
| `4088cc02e3` | 16-row bucket for the scattering expert kernel | bitwise | j-cfe2ad36 | **186.7** |

Caveat on the last row: most of its gain is a ~17 us TensorCore stall (on the next layer's
all-to-all call-start, 5 of 10 layers) disappearing. Confirmed as an instruction-overlay wait: the
previous layer's GDN `fused_conv1d_gdn_per_seq` kernel evicts top-level code, which the MoE op then
reloads without a prefetch (j-8eb36bec with overlay profiling). It depends on the compiled program's
code layout and could come back with other code changes; three libtpu overlay flags did not help.

Tried and not kept (details below): pass-through conditional -> select (+5.5 us), in-kernel token
dispatch via TC remote DMAs (iteration 2: +24 us; branch `gxd/q38-moe-kernel-dispatch`), whole-group
sizes with a VMEM receive buffer (VMEM OOM), expert-kernel weight parts 1/4/8 (2 is optimal), W4A8
inside the expert kernel (+4.8 us; branch `gxd/q38-moe-w4a8`), two libtpu SparseCore scheduling flags
(no effect).

Answers to the plan's questions:
- Hot path: confirmed "ragged-all-to-all" + capacity exchange + `gmm_v2_mlp_scatter` (all 10 layers).
- Idea 1 (overlap dispatch with the expert FFN): the TC-side version lost; a SparseCore-driven wave
  dispatch is technically viable (SC -> remote TC semaphore hand-off verified by microtest, 7/7), but
  with fp8 dispatch the all-to-all wait is now ~2-3 us per layer, so its remaining upside is the TC
  work it could move (send ~3.5 us, part of the gather ~5.8 us).
- Idea 2 (fp8 + per-row scale): done, -12.3 us; the scale travels in the routing all-gather instead of
  a padded column.
- Idea 3 (DMA permutation instead of a materialized one): the materialized copy to HBM is gone
  (-3.7 us); fusing the gather into the expert kernel costs ~14 us of in-kernel relayout (measured).

Remaining per-layer time (186.7 us): expert kernel ~122, pre-dispatch serial chain ~28 (routing
all-gather ~11, router matmul ~4, plan 2.3, send 3.5, small ops), gather 5.8, all-to-all wait ~2.3,
post 1.9.

Setup: Qwen3.8 decode, 32 chips (tpu7x 2x4x4 = 64 TensorCores), attention DP16 x TP4, MoE EP64
(12 local experts of 768, top-10), 12 requests x 4 tokens (MTP k=3) = 48 tokens per DP rank,
10 layers. Recipe `gxd-qwen38-decode-synthetic-32-chips-pr2821` (cloud-devkit
`recipes/experimental/gxd/qwen38_decode_repo_32_chips_pr2821`), cluster cloud-tpu-shared-capacity,
xprof on.

## Metric

Measured from the TPU:0 device trace of host 0, rank 0 (`rank0_...server-0-0.xplane.pb`), over every
target-forward execution (module `fused_add_rms_norm_add`, ~11 ms, 51 executions per trace) and all
10 MoE layers in each.

- MoE layer window: from the end of the attention/GDN output all-reduce before `router_topk` to the
  end of the last op before the next RMS norm's reduce (`multiply_reduce_fusion`, the next layer's
  input norm or the final norm). Robust to the MoE result's producer moving.
- Headline number: mean over layer slots 0, 1, 3-9. Slot 2 is excluded because attention's
  `rpa_metadata_schedule` conditional (~140 us, not MoE) runs inside its window. The median over all
  slots is also reported.
- Tool: `/tmp/bea4/tools/moe_win.py <pickle> 15324563906602098645` (pickle from
  `/tmp/q38-tools/sc_cache.py`). Not in the repo.

## Which code path runs (answer to the plan's question)

Confirmed from the baseline trace: every one of the 10 MoE layers runs the ragged all-to-all
capacity exchange with the fused expert kernel and the owner combine:

`router_topk` -> `dma_all_gather` (routing words, s32[16,10,48]) -> `exchange_plan` ->
`row_copy_send` -> SparseCore `ragged-all-to-all` bf16[480,16,512] -> `sort` (route plan, in the
all-to-all's shadow) -> `row_copy_gather` -> `gmm_v2_mlp_scatter-g_12-m_640-...-x-t` (expert MLP,
output rows DMA'd to their owner ranks, shared-expert partials as `extra` rows, owner tail sums and
copies to the TP ranks) -> `conditional` (selects capacity vs full-size fallback). The full-size
fallback never runs (its branch never shows up as executed in the trace).

So yes: "ragged-all-to-all" + "capacity_exchange" + "gmm_v2_mlp_scatter" is the path.

## Baseline: j-bea4600e-fb05-4440-be8c

- src: gxd/q38-moe-from-2821 @ 8da010335 + the "port from pr2821" default flips (= 5e751ea554),
  `src-8da010335-dirty-6f5da7c9.tar.gz`
- xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-bea4600e-fb05-4440-be8c/outputs/j-bea4600e-fb05-4440-be8c-server-0/j-bea4600e-fb05-4440-be8c-server-0-0-qztqq/vllm-server/xprof/host0/plugins/profile/2026_11_26_05_25_29/rank0_j-bea4600e-fb05-4440-be8c-server-0-0.xplane.pb
- **MoE layer: 218.2 us mean (slots 0,1,3-9), median 218.7 us over all slots** (final window
  definition: AR end -> the op before the next layer's input-norm reduce; with the earlier "end of
  the conditional" definition it was 217.4 / 217.5).

Per-layer time by op (mean of slots 0,1,3-8, us):

| op | us | on the critical path? |
|---|---:|---|
| `gmm_v2_mlp_scatter` (expert MLP + combine DMAs + owner tail) | 125.2 | yes |
| ragged all-to-all `call-done` wait | 26 | yes (the all-to-all is in flight ~49 us) |
| `dma_all_gather` routing | 11.1 | yes |
| `conditional` (capacity/fallback select) | 8.5 | yes (its taken branch is one 1.8 us copy) |
| `sort` (route plan) | 6.7 | no, inside the all-to-all |
| `row_copy_gather` | 4.7 | yes |
| copy-done bf16[640,8192] VMEM -> HBM (gather result) | 3.8 | yes |
| router matmul fusion | 4.5 | yes |
| `row_copy_send` | 3.5 | yes |
| `exchange_plan` | 3.4 | yes |
| `router_topk` | 2.5 | yes |
| small XLA ops, rest | ~17 | partly |

Critical path: ~28 us from the all-reduce to the all-to-all start (router, top-k, routing
all-gather, plan, send copy), ~49 us all-to-all, 8.5 us gather + copy, 125 us expert kernel, 8.5 us
conditional.

Expert-kernel floor for reference: 12 experts x (8192x4096 + 2048x8192) fp4 weights = 302 MB per
layer, ~82 us at ~3.7 TB/s per TensorCore. The kernel runs at 125 us.

## Iterations

### Iteration 1: HBM gather result + select instead of a pass-through conditional

Commits:
- `13e7c1b4aa` Keep the row gather's result in HBM (removes the 3.8 us VMEM -> HBM copy)
- `4f44e105a8` Pick the capacity exchange's fallback with a select (path `a2a_fallback_select`),
  later dropped (see verdict)

Jobs (src `src-4f44e105a-dirty-e3b0c442.tar.gz`, the same tree as 4f44e105a):
- j-9e6babd3-a481-4147-873e: both changes. **220.0 us mean (median 220.2): no gain.**
  xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-9e6babd3-a481-4147-873e/outputs/j-9e6babd3-a481-4147-873e-server-0/j-9e6babd3-a481-4147-873e-server-0-0-bm9nk/vllm-server/xprof/host0/plugins/profile/2026_11_26_06_43_23/rank0_j-9e6babd3-a481-4147-873e-server-0-0.xplane.pb
  - The copy-done of bf16[640,8192] is gone and the post-kernel time fell 9.1 -> 1.7 us.
  - But XLA scheduled the now-independent conditional (6.1 us, plus the shared expert's down
    projection) between `row_copy_send` and the all-to-all start, so the all-to-all starts ~6 us
    later and its wait grew 26 -> 31 us. The all-to-all chain is the critical path, so moving work
    off the kernel's tail into the all-to-all's shadow buys nothing while the chain is the bottleneck.
  - Lesson: the conditional itself costs ~5 us of overhead wherever it runs (only 1.7 us in the last
    layer). It should be removed rather than moved: see iteration 2b.
- j-b040cbeb-0231-46f2-bb57: died at startup (host-0 ZMQ bind `Cannot assign requested address`,
  infra); resubmitted as j-336fa10a-5594-4f3b-9f89 (`TPU_DECODE_FAST_PATHS_OFF=a2a_fallback_select`,
  HBM gather only). **214.5 us mean (-3.7 us vs baseline)**: the copy-done of bf16[640,8192] is
  gone, everything else unchanged.
  xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-336fa10a-5594-4f3b-9f89/outputs/j-336fa10a-5594-4f3b-9f89-server-0/j-336fa10a-5594-4f3b-9f89-server-0-0-6rndk/vllm-server/xprof/host0/plugins/profile/2026_11_26_06_49_55/rank0_j-336fa10a-5594-4f3b-9f89-server-0-0.xplane.pb
- Verdict: keep `13e7c1b4aa` (HBM gather, -3.7 us). The select commit cost +5.5 us on top of it and
  was dropped from the branch (rebased out; a copy is on branch `backup/iter2-before-drop-select`).
- unit tests j-4ca3c7ea-8b8f-4b0a-a836 (test_row_copy, test_decode_switches,
  test_ragged_a2a_exchange_knobs, test_ragged_a2a_moe_wiring): 430 passed.

### Iteration 2: the expert kernel does the dispatch (idea 1 + idea 3)

Commit `3ff4162c18` Dispatch the capacity exchange's tokens from the expert kernel (path
`a2a_kernel_dispatch`). After its member barrier, `gmm_v2_mlp_scatter` pushes this member's token
rows (16 KiB each, one remote DMA per row) into the receive buffers of the members that compute
them, at the rows the ragged all-to-all would have used; the receive buffer is a VMEM scratch. It
waits for its arrivals, then each tile builds its matmul rows from VMEM (relayout + the same
canonicalization as `gather_rows`). Removes `row_copy_send`, the SC ragged all-to-all and
`row_copy_gather`.

- unit tests j-cd7e92f9-bfe8-4835-9333: the new multichip test passes bitwise on 8 TPU devices for
  7/8 layouts; layout 1 failed because its capacity rows rounded up to the whole group (the exchange
  takes the full-size path, so the dispatch kernel never appears). Test layout fixed in 2b.
  Final: 326 passed, 1 failed (that layout); owner tail, gmm_v2 and exchange option matrix
  (including the kernel_dispatch options) all pass.
- benchmark j-c33c9ad7-d9b2-4109-becb (src a196c7263, still with the select path):
  **246.1 us mean: worse.** The chain before the kernel is gone (no send, all-to-all, gather), but
  the expert kernel grew 124 -> 180 us, i.e. the in-kernel dispatch costs about what the SC
  all-to-all did (~56 us), and nothing overlaps it except the first expert's weight DMA.
  xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-c33c9ad7-d9b2-4109-becb/outputs/j-c33c9ad7-d9b2-4109-becb-server-0/j-c33c9ad7-d9b2-4109-becb-server-0-0-vfk98/vllm-server/xprof/host0/plugins/profile/2026_11_26_06_50_47/rank0_j-c33c9ad7-d9b2-4109-becb-server-0-0.xplane.pb

### Iteration 2b: whole-group sizes under the kernel dispatch (no fallback conditional)

Commit `b80c5ab1df` Size the kernel-dispatched exchange for the whole DP group. With the receive
buffer in the kernel's VMEM, the DP group's every token fits (768 x 16 KiB = 12 MiB), and the route
plan's arrays can hold every route (7680), so no step can overflow and the `lax.cond` fallback is
removed.

- unit tests j-c1362f61-4e5f-41ec-8a0f (`-k dispatch`: multichip test at whole and capacity sizes,
  exchange option matrix, switch table): passed.
- benchmark j-631ae66e-db72-4dc5-80e9: **failed to compile: VMEM OOM** (65.29M of 63.94M). The
  expert kernel's weight slots keep fp4 at one byte per element in VMEM (gate 16 MiB + up 16 MiB +
  down 16 MiB = 48 MiB), so the 12 MiB whole-group receive buffer does not fit (iteration 2's
  7.5 MiB did, at ~60.8 MiB). The VMEM budget in `dispatch_supported` (16 MiB) was wrong. Whole-group
  sizes need the receive buffer in HBM (per-tile row DMAs into VMEM).

Diagnosis runs for the 56 us inside the kernel (debug-only patch `/tmp/bea4/debug_dispatch.patch`,
not committed, `TPU_DEBUG_DISPATCH` env) on iteration 2 without the select path
(src `src-3ff4162c1-dirty-7d586f97`). A first pair built on 2b was deleted (same VMEM OOM).
- j-8bd12779-660a-449c-89f8 `ref` (no debug env): iteration 2 as committed: **242.4 us** (median
  242.7), expert kernel 182.2 us.
- j-19edc822-99f8-47ef-80af `late_wait`: wait for the arrivals after the tile loop instead of before
  (numerics wrong; timing = the kernel if the dispatch fully overlapped the experts). Pending.
- j-0043d84f-5ada-4c09-ae5a `no_lhs`: skip the per-tile lhs relayout. INVALID: the stale VMEM lhs
  made garbage expert outputs, routing collapsed in later layers, the capacity overflowed and the
  full-size fallback ran (1.5 ms per layer). `late_wait` (j-19edc822) had the same flaw and was
  deleted. Redone with a zero lhs (sane routing):
- j-d21ea547-7a36-424b-951c `zero_lhs`: arrivals waited up front, lhs = 0, no relayout: **228.7 us,
  kernel 167.9 us** -> the per-tile lhs relayout costs ~14 us per layer inside the kernel; the
  remaining ~44 us over the baseline kernel (124 us) is the pushes + the arrival wait.
- j-16014b36-c672-46d1-aa4a `late_zero`: lhs = 0, arrivals waited after the tile loop: **199.7 us,
  kernel 138.4 us** -> the exposed arrival wait is ~29.5 us.
- Breakdown of iteration 2's +58 us in the kernel: arrival wait ~29.5, lhs relayout ~14, row pushes
  (~115 single-row remote DMAs issued from scalar code) and other overhead ~14.
- Read: even with the arrivals fully overlapped (per-expert waits + expert-ordered sends), the
  relayout and push costs keep the layer at ~214 us, no better than the HBM-gather baseline
  (214.5). The in-kernel dispatch only wins if the relayout (~1.1 us per tile) and the push issue
  also shrink; parked for now.

### Iteration 3: fp8 dispatch on the all-to-all path (idea 2)

Branch `gxd/q38-moe-fp8-dispatch` (worktree `/mnt/ssd/vllm-torchtpu-wt/q38-moe-fp8`), commit
`fcd0d10229` "Send the capacity exchange's token rows as fp8", on top of `13e7c1b4aa` (the best so far,
214.5 us). Opt-in with `TPU_MOE_A2A_FP8_DISPATCH=1` (changes numerics). Token rows go as fp8 e4m3
(8 KiB instead of 16 KiB) with one f32 scale per token; instead of a padded scale column (+50% at
tile granularity), the scale rides along as an extra word per token in the routing all-gather
(`s32[16,11,48]` instead of `[16,10,48]`), so every member knows every token's scale. The send
buffer is written with per-row DMAs (`copy_rows`, no relayout for fp8), and `gather_rows`
dequantizes to bf16 while it relayouts.

- unit tests j-c6c73c72-389f-4782-bbea: 322 passed incl. all 24 new fp8 kernel tests; the 8
  multichip cases failed only because the test used H = 2048 (fp8 whole-tile rows need H % 4096 ==
  0, so the path correctly stayed bf16). Fixed to H = 4096: j-6171a339-078c-457a-a5d2, 8 passed
  (relative error vs the bf16 exchange < 5%, owner/streamed combine, TP 1/2/4, overflowing step).
- benchmark j-e1c00402-4413-4e47-b0f6: died at startup (host-0 ZMQ bind, infra). Resubmitted as
  j-4a554085-df5c-4e91-addc: **249.6 us: worse.** Two fp8-dtype problems, not bandwidth:
  `row_copy_rows` took 30 us (XLA tiled the fp8 send buffer T(8,128)(4,1) vs its source's
  T(32,128)(4,1), so every row DMA re-tiled), and the fp8 ragged all-to-all was NOT offloaded to
  the SparseCore: a synchronous TensorCore op of 31 us.
  xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-4a554085-df5c-4e91-addc/outputs/j-4a554085-df5c-4e91-addc-server-0/j-4a554085-df5c-4e91-addc-server-0-0-4c4lz/vllm-server/xprof/host0/plugins/profile/2026_11_26_07_34_34/rank0_j-4a554085-df5c-4e91-addc-server-0-0.xplane.pb
- Fix (amended commit `accad19a8f`): the fp8 rows travel as uint32 words [8, H/32] (native 32-bit
  tiles), packed in the order Mosaic's packed bitcast reads back, so the row DMAs are tile-aligned
  and the all-to-all moves 32-bit data; the gather bitcasts back to fp8 for free.
  tests j-d4413065-9201-45ef-bcc9: 253 passed (fp8 kernels with the word packing + the multichip
  tolerance test). Benchmark j-64cfc7e4-5f4d-4a4e-85d8: **230.8 us** (on 13e7c1b4aa, still worse),
  but the all-to-all is now SC-offloaded and takes **~27 us in flight vs ~47 us for bf16** (wait
  ~5 us vs ~27). Two overheads ate the gain: `row_copy_rows` (one 8 KiB HBM->HBM DMA per row,
  ~270 ns each) took 31 us, and the XLA packing transposes ~7 us.
  xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-64cfc7e4-5f4d-4a4e-85d8/outputs/j-64cfc7e4-5f4d-4a4e-85d8-server-0/j-64cfc7e4-5f4d-4a4e-85d8-server-0-0-54p7v/vllm-server/xprof/host0/plugins/profile/2026_11_26_07_54_57/rank0_j-64cfc7e4-5f4d-4a4e-85d8-server-0-0.xplane.pb
- Fix 2 (amended + rebased onto iteration 4: commit `78c3ee7183` on `gxd/q38-moe-fp8-dispatch`, parent
  `46a75793af`): a fused send kernel `copy_send_rows_fp8` (like `copy_send_rows`: quantize the bf16
  token rows in VMEM with the per-token inverse scale, pack with `pltpu.bitcast` into the words,
  copy rows in VMEM, chunked DMAs out). XLA only computes the per-token scale (CSE'd with the shared
  expert's own fp8 row scale). `copy_rows` removed.
  tests j-ccb8b774-1873-4ceb-84b5 (fp8 send kernel bitwise vs the XLA packing, gather, multichip
  tolerance at capacity and whole-group sizes): **306 passed** (the in-kernel quantize + pack is
  bitwise the XLA reference packing). Benchmark j-a0ca63ee-f61d-4092-b3f7
  (`TPU_MOE_A2A_FP8_DISPATCH=1`, cumulative with iteration 4): **197.2 us mean (median 201.0):
  -12.3 us vs iteration 4, -21.0 us (-9.6%) vs the 218.2 us baseline. Best so far.**
  Bimodal by layer slot: slots 0/3/7 ~187.5 us (all-to-all wait ~3 us), slots 1/4/5/6/8 ~201 us
  (wait ~17 us); slot 9 pays the 19 us exchange_plan seen in iteration 4. "other" small ops +5 us
  (scale computation and the receive-row scale scatter).
  xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-a0ca63ee-f61d-4092-b3f7/outputs/j-a0ca63ee-f61d-4092-b3f7-server-0/j-a0ca63ee-f61d-4092-b3f7-server-0-0-5qq9x/vllm-server/xprof/host0/plugins/profile/2026_11_26_08_11_53/rank0_j-a0ca63ee-f61d-4092-b3f7-server-0-0.xplane.pb

### Branch layout after the diagnosis

- `gxd/q38-moe-from-2821` (this worktree): reset to `13e7c1b4aa` (HBM gather, best so far) and
  continued from there.
- `gxd/q38-moe-kernel-dispatch`: iterations 2 + 2b (in-kernel dispatch, parked).
- `gxd/q38-moe-fp8-dispatch`: iteration 3 (fp8 dispatch on the all-to-all path; fast-forwarded into
  `gxd/q38-moe-from-2821`).
- `gxd/q38-moe-w4a8`: W4A8 inside the expert kernel (parked, slower).
- `gxd/q38-sc-tc-semaphore-test`: the SC -> TC semaphore microtest (`3798b7bba1`, 7/7 pass).

### Iteration 4: whole-group sizes on the all-to-all path (no fallback conditional)

Commit `46a75793af` "Size the capacity exchange for the whole DP group" (path `a2a_whole_group`). The
2b idea without the in-kernel dispatch: with the plan kernel and the row copies, nothing in the
capacity exchange costs time in proportion to its sizes, so it can be sized for the DP group's every
token (768 rows) and every route (7680), where no step can overflow, and the `lax.cond` fallback is
removed (8.5 us per layer in the baseline). The A2A receive buffer is in HBM, so there is no VMEM
problem; costs: the gather's HBM result grows to 7680 x 8192 bf16 (126 MiB, written only up to the
step's routes), 12 MiB of VMEM in the standalone gather kernel, and larger SMEM index arrays.

- unit tests j-8aa82205-837a-444c-b038 (exchange option matrix incl. whole-group options, the
  no-conditional lowering test, switch table, owner tail, row copies): **448 passed**, 0 failed.
- benchmark j-ca44bcf0-5f8b-493c-a616 (src `src-46a75793a-dirty-e3b0c442`, the tree of 46a75793af):
  **209.5 us mean (median 208.0), -5.0 us vs 214.5, -8.7 us vs the 218.2 baseline.** Slots 0-8 are
  207.3-207.6 us; the conditional is gone (post-kernel 9.1 -> 1.9 us). The last layer pays ~17 us
  extra in its `exchange_plan` (19 us instead of 2), which overlaps a long SparseCore op
  (`ragged-all-to-all.29`, ~420 us span, apparently the EP-sharded embedding/LM-head exchange) that
  XLA now schedules across the last layer (the conditional used to be a scheduling barrier there).
  xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-ca44bcf0-5f8b-493c-a616/outputs/j-ca44bcf0-5f8b-493c-a616-server-0/j-ca44bcf0-5f8b-493c-a616-server-0-0-xndv7/vllm-server/xprof/host0/plugins/profile/2026_11_26_07_46_09/rank0_j-ca44bcf0-5f8b-493c-a616-server-0-0.xplane.pb

### Iteration 5 (experiment): expert-kernel weight parts

The expert kernel is 124 us against a ~90 us HBM floor for its 302 MB of fp4 weights. It loads each
expert's gate/up weights in `num_parts` = 2 column parts, each refilled for the next expert right
after its last use. Debug-only env override (not committed, src `src-46a75793a-dirty-a58ee42b` =
iteration 4 + `TPU_DEBUG_MLP_PARTS`), VMEM unchanged:
- j-be7051b7-d542-4e53-8dd5: `num_parts` = 4: **213.8 us, kernel 128.0 us (+4.3)**: worse. Finer
  parts add per-part overhead without better overlap: the kernel is not weight-DMA-bound.
- j-55dfe3f2-6661-4aee-b24c: `num_parts` = 8: **224.1 us, kernel 138.9 us (+15)**: worse still.
- j-26d1c988-44a0-4170-9c5a: `num_parts` = 1: **242.8 us, kernel 157.0 us (+33)**: with one part
  the next expert's gate/up refill cannot overlap, so weight DMA is exposed.
- Verdict: the default 2 is the sweet spot (1: 157.0, 2: 123.7, 4: 128.0, 8: 138.9 us). The kernel
  sits at a balance of weight-DMA overlap and per-part overhead; this knob is exhausted. Not
  committed.

### Subagent audit of the critical path (iteration 4 trace j-ca44bcf0, layer slot 5)

Script and per-op dump: /tmp/agent-glue/win.py, /tmp/agent-glue/l5.txt (not persistent).
- Pre-dispatch path is fully serial: 27.6 us busy of 27.7 us wall (norm 1.1, router matmul 4.1,
  softmax/topk/renorm/pack 3.7, dma_all_gather 12.0, copy-done of the gathered words HBM->VMEM 0.9,
  exchange_plan 2.3, row_copy_send 3.3).
- **Found: the NEXT layer's ragged-all-to-all `call-start` (operand-less SparseCore async-start) is
  hoisted right after this layer's call-update and blocks the TensorCore ~16.5 us in 5 of 9 layers**
  (call-done is short in exactly those layers). Seen in iteration-4+ runs (ca44bcf0, be7051b7,
  a0ca63ee), never in older ones: the fallback conditional used to be a scheduling barrier. This is
  the likely cause of the fp8 run's bimodal layers (~187.5 vs ~201 us).
- Shadow work during the all-to-all (~19.9 us busy): route_plan 8.3 (its 7680-key sort 6.7),
  `_owner_routes` tail 4.4 (12 small ops, three 0.67 us HBM copies of [12,10] arrays), owner extras
  2.3, shared expert 3.5, next layer's GDN parameter casts 1.3.
- Proposals (agent's ranking): (1) fuse dma_all_gather + exchange_plan + the local send (3-5 us,
  critical); (2) fold row_copy_gather into the expert kernel (iteration 2 measured that in-kernel
  relayout costs ~14 us, so likely a loss); (3) move softmax/renormalize/packing into router_topk
  (~1 us); (4) trim shadow work (replace the sort by a Pallas compaction, pack the owner-tail
  operands into one array, store GDN params as f32, emit the shared down-projection as [48,16,512]).

### Iteration 6 (experiment): stop the hoisted next-layer all-to-all call-start

On the current best build (78c3ee7183 + `TPU_MOE_A2A_FP8_DISPATCH=1`, src `src-78c3ee718.tar.gz`),
libtpu scheduler flags via `EXTRA_LIBTPU_INIT_ARGS` (names found in a libtpu 0.0.50 binary; the image
has 0.0.47, so a missing flag would abort startup):
- j-d192eb04-42c8-47a2-bba9: `--xla_tpu_sparse_core_ici_overlap_limit=1`: **197.5 us, no effect**
  (identical stall pattern).
- j-4677aaa0-83b8-4fc0-bcaa: `--xla_tpu_enable_concurrent_sparse_core_offloading=false`: **196.1 us,
  no real effect** (same stall pattern; -1 us is within run-to-run noise).
- Verdict: neither flag stops the hoisted call-start stall; waiting on the overlay diagnosis run
  j-8eb36bec-e128-4b83-92db.

### Subagent study of the expert kernel (static analysis, unverified without LLO)

- Estimated split of the 124 us: matmul bodies ~95-100 (with bf16 activations Mosaic likely widens
  fp4 to bf16: ~32 MXU weight pushes + ~64 widen ops per 256x256 tile vs 16 for fp8), 3 `lax.switch`
  regions per tile ~6.5 (~180 ns each, consistent with the num_parts results), down-projection
  epilogue 5-13 (mask + cast + [bm,16,512] row/sublane reshape over all 32 bucket rows), row sends +
  scalar 2-5, entry barrier 2-5, exit (slowest sender + owner tail) 3-6.
- Not viable: accumulating k-blocks in the MXU or one scale per column block (scales are per
  (k-block, column)); bf16 scale multiply; hoisting the widen.
- Proposed experiments: E1 W4A8 inside the kernel (fp8 activations with a per-row scale applied
  once per column block; numerics change; est. -20..-25 us, capped by the weight-DMA floor);
  E2 down epilogue (exact; est. -5..-10 us); E3 a 16-row bucket (exact; est. -3..-6 us).

Measurements (debug patch on 78c3ee7183, src `src-78c3ee718-dirty-955ea5ad`, fp8 dispatch on):
- j-cfe2ad36-04a9-4faf-8512 E3: `TPU_DEBUG_MLP_BUCKETS=16,32` (buckets 16/32/64): **186.7 us mean
  (median 186.8), -10.5 us vs 197.2. Best so far.** The kernel gains only ~0.6 us (122.1 vs 122.7);
  the ~17 us call-start stall is gone in every layer, and so is the last layer's slow exchange_plan
  (2.8 vs 19.7 us). All ten slots 186-187 us. This supports the instruction-overlay hypothesis: the
  extra 16-row body copy changed the program's code layout. The gain is real but fragile (another
  code change may bring the stall back).
  xprof: go/cdk-xprof?gs://cloud-devkit/jobs/j-cfe2ad36-04a9-4faf-8512 (host0 rank0 xplane via `cdk job xprof -a`)
  Committed as `4088cc02e3` "Give the scattering expert kernel a 16-row bucket" (scoped to the
  scattering kernel's call site; other kernels keep the default buckets). Bitwise test + owner tail +
  streamed combine tests: j-7be0fe65-31ec-4459-aa4b: **164 passed** (incl. 16 bitwise bucket cases).
- j-5828a74d-6dca-4911-981d E2 probe: `TPU_DEBUG_MLP_STAGE=zero` (stage writes zeros; timing only,
  outputs wrong): **inconclusive.** Zero expert outputs changed the downstream routing: rank skew
  shows up as a 31-39 us routing all-gather (vs ~11) and the kernel time varies 116-142 us by slot.
  The cleanest slots hint at ~5-6 us for the epilogue (116-117 vs 122 us); not acted on.

### Subagent: the bimodal fp8 layers (trace j-a0ca63ee)

Scripts: /tmp/agent-bimodal/ (not persistent). The SparseCore all-to-all takes 26.7-28.4 us in every
slot and starts 0.1 us after call-update. In slots 1/4/5/6/8 the NEXT layer's `call-start` sits right
after this layer's call-update and stalls the TensorCore ~17 us (pattern `fSffSSSfSf` in all 55
forwards, both TCs). Nothing else runs on the SparseCore. The same 17 us also lands on the purely
local `exchange_plan` kernel in other builds/slots, and vanished with an unrelated kernel change
(num_parts 8), so the agent's hypothesis is a layout-dependent TC instruction-overlay miss, not a
SparseCore handshake (unconfirmed; the trace's overlay line is empty). Potential ~8.5 us/layer.
Next: overlay diagnostics (`--xla_tpu_show_overlay_waits_in_profiler=true`), and the two hoist flags
above (they exist in libtpu 0.0.47: /tmp/flags_libtpu047.txt).

### Iteration 7: W4A8 inside the expert kernel (proposal E1)

Branch `gxd/q38-moe-w4a8`, commit `e071baf222` (on 78c3ee7183), written by a subagent and reviewed:
`TPU_MOE_MLP_W4A8=1` (default off; numerics change) quantizes the expert kernel's rows (token rows and
the intermediate) to fp8 per row and widens fp4 weight blocks to fp8 (exact), scales per (k-block,
column) as before, row scale once per column block.
- tests j-bd8f4c9e-464e-4f60-a988 (multichip, relative error < 5% vs bf16): **5 passed** (the fp8 x fp8 matmul lowers and compiles in Mosaic).
- benchmark j-0ea20230-d4dd-4d37-933a (fp8 dispatch + W4A8, on 78c3ee7183 without the 16-row
  bucket): **expert kernel 127.5 us vs 122.7: +4.8 us, slower.** The layer is 191.8 us only because the
  ~17 us stall also vanished in this build (more evidence that the stall depends on code layout).
  The static-analysis premise (bf16 activations double the per-tile work vs fp8) does not hold: the
  per-row quantization and fp8 converts cost more than they save. Parked on `gxd/q38-moe-w4a8`
  (numerics change, no gain); not merged.

### Research: TC <-> SC semaphore hand-off for a wave-based dispatch (subagent, static)

Verdict: partially feasible (nothing run yet). One kernel cannot span TC and SC (`mpmd_map` rejects
TensorCore meshes: jax/_src/pallas/mosaic/core.py:450, pallas_call_registration.py:674). But an SC
kernel can output a semaphore in TC memory (`pltpu.SemaphoreType.DMA(()) @ tc_mesh`), signal it with
remote DMAs and exit; a later TC kernel takes (recv, sem) and waits on it (upstream
`tests/pallas/tpu_pallas_mpmd_test.py::test_async_sc_tc_prefetch_vmem`; needs
`xla_mosaic_unsafe_allow_multicore_remote_dma=True`). All present in jax 0.10.2 (the serving pin).
XLA schedules the TC kernel after the SC kernel's done (data edge), so the overlap is "SC issues all
waves and exits, TC waits per wave while DMAs land" (delivery overlapped with expert compute; ~115
row-DMA issues move from the TC scalar unit to 32 SC subcores). Risks: SC -> remote TC semaphore has no
upstream test ("unsafe" flag), SC indirect DMAs cannot target remote devices (contiguous chunks only),
possible global flag side effects (MSA), row-granular writes into tiled HBM.
First step: a 2-chip microtest of SC -> remote TC semaphore with an HBM receive buffer.
- **Microtest j-44fdc96d-67ec-4603-9f4f: 7/7 passed on TPU** (file
  `tests/kernels/test_sc_tc_remote_semaphore.py`, uncommitted, on scratch branch `gxd/q38-moe-w4a8`):
  SC -> local TC semaphore with an HBM buffer (with and without the MSA/VMEM flags, and even without
  the "unsafe" multicore flag); **SC -> remote TC semaphore, ring shift over 2 devices (same chip) and
  8 devices (across chips)**, correct data; **two waves** (one semaphore per wave, or one waited twice):
  right after the TC's wave-0 wait, wave 1 had already landed on half the devices (2048 bytes) and not
  on the others, so the per-wave wait really gates. The SparseCore-driven wave dispatch is viable on
  this stack (jax 0.10.2, libtpu 0.0.47).
- Overlay diagnosis run j-8eb36bec-e128-4b83-92db (78c3ee7183 + fp8, `--xla_tpu_show_overlay_waits_in_profiler=true
  --xla_tpu_show_overlay_waits_in_xprof=true --xla_tpu_show_overlay_details_in_profiler=true`): 198.5 us, same stall
  pattern (slots 1/4/5/6/8 call-start ~17 us, slot 9 exchange_plan 19.7 us); the trace now has "TC Overlay",
  "Scalar Unit" and "Pallas Primitives" lines (empty; the overlay events are on "XLA TraceMe").
  **Confirmed overlay waits** (subagent, /tmp/agent-overlay/): each slow call-start (slots 1/4/5/6/8)
  and the slow exchange_plan (slot 9) contains one `overlay-wait` of ~16.7 us right after an
  `overlayer-cut-and-add` issued at that op (no prefetch); fast slots find the code resident. The
  evictor is the previous layer's `fused_conv1d_gdn_per_seq` (GDN) kernel, which itself waits ~7.5 us
  for its own code. Overlay waits total ~169 us per forward (incl. 7 x 7.6 us in gdn_per_seq). The
  overlay id/size is not in the trace.
- Overlay-flag A/B on the stall-prone build (78c3ee7183 + fp8; if a flag removes the stall here, the
  gain no longer depends on layout luck):
  - j-75a8b987-9c88-4a7d-bab1 `--xla_tpu_overlay_enable_hlo_function_packs=true`: **198.2 us, no effect** (same stall pattern; likely already the default or irrelevant here).
  - j-8e996a6c-ad0f-4000-96db `--xla_tpu_merge_small_overlays_into_big_neighbors=true`: died at startup (TCPStore: 4/8 hosts joined, infra); resubmitted as j-2be8fdf3-0582-43d9-8d0b: **198.4 us, no effect**.
- Verdict: none of the three overlay flags removes the stall on the stall-prone build. The stall-free
  layout of `4088cc02e3` is the fix in hand; a robust fix needs compiler-side overlay placement or a
  smaller `fused_conv1d_gdn_per_seq` kernel (the evictor; outside the MoE layer). The overlay waits
  outside the MoE (~53 us per forward in gdn_per_seq itself) are a separate opportunity.
  - j-cac7c097-eefc-4f95-bb5c `--xla_tpu_last_overlay_prefetches_first=true`: **197.3 us, no effect** (same stall pattern).

## Round 2 (goal: 130-140 us/layer)

### Expert-kernel microbenchmark (single host, 4 chips / 8 TCs)

Bench: `tests/kernels/bench_moe_mlp_decode.py` (untracked; run via the pytest recipe with the tree mapped,
helpers /tmp/bench/{stage,submit,wait}.sh, not persistent). Shape: 12 experts, H 8192, I 2048, fp4
weights with 256-row block scales, 120 routed rows (multinomial over the experts), buckets 16/32,
num_parts 2. Time per call = difference of two jitted fori_loop lengths.

| variant | us/call | job |
|---|---:|---|
| `gmm_v2_mlp_scatter`, 8 members, remote row sends | 138.3 | j-84b85523 |
| same, 1 member (local copies only) | 108.9 | j-84b85523 |
| `gmm_v2_mlp` (no scatter) | 110.5 | j-84b85523 |
| weight DMAs only, the kernel's slot/refill pattern | 94.4 / 94.8 | j-84b85523 / j-d8c081ce |
| 1 member, weights loaded for the first expert only (compute only) | **108.6** | j-d8c081ce |
| 8 members, compute only | 136.6 | j-d8c081ce |

Findings: **the kernel is compute-bound** (108.6 us of compute against 94.8 us of weight DMA, which is
fully hidden), and the remote row sends add ~28 us of serial TensorCore time on top in the 8-member
bench (production: 122 us vs ~109 us local-only). Two targets: (1) cheaper per-tile matmul work
(-14 us to the DMA floor); (2) the scatter epilogue.

### Production region trace of the expert kernel (j-7f3d4ed1, best build + `--xla_enable_custom_call_region_trace=true`)

187.5 us/layer (unchanged by the flag). Named-scope regions land on the "XLA TraceMe" line (subagent,
scripts /tmp/agent-regions/). Per kernel (122.9 us, 458 instances, 12 tiles = 12 experts):
compute (up0 + up1 + down0, ~2.4 us each per tile) 86.3; weight waits 0.16; stage waits 0.4;
`send_rows` 3.9; gaps 1.7; **entry (barrier over 64 members, extra-row sends, prologue) 13.1**;
**end: final wait for incoming rows 6.7 + owner tail 10.5** (tail_sum 5.5, tail_peers 3.7, tail_push
1.3). About 36 us of the kernel is collective overhead at its two ends.

More bench probes:
- weight DMAs only on 1 / 2 (one chip) / 8 cores at once: 94.8 / 100.2 / 107.0 us (j-61105d9c):
  the cores of a chip share HBM; the DMA floor with every core busy is ~107 us.
- 8 members, compute only (no weight DMAs): 133.7 (local rows) / 136.0 (remote) vs 109 for one member:
  the multi-member penalty is not HBM and not the remote sends.
- W4A8 kernel (`e071baf222`): 112.4 us vs 109.2 bf16 (compute only 111.7): slower in isolation too.

### Iteration 8: barrier wait deferred to the first send (`e71b151eae`)

Signal the barrier's first round at entry, wait for it right before the first row leaves (after
tile 0's compute); the second round is signalled then and waited for at exit. Bench (8 members):
138.3 -> 134.8 us (remote), 132.6 (local only), 1 member unchanged (j-3f736b96).
Production: j-da7eaa93-8a4d-4ed3-b96c (pending); with the fused routing kernel on top
(`3cc929141e` cherry-picked, = 63e7b05890): j-3e126242-d0b0-414a-bd69 (pending).

### Fused router top-k (subagent, branch `gxd/q38-moe-fused-routing`, `3cc929141e`)

`select_softmax_renormalized`: top-k on the router logits, softmax over only the k selected (same
selection; weights within 3.3e-7 rel. in f32, 0 bf16 ulps on TPU). Gated on rowmax top-k, softmax
scoring, renormalize, no score bias. Micro-timing at [48,768] top-10: ~0.7 us vs ~3.4 us for the old
softmax + top-k + renormalize sequence. Tests: j-5c474f09 110 passed.
- Production j-da7eaa93-8a4d-4ed3-b96c (`e71b151eae`): **187.2 us** (kernel 122.0): no gain, as expected
  once the kernel is understood as weight-stream bound (~105 us of weights from kernel start; the 13 us
  entry was the weights' head start). Tests j-f58a94e1: 418 passed. Kept (harmless, cleaner), not a win.
- Production j-3e126242-d0b0-414a-bd69 (+ fused router top-k): **187.3 us**: top-k 2.5 -> 1.0 us and
  glue -0.5, absorbed by the routing all-gather wait (+1.1) and kernel noise (+2.4). Run-to-run /
  layout noise is +-1-2 us per layer, so ~2 us pre-dispatch gains are not resolvable alone.
  (moe_win.py now also matches `router_softmax_topk`.)

### Where the floor is (as of iteration 8)

The expert kernel streams 302 MB of fp4 weights per core per layer; with every core busy that takes
~100-107 us. The layer is 187 us = ~62 us before the kernel (router, routing all-gather, plan, send,
all-to-all with the route plan in its shadow, gather) + ~105 us weight stream + ~17 us kernel end
(final wait for rows + owner tail). Reaching 130-140 needs the weight stream to overlap the
pre-kernel phase: the all-to-all and gather must move inside the expert kernel (weights stream while
rows are in flight), and the route plan (now hidden under the all-to-all) must shrink.
- TC DMA ragged all-to-all (subagent, `b00bc36561` on `gxd/q38-moe-dma-a2a`, `TPU_MOE_A2A_DMA=1`,
  20/20 kernel tests j-96a0271e, 48/48 e2e j-4bb05178, bitwise vs XLA): production j-b7901a41 (pending).
- Pallas route-plan kernel (subagent, branch `gxd/q38-moe-route-plan-kernel`): in progress.
- DMA all-to-all production j-b7901a41-f16b-4e83-bef6 (`TPU_MOE_A2A_DMA=1`): **210.6 us, worse**: the TC
  DMA all-to-all takes 25.7 us median (min 23.0), as long as the SparseCore one, and blocks the TC (its
  shadow work becomes serial). ~1 MB of fp8 rows per core over 16 peers is interconnect-bound
  (~8 us barrier + ~15-17 us transfer). Flag stays off; branch parked.

### Iteration 9: the kernel's 11 us prologue was the first row block waiting behind its weights (`7556569f6e`)

Subagent (Tensor Core trace marks + `ep_*` TraceMe regions, scripts /tmp/agent-entry/): of the ~11.4 us
before tile 0, ~8.3 us is emit_pipeline's `ep_wait_in` for the first lhs block, whose copy queued
behind the first expert's ~25 MB of weight copies (started earlier); ~2.4 us is a marker-less gap at
entry (likely the scalar prefetch of dest_member/dest_row, 2 x 7680 words); overlay waits ruled out.
Before e71b151eae the barrier hid the same 8 us. Fix: start the first tile's weight copies in its body.
- Production j-4c5f65a2-eb12-4753-83db: **186.7 us** (kernel 122.0 -> 120.4): prologue 11.4 -> 3.8 us,
  but the first tile now waits 3.9 us for its first 8.4 MB weight part.
- Correction to the earlier reading: after tile 0 no tile waits for weights at a 7.5 us tile period,
  so production sustains >= 3.35 TB/s per core; the kernel is compute-bound (86 us of matmul).

### Iteration 10: row destinations copied in-kernel behind tile 0 (`f817f3f780`)

dest_member/dest_row (2 x 7680 int32) were scalar-prefetched (~2.4 us before the body starts); now
passed in HBM and copied to SMEM at entry, waited for before the first row leaves. Tests j-60c2e4ba
(341 passed); 7556569f6e's tests j-b89506c9 (259 passed).
- Production j-a0306c7e-dbf5-4da4-9e67: **185.7 us (best)**, kernel 119.2 (entry 2.8, first weight
  wait 3.9, compute 85.3, final wait 8.4, tail 10.4).

### LLO analysis of the expert kernel's 16-row bodies (dump j-bf40b694, subagent, /tmp/agent-llo/)

14.67 bundles per 256x256 weight tile (gate/up), 14.25 (down); implied TC clock ~1.6 GHz. Mosaic
already widens fp4 -> fp8 e4m3 (vunpack.c.l/h.s4 + vcvt, 32 VALU ops per tile = 79% of VALU) and the
MXU runs fp8 weights x bf16 rows (8 subr/push pairs per tile, k-blocks alternate mxu0/mxu1). Floors:
VALU 10.1, MXU 9.0 bundles/tile; the bound is the per-MXU weight-push rate (pairs every 2-4 bundles,
4 when the VALU slots are full). No spills at bm16; interleave_lane costs nothing. Options: coarser
scale blocks (512/1024 rows, accumulate in the MXU: -8%/-15%, needs re-quantized checkpoints);
pre-converted fp8 weights (doubles DMA, loses); activation-stationary (worse). Code: 39.8k bundles,
bm16 bodies 11.5k, bm32 12.5k, bm64 15.3k. => the matmul body is near its structural floor.

### Iteration 11: row gather's index copied in-kernel (`94834e118c`)

The gather scalar-prefetched its 7680-word index; now copied to SMEM alongside the source rows. Named
scopes added. Tests j-dcdc12ef: 298 passed. Production j-d3401ae2-29bc-4559-851a: **182.3 us (best)**, gather 5.8 -> 5.4 us (entry
1.2, copy-in 1.3, rows 1.8, store waits ~1.0), kernel 117.8 (partly noise).

### Next (queued with the user): SparseCore multi-wave dispatch overlapped with expert compute

Design: an SC Pallas kernel issues the fp8 row DMAs expert by expert (wave e = rows for the receiver's
local expert e), one row per route, landing directly in the expert kernel's row order (offsets from
the all-gathered routing), signalling one DMA semaphore per wave in the receiver's TC memory; the TC
expert kernel waits per wave, so the ~25 us transfer hides under ~86 us of compute and the gather
kernel disappears (in-kernel fp8 -> bf16 conversion, ~2 us). Estimated net -15..-20 us (-> ~162-167).
Prerequisites: fast route plan (subagent), SC kernel async w.r.t. TC work, SC-side barrier, per-wave
waits in the expert kernel. Step 1: feasibility microtests (subagent, branch
`gxd/q38-sc-wave-dispatch-test`).

#### SC wave-dispatch feasibility (subagent + jobs j-46fde198, j-4fb8c157, j-f4f5c128)

Branch `gxd/q38-sc-wave-dispatch-test`, commit `f0758b098f` (test_sc_wave_dispatch.py: 4 passed).
- Works: SC scalar-subcore `pl.kernel` issuing remote row DMAs into each receiver's HBM buffer in
  wave order, one DMA semaphore per wave in TC memory, TC kernel waits per wave; barrier semaphore +
  collective_id on the SC; buffer/semaphores passed as SC-kernel outputs into the TC kernel (no copy);
  32 back-to-back skewed calls correct (needs a sentinel row per empty pair to keep the barrier exact).
- **Async overlap confirmed**: the SC Pallas kernel lowers to async-start/update/done on the
  "sparsecore" thread; XLA schedules TC work in between: SC 25 us + matmul 55.6 us -> 58.2 us.
- DMA issue rate is the limit: ~126-130 ns per DMA (~8 DMAs/us) from 1 scalar subcore and also from
  16 vector subcores (so it is a DMA-engine/descriptor rate, not the issuer); 32 vector subcores have a
  ~193 us launch cost (unusable). 120 rows/sender: +11.6 us (1 scalar subcore), +8.0 (2) over an
  ~8 us empty-call cost. Dynamic pl.delay unsupported on SC; semaphore refs cannot be loop carries.
- Implication: one DMA per row (~120-200/sender) costs 15-25 us of SC time, hidden only by overlap;
  fewer, larger DMAs (per destination x coarse wave) are the way to cut it.

### Route-plan kernel (subagent, `6cad38aef9` on `gxd/q38-moe-route-plan-kernel`, `TPU_MOE_ROUTE_PLAN_KERNEL=1`)

One Pallas kernel computes route_plan (token_sorted, group_sizes, dest_member/dest_row), the fp8
row_scale scatter and the owner-combine arrays: **3.6 us vs 18.9 us** of XLA ops at the decode shape
(bench, one core), bitwise identical (tests j-205a2ae9: 429 passed). Today it runs in the all-to-all's
shadow; it is a prerequisite for the SC side-channel dispatch. Production on top of 94834e118c
(cherry-pick daac504c20): j-3e0e0d79-d47a-4999-9821: **183.7 us** (noise vs 182.3); the all-to-all wait grows 2.6 -> 12.6 us: the route-plan shadow work is gone and the SC all-to-all (~27 us) is the bare critical path, ~12 us of TC idle inside it.
- Issue-and-exit (subagent, `bd1bfcc747`, tests j-2a948ab8: 8 passed): the dispatch kernel exits right
  after issuing; its send semaphore must stay on the SC (`DMA((1,)) @ sc_mesh`; a TC-memory send
  semaphore is rejected: "Source semaphore and source ref core type mismatched"), waited by a second
  SC drain kernel that also keeps the source rows alive. Evidence (j-4eae8a4e): at consumer entry only
  1/12 of wave 0 has landed; with 64 KiB rows 40 us of consumer compute is fully hidden (total ~
  max(transfer, issue + compute)). DMA issue ~20-35 ns per extra DMA (j-a51597cd); fixed cost
  (launch + barrier + consumer + drain) ~6.5 us/call.

### Building the SC side-channel dispatch (design: sc-dispatch-design.md, untracked)

Base branch `gxd/q38-moe-sc-dispatch-base` (= 94834e118c + route-plan kernel, daac504c20). Two
parallel subagents: SC dispatch kernel + sender plan + drain (`gxd/q38-sc-dispatch-kernel`), and the
expert-kernel wave consumer with in-kernel fp8 -> bf16 conversion (`gxd/q38-sc-dispatch-expert`).
- Dispatch side done (subagent): `346f9cbf8e` (`fp8_source_rows`, row_copy tests j-58be5abf: 253
  passed) + `0c7a54179c` (`sc_dispatch.py`: `sc_send_plan`, `sc_dispatch`, `sc_drain`; tests
  j-137d1e8a: 11 passed, bitwise vs local_route_plan order, back-to-back + overflow). Bench
  j-bd533e4f: send plan 4.9 us (XLA, ~10 fusions), source rows 5.7 us, dispatch to all rows landed
  ~41 us at ~480 rows/sender on an 8x1 mesh (production: ~120 rows/sender per TP rank). wave_sems has
  one extra semaphore for sentinel rows. Integration must not copy the test's MSA-off flags; only
  xla_mosaic_unsafe_allow_multicore_remote_dma=True is needed.
- Consumer side done (subagent): `4bce2bb0bb`..`1272e0ba73` (`wave_lhs=(recv, row_scale, wave_sems,
  sentinels)` mode of gmm_v2_mlp_scatter: manual tile loop, per-wave waits, in-kernel fp8 -> bf16
  conversion bitwise as gather_rows; tests j-5ae11005 3/3 + existing suites pass). Local 8-core bench
  (j-db366962): wave mode 212-216 us vs gather + kernel 201-204 (kernel alone ~198): ~11-15 us of
  overhead not yet explained (skipping the conversion saves only ~2); the bench omits the all-to-all
  that wave mode replaces. row_scale via an XLA gather costs ~40 us -> must come from the route-plan kernel.
- Integration branch `gxd/q38-moe-sc-dispatch` (both sides cherry-picked onto daac504c20, HEAD
  03e1ff71d4); end-to-end wiring behind `TPU_MOE_A2A_SC_DISPATCH` by a subagent (in progress).
- Integration done (subagent): `4271bf8ce7` (route-plan kernel emits route_scale, the token scale per
  sorted route, through its sort) + `30d768c844` (`TPU_MOE_A2A_SC_DISPATCH=1`: fp8_source_rows ->
  sc_send_plan -> sc_dispatch (barrier id 8) -> route plan -> expert kernel wave_lhs; drain tied to the
  MoE result with an optimization barrier). Needs libtpu `--xla_mosaic_unsafe_allow_multicore_remote_dma=true`
  (cannot be per kernel). Tests j-b3f5a67d: 372 passed, incl. end-to-end bitwise vs the fp8 path on all
  owner-combine layouts and 3 layers back to back.
  Production j-060d6d5c-7a61-402f-946c (pending).
- Production j-060d6d5c-7a61-402f-946c (`30d768c844`, SC dispatch + route-plan kernel): **189.1 us,
  worse than 182.3**. The expert kernel starts 11 us earlier (51 vs 62 us into the layer: no
  all-to-all wait, no gather) but takes 136.4 us (vs 117.8): ~33 us of it is outside every named
  region (vs ~10 before) = the wave waits / per-tile row copies / conversion. Before the kernel:
  sc_send_plan XLA ops ~6 us ahead of the dispatch start, and 12 serial ~0.6 us copy-dones of the
  route-plan kernel's VMEM outputs (~7 us).
  Fixes: e988996493 (route-plan outputs staged and DMA'd to HBM in-kernel), b83b6dfa23 (named scopes
  wave_wait / wave_rows_wait / wave_convert / wave_final_wait). Tests j-8236fdcf, production
  j-e16cbd7c (pending).
- Production j-e16cbd7c-2f05-41fc-aadc (+ e988996493, b83b6dfa23): 188.3 us; kernel 140.7 with the new
  regions: **wave_rows_wait 15.7 us** (each tile's row copy queued behind the next expert's ~25 MB of
  weight copies, the same queueing as iteration 9), wave_convert 3.6, wave_wait (rows still in
  flight) 3.0. Tests j-8236fdcf: 87 passed.
  Fixes: 54bd8e5db4 (at each tile start, before it issues weight copies, start the next tile's rows
  copy if a semaphore read shows its waves landed), bd07030a1b (SC path drops exchange_plan: whole-
  group never overflows). Tests j-584f76c9, production j-42d63d6d (pending; has 54bd8e5db4 only).
  Pallas sc_send_plan (~6 us of XLA today): subagent, branch gxd/q38-sc-send-plan-kernel.
- Production j-42d63d6d-e964-4384-8400 (+ 54bd8e5db4): **182.1 us** (ties the best 182.3); kernel 134.3:
  wave_rows_wait 15.7 -> 8.5, now ALL on tile 0 (8.4 us; tiles 1-11 ~0.01): tile 0's wait covers the
  HBM -> SMEM copy of the whole 7680-entry row-scale table, issued with the 2 x 7680 dest tables.
  wave_wait #0 2.7 us (expert 0's rows still landing), later waves ~0.03. Tests j-584f76c9: 60 passed.
  Fix cd3b66a685: copy only the routed rows' prefix of the SMEM tables (aligned chunks). Also in this
  build: bd07030a1b (no exchange_plan on the SC path). Tests j-680ca613, production j-b2022c4a (pending).
- Production j-b2022c4a-7031-4b68-9137 (+ cd3b66a685, bd07030a1b): **180.4 us (new best)**, exchange_plan
  gone (plan 0.0), kernel 134.3; tile 0 still starts its matmuls 12-18 us into the kernel: its rows
  copy, issued right before the first expert's ~25 MB of weight copies, completes only after them.
  Fix ec72b78956: wait for the first rows + scales, then start the first weights. Tests j-456ade64,
  production j-39d62aeb (pending).
- Production j-39d62aeb-34e0-43bd-93a2 (+ ec72b78956): **176.1 us**, kernel 130.6 (first matmul at 13 us:
  entry 2.7, expert 0's rows landing ~3, rows copy 1.2, first weight part 2.6).
  BUT tests j-456ade64 failed 25/60: PrefixCopy (cd3b66a685) copied short tables (length not a
  multiple of 512) as one chunk at a dynamic offset, which Mosaic rejects at the test shapes; the
  production shape (7680 = 15 x 512) compiles, so its numbers stand. Fix b5340b0fdb (static whole
  copy). 61d06bd5b4: first gate/up part starts at entry while the first wave lands.
  Tests j-5ebe2c78: 97 passed (wave lhs, fp8 dispatch incl. SC, owner tail, buckets).
- fp8 combine (expert outputs as fp8 + per-row f32 scale, owner tail dequantizes): not explored
  before; subagent on branch `gxd/q38-moe-fp8-combine` (flag TPU_MOE_A2A_FP8_COMBINE). Expectation:
  modest (row-send issue and the final wait's skew do not scale with bytes; accuracy hit larger
  than the dispatch's since rounded outputs are summed).
- Production j-6ad82ac4-b891-4f17-a96c (+ b5340b0fdb, 61d06bd5b4): **175.3 us** (median 174.2); kernel 129.7,
  first matmul at 11.6 us (the first part's wait moved into tile 0's rows copy, now queued behind it).
- Pallas sc_send_plan (subagent, `0732da65a2`, cherry-picked as a2c9caaac8; tests j-9ec88394: 92 passed,
  bitwise vs the XLA plan): ~4 us vs 7 us (bench). Production j-33b6b4e2-4dd2-4018-8582: **167.4 us (new best)**, other glue 29.8 -> 23.0 us.
- Fused router top-k cherry-picked (6f5e6b5996): production j-ab4fdd5e-505d-4b9f-9ef0: **165.1 us (new best)**, top-k 2.4 -> 1.0.
- Requant block 512 (user OK; recipe MOE_REQUANTIZE_BLOCK_SIZE=512) on the same build: j-547cedbe (pending).
- Removing the owner tail's TP replication (sequence-parallel residual stream): feasibility study by a
  subagent (pending).
- fp8 combine (subagent, `ffb968c143` + `409deed5c0` on gxd/q38-moe-fp8-combine; tests j-1d497039 12/12,
  j-2cbf201b 42): rows uint32 [8, H/32 + 128] (fp8 + a scale tile, one DMA per row, 25% fewer bytes);
  error vs bf16 combine 2.4-2.7% per row (fp8 dispatch: 3.4-3.9%). Local 8-device bench: +1.4 us
  (bytes -2, amax + dequantize +3.3). Production on the best build (cherry-pick 54ac9a6dae):
  j-94f09572-569b-4285-9279: **166.1 us vs 165.1 without, kernel 131.1 vs 129.2: slower, not adopted**
  (stays opt-in on its branch).
- Sequence-parallel residual / no owner-tail TP replication (subagent study, /tmp/sp-study/): **no-go**.
  The 16-way dispatch needs every TP rank to hold all 48 tokens; SP that keeps it costs RS 17.6 + AG
  ~8.5 + AG ~8.5 ~= 35 us/layer vs today's AR 20.2 + tail ~6 (+5..9 us worse); a 64-way dispatch from
  each rank's own tokens + replicated shared expert is break-even at best (>1500 lines). TP4
  collectives here are latency-bound (a 192 KiB AR still takes 14.5 us). Upstream has the SP mode
  (use_attn_reduce_scatter_for_moe), forced off by the plugin. Better lever: the AR's latency floor.
- Requant block 512 (j-547cedbe-e26f-4855-9f56, knob confirmed 512 in the rendered recipe): **166.2 us,
  no gain**; up/down regions unchanged (~28.4 us each): the bundle bound is the per-MXU weight-push
  rate + fp4 widen, the per-block pop/mul/add it halves are a minor share. Keep 256 (more accurate).
- Owner-tail sum (subagent, gxd/q38-moe-tail-sum; tests j-09988d7a 99 passed, bitwise): d598a6ba48 keeps
  each TP rank's running partial in its spent stage row instead of 4 selects per route (12-token sum
  2.83 -> 1.75 us locally); ec8c6de97a waits per token for its rows. Cherry-picked as 39ee218a22 and
  89ec792b1f; production A/B: commit 1 only j-b3769b4e-8720-4172-a6d4: **164.8 us (final best)**, kernel
  127.9; both j-f8fa277c-64c8-413f-bca2: 165.3 (kernel 129.2): the per-token waits give nothing ->
  reverted on the branch (24ceba849a).
