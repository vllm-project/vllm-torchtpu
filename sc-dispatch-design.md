# SparseCore side-channel MoE dispatch: design and interfaces

Goal: replace `row_copy_send_fp8` -> XLA SC `ragged_all_to_all` (~27 us, on the critical path)
-> `row_copy_gather` (5.4 us) with an SC kernel that DMAs each route's fp8 row straight to its
landing row on the receiver, in wave (= receiver-local expert) order, and exits right after
issuing. The TC expert kernel waits per wave while the rows land, so the transfer hides under the
expert compute (12 experts x ~7.2 us).

Proven on this stack (branch `gxd/q38-sc-wave-dispatch-test`, tests/kernels/test_sc_wave_dispatch.py,
commits f0758b098f + bd1bfcc747; read them, they are the reference patterns):
- SC scalar-subcore `pl.kernel`: barrier (barrier semaphore + collective_id), remote row DMAs into
  each receiver's HBM buffer, one DMA semaphore per wave in the receiver's TC memory
  (`SemaphoreType.DMA((K,)) @ tc_mesh` output), sentinel row per empty (sender, receiver) pair.
- Issue-and-exit: the dispatch kernel does not wait for its sends; its send semaphore is an output
  `DMA((1,)) @ sc_mesh`, waited by a second SC "drain" kernel that also takes the source rows (keeps
  them alive). The TC consumer starts while rows are still in flight; XLA runs SC kernels async.
- DMA issue ~20-35 ns per extra DMA; fixed cost (launch, barrier, consumer, drain) ~6.5 us/call.

## Shapes (Qwen3.8 decode, 32 chips)

Mesh axes (DP_AXIS = 16 members, TP_AXIS = 4). EP rank of device (d, t) owns local experts
`[ep * 12, ep * 12 + 12)`; check `exchange_plan.py` / `ragged_a2a_moe.py` for the exact
(d, t) -> ep mapping and `experts_start`. S = 48 tokens per DP group, K = 10 (top-k), hidden 8192.
The capacity exchange today sends over DP_AXIS only: device (d, t) sends its DP group's tokens'
rows to devices (d', t) (same TP coordinate). Keep that: the dispatch is a collective over the 16
members of DP_AXIS.

Everything every device needs is in the all-gathered routing words (`_gather_routing`:
s32 [D, K + 1, S], id << 16 | bf16 weight bits, plus the fp8 scale word row) and `fits`.

## Landing order (must equal the route plan's order)

The route plan (`route_plan` in ragged_a2a_moe.py, or the kernel `local_route_plan` on branch
`gxd/q38-moe-route-plan-kernel`, commit 6cad38aef9) sorts a receiver's computed routes by
(local expert, source DP member, token, k). Route j of that order is lhs row j of the expert
kernel, and `dest_member[j]`, `dest_row[j]`, `group_sizes` come from it. The dispatch lands route
(source s, token t, k) with local expert e on receiver r at row

  land = start_e(r) + #{computed routes of r with expert e and (s', t') < (s, t)}

(a token's routes name distinct experts; follow the route plan's tie handling for repeated ids).
A route is "computed" iff its expert is local to r, its weight is finite and nonzero (skip padded
tokens), and `fits`. Rows past sum(group_sizes) are never written.

## Components

1. Sender plan (TC, Pallas or XLA, from the gathered words): for this device's sends, a table
   per route: (source token t, destination member d', landing row, wave e), issued wave-major
   (all destinations' wave 0 first). Plus per destination whether the pair is empty (sentinel).
2. Source rows (TC): this DP group's S token rows quantized to fp8 with one f32 scale per token,
   as uint32 words [S, 8, H / 32] in `fp8_row_words` order (reuse `copy_send_rows_fp8`'s quantize
   math and `_fp8_scales`; bitwise-same words as today's send buffer rows).
3. SC dispatch kernel: barrier over DP_AXIS, per-route row DMAs (one 8 KB row each) to
   `recv.at[land]` on member d', signalling `wave_sems.at[e]` there; returns
   (recv u32 [P, 8, H/32] HBM, wave_sems DMA((K_local[+1],)) @ tc_mesh, send_sem @ sc_mesh).
   P = max routes per receiver (whole group: D * S * min(K, 12) = 7680), a multiple of 16.
4. SC drain kernel: waits this device's sends, takes the source rows; its tiny output is consumed
   late (e.g. by the op after the expert kernel) so XLA keeps it.
5. Expert kernel (TC), `gmm_v2_mlp_scatter` variant: instead of a bf16 lhs from the gather it
   takes recv (fp8 words, route order), `row_scale_sorted` [P] f32 (the token scale of route j),
   wave_sems and group_sizes. Before tile g of expert e it waits until wave e's rows have landed
   (wave e expects group_sizes[e] rows; wait in row-sized chunks like gmm_v2 `wait_rows`), copies
   the tile's rows to VMEM and converts each row to bf16 exactly as `gather_rows` does
   (`pltpu.bitcast(words, fp8).astype(f32) * scale -> bf16`, zero rows past the routed ones,
   canonical zeros), then runs the existing matmul body. Output must be bitwise the current path.
   Waits for a wave are cumulative per expert; every wave semaphore must end balanced.

## Integration

New env flag `TPU_MOE_A2A_SC_DISPATCH` (default off), plumbed like `TPU_MOE_A2A_FP8_DISPATCH`,
only with fp8 dispatch, whole-group sizes, plan kernel, owner combine. The route-plan kernel
(`TPU_MOE_ROUTE_PLAN_KERNEL`) provides token_sorted / group_sizes / dest_* / row scales.
End-to-end result must be bitwise the current fp8 path.
