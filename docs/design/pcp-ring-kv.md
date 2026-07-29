# Draft design: ring KV inside the all-gather PCP kernel (`rpa_v3_cp`)

Status: **draft for discussion**. Not implemented. Written against
`worktree-pcp` @ `40d3d1b1`.

## 1. Problem

`rpa_v3_cp` (all-gather PCP) moves **queries**: each rank all-gathers the full
`2*pcp*C` query buffer, attends it against its own KV shard, and LSE-merges the
partials. All communication happens in the wrapper via `lax.all_gather`; the
kernel itself is transport-free.

This costs **memory**: every rank materialises the whole gathered Q (`P x |Q|`)
plus its attention output before the merge. At very long contexts with large
chunked-prefill buckets that is the dominant VMEM/HBM pressure.

Ring KV inverts it: **Q stays local**, KV shards rotate around the ring, and
each rank folds `P` partial results into a running accumulator. Peak memory
becomes `|Q_local| + 2 x |KV_shard|` (double-buffered) instead of `P x |Q|`.

### When ring actually wins

Per layer, per step:

| approach | bytes moved | peak extra memory |
|---|---|---|
| all-gather Q | `(P-1) x |Q|` | `P x |Q|` |
| ring KV | `(P-1) x |KV|` | `2 x |KV_shard|` |

The cache phase attends the whole context, so `|KV| >> |Q|` there and ring is
*more* expensive in bandwidth. **Ring is a memory play, not a bandwidth play.**
It is worth doing when:

- the current chunk `Q` is large (big chunked-prefill buckets, e.g. 262k-context
  serving), and
- we are VMEM/HBM bound rather than interconnect bound.

It is **not** worth doing for gsm8k-scale prompts, where all-gather-Q is both
cheaper and simpler. This should be a per-deployment choice, not a replacement.

## 2. Prior art in this repo

`pcp_streaming_rpa` already implements in-kernel ring DMA on this exact stack,
so feasibility is settled. Relevant machinery (`pcp_streaming_rpa/kernel.py`):

```python
my_id      = lax.axis_index(pcp_axis_name)          # ~603
next_rank  = lax.rem(my_id + 1, pcp_size)
next_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, next_rank)

def _round_loop(round_idx, carry):                  # ~684
    curr_slot = lax.rem(round_idx, 2)               # double buffer
    src_rank  = lax.rem(my_id + pcp_size - round_idx, pcp_size)
    ...
    remote_op = pltpu.make_async_remote_copy(       # ~715
        send_sem=remote_send_sems.at[lane, safe_round_idx],
        recv_sem=remote_recv_sems.at[lane, safe_round_idx],
        device_id=next_device_id,
        device_id_type=pl.DeviceIdType.MESH,
    )
```

Key patterns to reuse: two-slot double buffering (`round_idx % 2`), per-round
send/recv semaphore pairs indexed by a clamped `safe_round_idx`, a
`group_slot_free` semaphore to stop a group overrunning the next, and a
host-side schedule streamed in per round.

## 3. What `rpa_v3_cp` already gives us

The all-gather kernel was built for two-phase LSE merging, which is exactly what
a ring needs. Already present, no change required:

- `return_lse=True` — emits per-token log-sum-exp. **This is the enabler**;
  cross-round merging is impossible without it.
- `cp_rank` / `cp_group_size` — the kernel already knows its ring position.
- `skip_cache_attn` / `skip_current_attn` — already supports running over a
  *subset* of KV and merging afterwards, which is structurally identical to
  "attend round `i`'s shard".
- `kv_cache_lens`, `q_pos_offsets`, `pcp_chunk_size` — per-chunk causal masking
  already handles a rank's `[head | tail]` pair at arbitrary global offsets.

The merge math is already proven exact: `_lse_all_to_all` (now in
`layers/common/attention_interface.py`) matches a numpy reference to **0.00000
relative error in fp32** on a real 8-device mesh.

## 4. Proposed design

Add a **ring mode** to `rpa_v3_cp`, selected by a new kernel argument, leaving
the existing all-gather path untouched and default.

### 4.1 Execution model

Q stays local: each rank holds its `[head_chunk | tail_chunk]` of `2C` tokens.
For `round in 0..P-1`:

1. `src_rank = (my_id - round) mod P` — the shard currently resident.
2. Attend local Q against that shard, producing `(o_r, lse_r)`.
3. Fold into the accumulator with the existing merge:
   `(acc_o, acc_lse) = merge(acc_o, acc_lse, o_r, lse_r)`.
4. Concurrently DMA the resident shard to `next_device_id` and receive the
   next one into the alternate slot.

Rounds `1..P-1` overlap step 2 with step 4. Round `P-1` skips the send.

Because the merge is associative and order-independent (it is a max-shifted
sum), rounds may complete in any order — no ordering constraint beyond the
buffer handoff.

### 4.2 Which phase gets the ring

The wrapper runs two kernel launches. They should be decided independently:

- **Cache phase** — attends the strided KV cache, `|KV| >> |Q|`. Ring moves
  more bytes here. *Recommendation: keep all-gather-Q initially.*
- **Current phase** — attends the current chunk's KV, `|KV| ~= |Q|`. Bytes are
  a wash and the memory saving is real. *Recommendation: ring here first.*

Starting with the current phase also keeps the cache phase — the part we have
an unresolved correctness bug in — out of the change.

### 4.3 KV layout

All-gather stripes the cache with **interleave 1** (global token `g` on rank
`g % P`, local slot `g // P`); streaming uses **interleave `page_size`**. A ring
over interleave-1 is mathematically fine — each round hands you one rank's
strided shard, and `cp_rank`/`cp_group_size` already tell the kernel how to
index it. No layout change, and therefore **no change to cache
interoperability**: a ring-mode server still reads and writes caches written by
the existing all-gather path.

This is a deliberate divergence from streaming, whose page bookkeeping assumes
its own striping. We do **not** reuse `schedule.py`.

### 4.4 Schedule

All-gather's layout is uniform — every rank holds exactly `2C` query tokens and
an equal cache stride — so the per-round work is derivable in closed form from
`(round_idx, cp_rank, cp_group_size)`. We should **not** need streaming's
1748-line `schedule.py`; a handful of index expressions should do. If that turns
out to be false, it is a strong signal to stop and reuse streaming instead.

## 5. Interfaces

```python
# rpa_v3_cp/kernel.py
def ragged_paged_attention(..., ring_mode: bool = False,
                           ring_send_sems=None, ring_recv_sems=None, ...)
```

Wrapper (`layers/common/attention_interface.py`,
`sharded_pcp_allgather_ragged_paged_attention`): when ring mode is on, skip
`ag(k_l)`/`ag(v_l)` for the current phase and pass `k_l`/`v_l` directly; the
kernel does the rotation. `merge_attn_states` across the two phases is
unchanged.

No change to: `PCP_ALLGATHER_RPA_INPUT_PARTITION_SPECS`, the sequence layout
planner, or any runner metadata. Ring mode is invisible above the wrapper.

## 6. Risks and blockers

1. **Forking `kernel.py` from upstream — permanent, and the biggest cost.**
   `rpa_v3_cp/kernel.py` is currently byte-identical to tpu-inference
   `origin/main` (comments and import path only). That is why upstream's 24-test
   CP suite validates our kernel for free and why syncs are a no-op diff. Adding
   in-kernel DMA ends both. *Mitigation: propose the ring upstream instead of
   carrying it locally, or keep it in a separate `rpa_v3_cp_ring/kernel.py` that
   imports shared helpers.* Strongly prefer upstreaming.

2. **Mesh coverage at TP>1 — already broken, becomes load-bearing.**
   `get_or_create_pcp_mesh` builds a 1-D mesh over only the PCP group, so XLA
   rejects the collective at TP=4xPCP=2 (`replica_groups={{0,1}}` names 2 of 8
   flattened IDs). `DeviceIdType.MESH` addressing needs a mesh that spans the
   ring, so this must be fixed first — see
   `[[pcp-allgather-tp-mesh-limitation]]`. Fix is a 2-axis `(tp, pcp)` mesh.

3. **MPMD and `lax.axis_index` — needs confirming.** The all-gather *wrapper*
   cannot use `lax.axis_index` (lowers to `PartitionId`, rejected by torchtpu's
   MPMD compiler), which is why `cp_rank` is baked in as a constant. But
   streaming's ring calls `lax.axis_index(pcp_axis_name)` *inside* the Pallas
   kernel (`kernel.py:603`) and ships today, while another path threads
   `pcp_rank` as a parameter (`kernel.py:562`). **Open question:** is
   `axis_index` legal inside Mosaic but not at the shard_map level? Resolve this
   before writing code — if it is not, thread `cp_rank` through, which
   `rpa_v3_cp` already accepts.

4. **Correctness debt.** The all-gather path currently scores **gsm8k 0.00**
   (see `[[pcp-allgather-longcontext-bug]]`). Building ring mode on top means
   debugging two unknowns at once. **Ring work should not start until
   all-gather is green.**

## 7. Phasing

| phase | deliverable | gate |
|---|---|---|
| 0 | Fix all-gather correctness | gsm8k parity with streaming (~0.75) |
| 1 | Resolve blockers 2 and 3 | 2-axis mesh lands; TP=4xPCP=2 compiles |
| 2 | Ring in the **current phase**, `ring_mode=False` default | new tests below pass |
| 3 | Benchmark vs all-gather at 4k / 64k / 262k | memory win measured |
| 4 | Optional: ring the cache phase | only if phase 3 justifies it |

## 8. Test plan

Extend `tests/kernels/experimental/rpa_v3_cp/test_pcp_allgather.py`, which
already drives the wrapper through a real multi-device SPMD mesh:

- **Ring vs all-gather equivalence** — same inputs, `ring_mode` on/off, outputs
  must match to bf16 tolerance. This is the primary test.
- **Ring vs `ref_ragged_paged_attention`** — full-causal reference over the
  whole context, mirroring the existing `_assert_matches`.
- Both `pcp=2` and `pcp=4`; both `C >= PAGE` and `C < PAGE` (the branch
  production takes) — the existing parameterisations already cover this split.
- KV-cache write striding under ring mode, mirroring
  `test_kv_cache_write_chunk_below_page`.

## 9. Open questions

1. `lax.axis_index` inside Mosaic under MPMD — legal? (blocker 3)
2. Is the closed-form schedule sufficient, or do we need streaming's schedule
   machinery? (4.4)
3. Upstream first, or carry locally? (blocker 1) — affects whether this is a
   ~2-week or a ~1-week change.
4. Do we actually have a measured memory ceiling that all-gather hits? Phase 3
   is written as validation, but ideally we would have the motivating profile
   *before* phase 2.
