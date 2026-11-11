# PCP Streaming RPA Kernel

This directory contains the prefill context parallel (PCP) implementation for
TPU ragged paged attention. Each PCP rank stores a local shard of the paged KV
cache, writes its owned fresh KV rows into that local shard, and executes
unified causal Tokamax Splash attention over active virtual-page groups without
materializing the full paged KV table across inactive block-table slots.

## High-Level Flow

The production vLLM path enters through `vllm_adapter.py` and `wrapper.py`.
The attention backend passes the standard vLLM metadata tensors:

- `seq_lens`: total KV length for each active request.
- `block_tables`: paged KV cache block table for each request.
- `query_start_loc`: cumulative query-token offsets for the current batch.
- `request_distribution`: vLLM request distribution metadata.

`wrapper.py` runs inside a JAX `shard_map` over the PCP mesh:

1. `stage_plan.py` (`build_pcp_local_stage_plan`) builds a `PCPStagePlan`
   sorting active virtual pages ahead of inactive block-table entries, computing
   segmented query coordinates (`q_seg_ids`, `q_sequence`), and coalescing local
   KV writeback descriptors.
2. `kernel.py` (`pcp_streaming_attention_page_groups_packed_local_from_metadata`)
   writes captured fresh KV into the local paged KV-cache shard (across window 0
   plus overflow windows when a cache rank owns more than `local_tokens` rows)
   and runs unified Tokamax Splash attention (`splash_attention_kernel.py`) over
   `[history + current]`.

## Token and Cache Layout

PCP uses rank-major token packing. For a configured `pcp_size` and
`cp_kv_cache_interleave_size`, global query positions are split into interleave
chunks:

```text
rank 0: [0, interleave), [pcp_size * interleave, ...]
rank 1: [interleave, 2 * interleave), ...
...
```

The host runner packs these chunks by rank before moving inputs to the device.
Each rank sees compact local query rows while `PCPStagePlan` reconstructs the
request-absolute causal position (`q_sequence`) and request segment ID
(`q_seg_ids`) for every row.

Each virtual page spans `page_size * pcp_size` tokens striped across PCP ranks
at granularity `cp_kv_cache_interleave_size`. During each Splash group step,
ranks transpose only the active local pages into head-major `[kv_heads, ...]`
layout prior to `all_gather(tiled=True)` and slice padded odd-KV-head FP8
sublane planes afterward so the local transpose remains a zero-copy bitcast.

## Active Virtual-Page Ladder and Multi-Group Streaming

For `vpage_group_size == 128` (`PCP_STREAMING_RPA_VPAGE_GROUP_SIZE`):

- When `num_active_vpages <= 512`, `_attend_over_vpage_groups` dispatches one
  single-group Splash call (`save_residuals=False`) at the smallest `block_kv`-
  aligned rung in `PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS = (32, 64, 96, 128, 256, 512)`.
- When `num_active_vpages > 512`, `_attend_multi_vpage_groups` streams `128`-
  virtual-page groups (`save_residuals=True`) and merges per-group outputs
  online in `float32` via natural-log `logsumexp`.

## Legacy Ring Schedule Reference (`schedule.py`)

> Note: `schedule.py` and the split current/history ring Pallas helpers in
> `kernel.py` are retained temporarily for legacy ring unit tests and are slated
> for removal in a follow-up cleanup PR once unified Splash streaming lands.

## Entry Points

- `pcp_streaming_attention_page_groups_packed_local_from_metadata`: production
  metadata-driven entry consuming `PCPStagePlan` and running local KV writeback
  followed by unified Tokamax Splash attention.

## Current Constraints

The production path enforces the following invariants:

- Full causal attention only; sliding-window attention is not supported here.
- `cp_kv_cache_interleave_size <= page_size`, and `page_size` must be
  divisible by `cp_kv_cache_interleave_size`.
- `num_lanes == 1`.
- `head_dim == 64` and attention sinks are not supported.
