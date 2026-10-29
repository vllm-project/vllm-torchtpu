# SPDX-License-Identifier: Apache-2.0
"""Debug-only: overlap of the MTP verify rows' QSA selections per request.

``VLLM_TPU_QSA_MTP_UNION_STATS = <layer>``: on decode / verify steps (at most
``MAX_ROWS`` rows), for every request with at least two rows (the verify row
and its draft rows), the number of distinct compressed groups each row selects
and the size of the union over the request's rows. A union fetch would read
``union`` groups instead of ``sum(rows)``. Accumulated into a small int32
device buffer the model prints outside the compiled graph.
"""

import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

RATIO = 4
MAX_ROWS = 256
GROUP_BITS = 17  # groups per row < 2**17 (262144 tokens / 4)
# Union / largest row, in percent: histogram bucket upper bounds.
BUCKETS = (110, 125, 150, 200, 300)
FIELDS = (
    ("steps", "requests", "rows", "row_groups", "union_groups", "max_row_groups")
    + tuple(f"le{b}" for b in BUCKETS)
    + ("gt",)
)
STATS = len(FIELDS)


def mtp_union_step(selected, query_start_loc, stats):
    """``stats`` plus this step's sums. ``selected``: [T, k] token positions
    (-1: none); ``query_start_loc``: [R + 1] row offsets of the requests."""
    rows_total = selected.shape[0]
    if rows_total > MAX_ROWS:
        return stats
    big = jnp.iinfo(jnp.int32).max
    nreq = query_start_loc.shape[0] - 1
    qs = query_start_loc.astype(jnp.int32)
    row = jnp.arange(rows_total, dtype=jnp.int32)
    req = jnp.clip(jnp.searchsorted(qs, row, side="right") - 1, 0, nreq - 1)
    live_row = row < qs[nreq]
    g = jnp.where((selected >= 0) & live_row[:, None], selected // RATIO, big)
    g = jnp.sort(g, axis=1)
    prev = jnp.concatenate([jnp.full_like(g[:, :1], -1), g[:, :-1]], axis=1)
    distinct = (g != big) & (g != prev)
    per_row = jnp.sum(distinct.astype(jnp.int32), axis=1)
    keys = jnp.where(distinct, (req[:, None] << GROUP_BITS) + g, big).reshape(-1)
    keys = jnp.sort(keys)
    kprev = jnp.concatenate([jnp.full((1,), -1, keys.dtype), keys[:-1]])
    first = (keys != big) & (keys != kprev)
    union = jax.ops.segment_sum(
        first.astype(jnp.int32), jnp.where(first, keys >> GROUP_BITS, nreq), nreq + 1
    )[:nreq]
    nrows = qs[1:] - qs[:-1]
    counted = nrows >= 2
    seg = jnp.where(live_row, req, nreq)
    row_groups = jax.ops.segment_sum(per_row, seg, nreq + 1)[:nreq]
    max_row = jax.ops.segment_max(per_row, seg, nreq + 1)[:nreq]
    m = counted.astype(jnp.int32)
    ratio = union * 100
    hist = []
    lo = jnp.zeros_like(union, jnp.bool_)
    for b in BUCKETS:
        hit = counted & ~lo & (ratio <= b * jnp.maximum(max_row, 1))
        hist.append(jnp.sum(hit.astype(jnp.int32)))
        lo = lo | (ratio <= b * jnp.maximum(max_row, 1))
    hist.append(jnp.sum((counted & ~lo).astype(jnp.int32)))
    add = jnp.stack(
        [
            jnp.int32(1),
            jnp.sum(m),
            jnp.sum(nrows * m),
            jnp.sum(row_groups * m),
            jnp.sum(union * m),
            jnp.sum(max_row * m),
            *hist,
        ]
    )
    return stats + add.astype(jnp.int32)


@functools.cache
def build_mtp_union_stats_op():
    def core(
        selected: jax.Array, query_start_loc: jax.Array, stats: jax.Array
    ) -> jax.Array:
        return mtp_union_step(selected, query_start_loc, stats)

    op = pallas.jax_op("qwen4_exp::qsa_mtp_union_stats_v1", core)

    @op.register_fake
    def fake(selected, query_start_loc, stats):
        return torch.empty_like(stats)

    return op


def format_mtp_union_stats(values) -> str:
    v = dict(zip(FIELDS, (int(x) for x in values)))
    req = max(v["requests"], 1)
    hist = " ".join(f"{f}={v[f]}" for f in FIELDS[6:])
    return (
        f"steps={v['steps']} requests={v['requests']} "
        f"rows/request={v['rows'] / req:.2f} "
        f"groups/row={v['row_groups'] / max(v['rows'], 1):.1f} "
        f"union/sum_rows={v['union_groups'] / max(v['row_groups'], 1):.3f} "
        f"union/max_row={v['union_groups'] / max(v['max_row_groups'], 1):.3f} | {hist}"
    )
