# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap persistent TensorCore scoring with SparseCore HBM selection."""

from inspect import unwrap

import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

from ..streaming_topk import streaming_topk
from .persistent_logits import score_window
from .tuned import (
    PACKED_FP8_MQA_CONFIG,
    PACKED_FP8_PREFILL_ROWS,
    topk_reconstruction_threshold,
)


def prefill_topk(
    q,
    banks,
    weights,
    lengths,
    pages,
    offsets,
    requests,
    *,
    layout,
    width,
    k,
    unordered=False,
    reference_rows=None,
):
    """Keep six-row head groups, resetting at reference_rows when supplied.

    None matches one standalone scoring call over all queries. Serving callers
    can retain the boundaries of their existing score calls while reusing K.
    """
    tokens = q.shape[0]
    info = pltpu.get_tpu_info().sparse_core
    wave = info.num_cores * info.num_subcores
    rows = min(PACKED_FP8_PREFILL_ROWS[1], max(wave, (tokens + 1) // 2))
    windows = (tokens + rows - 1) // rows
    rows = (tokens + windows - 1) // windows
    tile = PACKED_FP8_MQA_CONFIG.query_tile
    starts = jnp.arange(windows, dtype=jnp.int32) * rows
    owners = jnp.where(
        requests[starts] >= 0,
        jnp.minimum(requests[starts], pages.shape[0] - 1),
        pages.shape[0] - 1,
    )
    group_starts = offsets[owners]
    if reference_rows is not None:
        group_starts = jnp.maximum(
            group_starts, starts // reference_rows * reference_rows
        )
    skips = (starts - group_starts) % tile
    skips = skips.at[0].set(0)
    skips = jax.lax.optimization_barrier(
        jnp.broadcast_to(skips[:, None], (windows, info.num_lanes))
    )
    state = None
    group_base = config.reserve_scheduling_group_ids(max(1, windows - 1))

    def score(i, group):
        nonlocal state
        size = min(rows, tokens - i * rows)
        scores, state = score_window(
            q,
            banks,
            weights,
            lengths,
            pages,
            offsets,
            layout=layout,
            width=width,
            start=i * rows,
            rows=size,
            state=state,
            reference_rows=reference_rows,
        )
        if group is not None:
            scores = xla_metadata.set_xla_metadata(scores, _scheduling_group_id=group)
        return scores

    def select(i, scores, group):
        size = min(rows, tokens - i * rows)
        return unwrap(streaming_topk)(
            scores,
            k=k,
            row_start=skips[i],
            row_count=size,
            valid_lengths=lengths[i * rows : i * rows + size],
            exclude_neg_inf=True,
            reconstruct_threshold=topk_reconstruction_threshold(k),
            unordered=unordered,
            scheduling_group_id=group,
        )

    outputs = []
    current = score(0, None)
    previous = None
    for i in range(windows - 1):
        if previous is not None:
            previous, current, q = jax.lax.optimization_barrier((previous, current, q))
            outputs.append(previous)
        group = group_base + i
        following = score(i + 1, group)
        previous = select(i, current, group)
        current = following
    if previous is not None:
        previous, current = jax.lax.optimization_barrier((previous, current))
        outputs.append(previous)
    outputs.append(select(windows - 1, current, None))
    return jnp.concatenate(outputs)
