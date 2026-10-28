# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_MTP_UNION_STATS: verify-row selection overlap accounting."""

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen4_exp.mtp_union_stats import (
    FIELDS,
    STATS,
    mtp_union_step,
)


def groups_to_tokens(groups, k):
    """Token positions of whole groups (4 tokens each), -1 padded to k."""
    tokens = [g * 4 + o for g in groups for o in range(4)]
    return tokens + [-1] * (k - len(tokens))


def test_union_counts_requests_with_several_rows():
    k = 40
    rows = [
        groups_to_tokens([1, 2, 3, 4], k),  # request 0, rows 0-2
        groups_to_tokens([2, 3, 4, 5], k),
        groups_to_tokens([1, 2, 3, 4], k),
        groups_to_tokens([7, 8], k),  # request 1: one row, not counted
        groups_to_tokens([10, 11, 12], k),  # request 2, rows 4-5
        groups_to_tokens([20, 21, 22], k),
        [-1] * k,  # padding row
        [-1] * k,
    ]
    selected = jnp.asarray(rows, jnp.int32)
    qs = jnp.asarray([0, 3, 4, 6, 6], jnp.int32)  # trailing padded request
    stats = mtp_union_step(selected, qs, jnp.zeros(STATS, jnp.int32))
    v = dict(zip(FIELDS, np.asarray(stats).tolist()))
    assert v["steps"] == 1 and v["requests"] == 2 and v["rows"] == 5
    assert v["row_groups"] == 4 + 4 + 4 + 3 + 3
    assert v["union_groups"] == 5 + 6  # {1..5} and {10,11,12,20,21,22}
    assert v["max_row_groups"] == 4 + 3
    # union/max_row: 5/4 = 125% -> le125; 6/3 = 200% -> le200
    assert v["le125"] == 1 and v["le200"] == 1 and v["requests"] == sum(
        v[f] for f in FIELDS[6:]
    )


def test_prefill_sized_steps_are_skipped():
    selected = jnp.zeros((300, 8), jnp.int32)
    qs = jnp.asarray([0, 300], jnp.int32)
    stats = mtp_union_step(selected, qs, jnp.zeros(STATS, jnp.int32))
    assert int(stats.sum()) == 0
