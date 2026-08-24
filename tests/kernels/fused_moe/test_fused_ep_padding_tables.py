# Copyright 2026 Google LLC
"""Padding-sentinel coverage for fused-EP routing tables."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402


def test_invalid_expert_sentinels_contribute_no_routed_rows():
    ep, t_local, g_local, topk = 8, 8, 64, 10
    e_total = ep * g_local
    total_tokens = ep * t_local
    rng = np.random.default_rng(11)
    idx = rng.integers(0, e_total, (total_tokens, topk), dtype=np.int32)
    valid = np.ones(total_tokens, dtype=bool)
    for rank in range(ep):
        start = rank * t_local
        valid[start + t_local - 2:start + t_local] = False
    idx[~valid] = e_total

    capacity = 128
    stride = host.ragged_stride_bound(total_tokens, topk, e_total, capacity)
    routing = host.build_routing_tables(
        jnp.asarray(idx),
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=host.routing_block(t_local, topk),
        tile_m=capacity,
        shard_stride=stride,
    )

    assert int(routing.run_rows.sum()) == int(valid.sum()) * topk
    flat_valid = np.repeat(valid, topk)
    slab_row = np.asarray(routing.slab_row)
    arrival_row = np.asarray(routing.arrival_row).reshape(-1)
    assert np.all(slab_row[flat_valid] < routing.rows_alloc)
    assert np.all(slab_row[~flat_valid] == routing.rows_alloc)
    assert np.all(arrival_row[~flat_valid] == 0)
