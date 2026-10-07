# Copyright 2026 Google LLC
"""Padding-sentinel coverage for fused-EP routing tables."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402


def _padded_routing_inputs():
    ep, t_local, g_local, topk = 8, 8, 64, 10
    e_total = ep * g_local
    total_tokens = ep * t_local
    rng = np.random.default_rng(11)
    idx = rng.integers(0, e_total, (total_tokens, topk), dtype=np.int32)
    valid = np.ones(total_tokens, dtype=bool)
    for rank in range(ep):
        start = rank * t_local
        valid[start + t_local - 2 : start + t_local] = False
    idx[~valid] = e_total

    capacity = 128
    kwargs = dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=host.routing_block(t_local, topk),
        tile_m=capacity,
        shard_stride=host.ragged_stride_bound(total_tokens, topk, e_total, capacity),
    )
    return idx, valid, kwargs


def test_invalid_expert_sentinels_contribute_no_routed_rows():
    idx, valid, kwargs = _padded_routing_inputs()
    ep, stride, topk = kwargs["ep"], kwargs["shard_stride"], idx.shape[1]
    routing = host.build_routing_tables(jnp.asarray(idx), **kwargs)

    assert int(routing.run_rows.sum()) == int(valid.sum()) * topk
    flat_valid = np.repeat(valid, topk)
    slab_row = np.asarray(routing.slab_row)
    arrival_row = np.asarray(routing.arrival_row).reshape(-1)
    mirror_row = np.asarray(routing.mirror_row).reshape(-1)
    assert np.all(slab_row[flat_valid] < routing.rows_alloc)
    assert np.all(slab_row[~flat_valid] == host.off_shard_slab_row(stride, ep))
    assert np.all(arrival_row[~flat_valid] == 0)
    assert np.all(mirror_row[~flat_valid] == 0)
    for me in range(ep):
        rows = np.asarray(host.local_slab_rows(routing, me, shard_stride=stride))
        assert np.all(rows[~flat_valid] >= stride)


def test_sharded_plan_matches_replicated_plan_with_padding():
    idx, valid, kwargs = _padded_routing_inputs()
    ep, e_total, t_local = kwargs["ep"], kwargs["e_total"], kwargs["t_local"]
    stride = kwargs["shard_stride"]
    whole = host.build_routing_tables(jnp.asarray(idx), **kwargs)
    counts = np.stack(
        [
            np.bincount(
                idx[d * t_local : (d + 1) * t_local].reshape(-1),
                minlength=e_total + 1,
            )[:e_total]
            for d in range(ep)
        ]
    ).astype(np.int32)

    for me in range(ep):
        shard = host.build_routing_tables_sharded(
            jnp.asarray(idx),
            jnp.int32(me),
            all_gather_rows=lambda row: jnp.asarray(counts),
            **kwargs,
        )
        np.testing.assert_array_equal(
            np.asarray(shard.pos),
            np.asarray(whole.arrival_row[me * t_local : (me + 1) * t_local]),
        )
        np.testing.assert_array_equal(
            np.asarray(shard.mirror_pos),
            np.asarray(whole.mirror_row[me * t_local : (me + 1) * t_local]),
        )
        np.testing.assert_array_equal(
            np.asarray(host.shard_token_gather(whole, me, shard_stride=stride)),
            np.asarray(host.shard_token_gather(shard, me, shard_stride=stride)),
        )
