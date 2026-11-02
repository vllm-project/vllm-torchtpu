# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The routing-table identities the fused EP MoE kernel derives rather than ships.

`shard_transport_tables_in_blocks` returns three tables that are the same array
as another it already returns, and the kernel used to have all of them
prefetched into SMEM. Each one is a blocking HBM->SMEM DMA at the head of every
call, on every layer, and the whole scalar prefetch has to fit in 1 MB -- which
it did not under tensor parallelism. So the kernel now takes one copy and
derives the rest.

That is only safe while the identities hold, and they hold for reasons that
live in `build_routing_tables`, not in the shard builders: `region_rows` is
`run_rows_aligned` reshaped, and `recv_base` is a cumsum of ALIGNED run
lengths. Either could reasonably be changed -- tracking true lengths instead of
aligned ones is an obvious future edit -- and the kernel would then address
remote arrivals at the wrong rows, silently, on a shard other than rank 0.
These tests are what would catch that.

The arrays run on CPU. They are `jnp` calls like any other, so on a TPU host
they would otherwise compile and execute on a chip to compute tables that are
pure index arithmetic; `_cpu_backend` below keeps them off it, which is worth
about 3x the wall clock of this file. It does NOT keep the process off the
accelerator -- `jax.devices("cpu")` goes through `backends()`, which builds a
client for every registered platform including TPU. Nothing here needs one, but
do not read the fixture as a claim that the chips stay free.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_backend():
    """Keep every array in this file off the accelerator."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


# One decode-sized and one prefill-sized geometry, both at the 397B expert
# layout the serving path uses (512 experts over 8 shards, top-10).
GEOMETRIES = [
    pytest.param(8, 64, 64, 10, id="decode_t64_ep8_g64"),
    pytest.param(8, 512, 64, 10, id="prefill_t512_ep8_g64"),
    pytest.param(4, 128, 32, 8, id="ep4_g32_topk8"),
]


def _routing_inputs(ep, t_local, g_local, topk, *, skew, seed):
    """Routing indices and builder arguments for one step.

    `skew` reproduces low concurrency, where only a few experts are selected at
    all and most runs are empty -- the case where the alignment padding and the
    empty-region predicates actually differ from the uniform case.
    """
    e_total = g_local * ep
    T = t_local * ep
    rng = np.random.default_rng(seed)
    if skew:
        hot = rng.choice(e_total, size=max(4, e_total // 12), replace=False)
        idx = rng.choice(hot, size=(T, topk))
    else:
        idx = rng.integers(0, e_total, (T, topk))
    capacity = 128
    stride = host.ragged_stride_bound(T, topk, e_total, capacity)
    block = host.routing_block(t_local, topk)
    return jnp.asarray(idx.astype(np.int32)), dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=block,
        tile_m=capacity,
        shard_stride=stride,
    )


def _routing(ep, t_local, g_local, topk, *, skew, seed):
    idx, kwargs = _routing_inputs(ep, t_local, g_local, topk, skew=skew, seed=seed)
    return host.build_routing_tables(idx, **kwargs), kwargs["e_total"]


def _assert_arrays_equal(left, right, *, shard):
    assert len(left) == len(right)
    for i, (actual, expected) in enumerate(zip(left, right)):
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.asarray(expected),
            err_msg=f"shard {shard}, table {i}",
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
@pytest.mark.parametrize("seed", [0, 1])
def test_sharded_plan_produces_the_same_kernel_operands(
    ep, t_local, g_local, topk, skew, seed
):
    """The decomposed plan changes plan arithmetic, not kernel inputs."""
    idx, kwargs = _routing_inputs(ep, t_local, g_local, topk, skew=skew, seed=seed)
    whole = host.build_routing_tables(idx, **kwargs)
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    flat = np.asarray(idx)
    counts = np.stack(
        [
            np.bincount(
                flat[d * t_local : (d + 1) * t_local].reshape(-1), minlength=e_total
            )
            for d in range(ep)
        ]
    ).astype(np.int32)

    for me in range(ep):
        seen = []

        def all_gather_rows(row, seen=seen):
            seen.append(np.asarray(row))
            return jnp.asarray(counts)

        shard = host.build_routing_tables_sharded(
            idx, jnp.int32(me), all_gather_rows=all_gather_rows, **kwargs
        )
        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0].reshape(-1), counts[me])
        np.testing.assert_array_equal(
            np.asarray(shard.pos),
            np.asarray(whole.arrival_row[me * t_local : (me + 1) * t_local]),
        )

        whole_rows = np.asarray(host.local_slab_rows(whole, me, shard_stride=stride))
        shard_rows = np.asarray(host.local_slab_rows(shard, me, shard_stride=stride))
        whole_live = whole_rows < stride
        shard_live = shard_rows < stride
        np.testing.assert_array_equal(whole_live, shard_live)
        np.testing.assert_array_equal(whole_rows[whole_live], shard_rows[shard_live])

        for builder in (
            host.shard_transport_tables_in_blocks,
            host.shard_push_tables_in_rows,
        ):
            _assert_arrays_equal(
                builder(whole, me, e_total=e_total, ep=ep),
                builder(shard, me, e_total=e_total, ep=ep),
                shard=me,
            )
        whole_experts = host.shard_expert_slabs(whole, me, e_total=e_total, ep=ep)
        shard_experts = host.shard_expert_slabs(shard, me, e_total=e_total, ep=ep)
        _assert_arrays_equal(whole_experts, shard_experts, shard=me)
        np.testing.assert_array_equal(
            np.asarray(host.shard_token_gather(whole, me, shard_stride=stride)),
            np.asarray(host.shard_token_gather(shard, me, shard_stride=stride)),
        )

        whole_visit = host.expert_visit_list(whole_experts[0], g_local)
        shard_visit = host.expert_visit_list(shard_experts[0], g_local)
        _assert_arrays_equal(whole_visit, shard_visit, shard=me)
        np.testing.assert_array_equal(
            np.asarray(
                host.shard_count_vector(
                    whole, whole_experts[0], me, e_total=e_total, ep=ep
                )
            ),
            np.asarray(
                host.shard_count_vector(
                    shard, shard_experts[0], me, e_total=e_total, ep=ep
                )
            ),
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_transport_tables_the_kernel_derives(ep, t_local, g_local, topk, skew, seed):
    """The three tables the kernel no longer prefetches equal the ones it does.

    Checked on every shard, not just rank 0: the builders slice by `me`, so an
    identity can hold at one shard and fail at another.
    """
    routing, e_total = _routing(ep, t_local, g_local, topk, skew=skew, seed=seed)
    for me in range(ep):
        (
            commit_start,
            commit_len,
            contrib_off,
            push_src,
            push_len,
            push_dst,
            _totals,
        ) = host.shard_transport_tables_in_blocks(
            routing, jnp.int32(me), e_total=e_total, ep=ep
        )
        true_rows, recv_row_off, _t = host.shard_push_tables_in_rows(
            routing, jnp.int32(me), e_total=e_total, ep=ep
        )
        del commit_start, true_rows

        np.testing.assert_array_equal(
            np.asarray(push_src),
            np.asarray(contrib_off),
            err_msg=f"push_src != contrib_off on shard {me}",
        )
        np.testing.assert_array_equal(
            np.asarray(push_len),
            np.asarray(commit_len),
            err_msg=f"push_len != commit_len on shard {me}; region_rows is no "
            "longer run_rows_aligned reshaped",
        )
        np.testing.assert_array_equal(
            np.asarray(push_dst) * host.ROWBLK,
            np.asarray(recv_row_off),
            err_msg=f"recv_row_off != push_dst * ROWBLK on shard {me}; "
            "recv_base no longer accumulates ALIGNED run lengths, so the "
            "kernel would push remote rows to the wrong offset",
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_receive_offsets_are_block_aligned(ep, t_local, g_local, topk, skew):
    """The premise of the derivation, stated on its own.

    `push_dst * ROWBLK` can only reconstruct a row offset while every receive
    offset is a whole number of blocks. If that stops being true the identity
    above fails too, but this says why in one line.
    """
    routing, e_total = _routing(ep, t_local, g_local, topk, skew=skew, seed=7)
    for me in range(ep):
        _tr, recv_row_off, _t = host.shard_push_tables_in_rows(
            routing, jnp.int32(me), e_total=e_total, ep=ep
        )
        rem = np.asarray(recv_row_off) % host.ROWBLK
        assert not rem.any(), (
            f"receive row offsets are not {host.ROWBLK}-row aligned on shard "
            f"{me}: {np.unique(rem)}"
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("all_padded", [False, True], ids=["partial", "all_padded"])
def test_padded_tokens_are_dropped_from_routing_tables(
    ep, t_local, g_local, topk, all_padded
):
    """Tokens with sentinel expert `e_total` add zero work and match across plans."""
    idx_jnp, kwargs = _routing_inputs(ep, t_local, g_local, topk, skew=False, seed=11)
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    idx = np.asarray(idx_jnp).copy()

    if all_padded:
        # Warmup (`_dummy_run`) marks every token on every shard as padded.
        idx[:, :] = e_total
    else:
        # Mark the second half of each shard's tokens as padded (`topk_idx = e_total`),
        # and mark shard 1 as completely padded (`0` active tokens).
        for d in range(ep):
            start = d * t_local if d == 1 else d * t_local + t_local // 2
            idx[start : (d + 1) * t_local, :] = e_total
    idx_padded = jnp.asarray(idx)

    whole = host.build_routing_tables(idx_padded, **kwargs)
    off_shard = host.off_shard_slab_row(stride, ep)

    # Every padded pair's slab_row must be the off-shard sentinel.
    padded_pair_mask = idx.reshape(-1) == e_total
    np.testing.assert_array_equal(
        np.asarray(whole.slab_row)[padded_pair_mask],
        np.full(padded_pair_mask.sum(), off_shard, dtype=np.int32),
    )
    if all_padded:
        np.testing.assert_array_equal(
            np.asarray(whole.run_rows), np.zeros((e_total, ep), dtype=np.int32)
        )
        np.testing.assert_array_equal(
            np.asarray(whole.expert_rows_aligned), np.zeros(e_total, dtype=np.int32)
        )
        np.testing.assert_array_equal(
            np.asarray(whole.recv_rows), np.zeros(ep, dtype=np.int32)
        )
    else:
        # Shard 1 has 0 valid tokens, so it sends 0 rows to every expert.
        np.testing.assert_array_equal(
            np.asarray(whole.run_rows)[:, 1], np.zeros(e_total, dtype=np.int32)
        )

    counts = np.stack(
        [
            np.bincount(
                idx[d * t_local : (d + 1) * t_local][
                    idx[d * t_local : (d + 1) * t_local] < e_total
                ],
                minlength=e_total,
            )
            for d in range(ep)
        ]
    ).astype(np.int32)

    for me in range(ep):
        shard = host.build_routing_tables_sharded(
            idx_padded,
            jnp.int32(me),
            all_gather_rows=lambda _row: jnp.asarray(counts),
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
        whole_rows = np.asarray(host.local_slab_rows(whole, me, shard_stride=stride))
        shard_rows = np.asarray(host.local_slab_rows(shard, me, shard_stride=stride))
        whole_live = whole_rows < stride
        shard_live = shard_rows < stride
        np.testing.assert_array_equal(whole_live, shard_live)
        np.testing.assert_array_equal(whole_rows[whole_live], shard_rows[shard_live])
        for builder in (
            host.shard_transport_tables_in_blocks,
            host.shard_push_tables_in_rows,
            host.shard_expert_slabs,
        ):
            _assert_arrays_equal(
                builder(whole, me, e_total=e_total, ep=ep),
                builder(shard, me, e_total=e_total, ep=ep),
                shard=me,
            )
        np.testing.assert_array_equal(
            np.asarray(host.shard_token_gather(whole, me, shard_stride=stride)),
            np.asarray(host.shard_token_gather(shard, me, shard_stride=stride)),
        )
        whole_experts = host.shard_expert_slabs(whole, me, e_total=e_total, ep=ep)
        shard_experts = host.shard_expert_slabs(shard, me, e_total=e_total, ep=ep)
        whole_visit = host.expert_visit_list(whole_experts[0], g_local)
        shard_visit = host.expert_visit_list(shard_experts[0], g_local)
        _assert_arrays_equal(whole_visit, shard_visit, shard=me)
        if all_padded:
            assert int(np.asarray(whole_visit[1]).item()) == 0


def test_combine_arrivals_isolates_nan_on_unrouted_padded_tokens():
    """NaN arrival rows stay isolated to padded rows and zero out via row_routes."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _combine_arrivals

    # Use t_local=8 so combine_step_tokens(t_local) is 0 and the XLA epilogue runs
    # on CPU without invoking the Pallas TPU combine kernel.
    t_local, topk, lane_blocks, lanes, tile_m = 8, 4, 2, 128, 128
    recv_rows = 32
    arrivals = np.ones((recv_rows, lane_blocks, lanes), dtype=np.float32)
    arrival_scales = np.ones((4, tile_m), dtype=np.float32)

    # Sentinel pairs receive `pos = mirror_pos =` their rank among sentinel pairs
    # within the routing block (`[0, block)`). Poison the entire range `[0, 16)`
    # that the 4 padded tokens (`4 * topk = 16` pairs) index into.
    sentinel_ranks = np.arange(4 * topk, dtype=np.int32).reshape(4, topk)
    arrivals[: 4 * topk] = np.nan
    arrival_scales[0, : 4 * topk] = np.nan

    pos = np.full((t_local, topk), 16, dtype=np.int32)
    mirror_pos = np.full((t_local, topk), 16, dtype=np.int32)
    topk_weights = np.full((t_local, topk), 0.25, dtype=np.float32)

    # Mark rows 4..7 as padded: pos and mirror_pos span [0, 16), topk_weights=0.0.
    pos[4:] = sentinel_ranks
    mirror_pos[4:] = sentinel_ranks
    topk_weights[4:] = 0.0
    row_routes = jnp.asarray([[True]] * 4 + [[False]] * 4)

    combined = _combine_arrivals(
        jnp.asarray(arrivals),
        jnp.asarray(arrival_scales),
        jnp.asarray(pos),
        jnp.asarray(mirror_pos),
        jnp.asarray(topk_weights),
        jnp.bfloat16,
    )
    out = np.asarray(jnp.where(row_routes, combined, jnp.zeros_like(combined)))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:4].astype(np.float32), 1.0)
    np.testing.assert_array_equal(
        out[4:].astype(np.float32), np.zeros((4, lane_blocks * lanes), np.float32)
    )


@pytest.mark.parametrize("scoring_fn", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_row_routes_rejects_nan_and_neg_inf_across_all_scoring_fns(scoring_fn):
    """Both NaN-masked rows and -inf replica-padded rows must return False."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _apply_scoring_fn, _row_routes

    gate_l = jnp.asarray(
        [
            [0.5, -0.2, 1.1, 0.0],
            [np.nan, np.nan, np.nan, np.nan],
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [-np.inf, 0.0, -np.inf, -np.inf],
        ],
        dtype=jnp.float32,
    )
    scores = _apply_scoring_fn(gate_l, scoring_fn)
    routes = np.asarray(_row_routes(scores, gate_l))
    np.testing.assert_array_equal(routes, np.array([[True], [False], [False], [True]]))
