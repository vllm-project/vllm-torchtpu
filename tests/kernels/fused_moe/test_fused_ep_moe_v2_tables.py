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

import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from jax.experimental.pallas import tpu as pltpu  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel  # noqa: E402


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


def _plan_counts(idx, *, ep, t_local, e_total):
    """The [ep, e_total] count table the sharded plan exchanges."""
    flat = np.asarray(idx)
    return np.stack(
        [
            np.bincount(
                flat[d * t_local : (d + 1) * t_local][
                    flat[d * t_local : (d + 1) * t_local] < e_total
                ],
                minlength=e_total,
            )
            for d in range(ep)
        ]
    ).astype(np.int32)


def _pad_half_of_each_shard(idx, *, ep, t_local, e_total):
    """Padding rows as the layer marks them: every selection set to e_total.

    The second half of each shard's rows, and all of shard 1's.
    """
    idx = np.asarray(idx).copy()
    for d in range(ep):
        start = d * t_local if d == 1 else d * t_local + t_local // 2
        idx[start : (d + 1) * t_local, :] = e_total
    return jnp.asarray(idx)


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "padded"])
@pytest.mark.parametrize("sharded", [False, True], ids=["replicated", "sharded"])
def test_slab_scatter_set_builds_the_add_tables(
    ep, t_local, g_local, topk, routing, sharded
):
    """Both slab tables come out of a scatter-set as they do of an add.

    That is what lets the chip rule (host.slab_tables_on_tensorcore) pick the
    form by the SparseCore alone. Every live row takes at most one pair and the
    slab starts at zero; the only rows the two forms can disagree on are
    dropped or past the slab's last row, where the kernel never reads. Padding
    rows (expert id e_total) go off-shard in both plans, so they stay out of
    the live rows too.
    """
    idx, kwargs = _routing_inputs(
        ep, t_local, g_local, topk, skew=routing == "skewed", seed=3
    )
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    if routing == "padded":
        idx = _pad_half_of_each_shard(idx, ep=ep, t_local=t_local, e_total=e_total)
    whole = host.build_routing_tables(idx, **kwargs)
    counts = _plan_counts(idx, ep=ep, t_local=t_local, e_total=e_total)
    # Row-scale bit patterns as the routing exchange carries them: one
    # positive f32 per token, repeated over its selections.
    rng = np.random.default_rng(4)
    tokens = ep * t_local
    scale_bits = jnp.asarray(
        np.repeat((rng.random(tokens, dtype=np.float32) + 0.5).view(np.int32), topk)
    )
    scale_size = host.act_scale_slab_rows(stride) * host.HIDDEN_LANE_BLOCK

    for me in range(ep):
        plan = (
            host.build_routing_tables_sharded(
                idx,
                jnp.int32(me),
                all_gather_rows=lambda _row: jnp.asarray(counts),
                **kwargs,
            )
            if sharded
            else whole
        )
        rows = host.local_slab_rows(plan, me, shard_stride=stride)
        gather_add, gather_set = (
            host.shard_token_gather(
                plan, me, shard_stride=stride, rows=rows, by_set=by_set
            )
            for by_set in (False, True)
        )
        np.testing.assert_array_equal(
            np.asarray(gather_set),
            np.asarray(gather_add),
            err_msg=f"token-gather table, shard {me}",
        )
        scale_add, scale_set = (
            host.slab_scatter(scale_size, rows, scale_bits, by_set=by_set)
            for by_set in (False, True)
        )
        np.testing.assert_array_equal(
            np.asarray(scale_set)[:stride],
            np.asarray(scale_add)[:stride],
            err_msg=f"activation-scale slab, shard {me}",
        )


@pytest.mark.parametrize(
    "chip,on_tensorcore",
    [
        ("TPU_7X", False),  # SparseCore 2 x 16: XLA offloads the scatter-add
        ("TPU_8T", False),  # 2 x 16
        ("TPU_V6E", False),  # 2 x 16
        ("TPU_V5P", False),  # 4 x 16: four SparseCores, all full
        ("TPU_8I", True),  # 1 x 4: the scatters stay on the TensorCore
        ("TPU_V5E", False),  # no SparseCore: the set is unmeasured, keep the add
    ],
)
def test_slab_tables_follow_the_sparse_core(chip, on_tensorcore):
    """Only a chip whose SparseCore has at most 4 subcores builds on the TensorCore."""
    info = pltpu.get_tpu_info_for_chip(getattr(pltpu.ChipVersion, chip), 1)
    assert host.slab_tables_on_tensorcore(info) is on_tensorcore


@pytest.mark.parametrize(
    "smem,tokens,fits",
    [
        (None, 131072, True),  # TPU8i's record: 1 MiB, so the full 512 KiB
        (None, 131073, False),
        (512 * 1024, 65536, True),  # half the SMEM, so half the budget
        (512 * 1024, 65537, False),
    ],
)
def test_scale_slab_budget_follows_the_chips_smem(smem, tokens, fits):
    """The lookup's scale table takes at most half of the chip's SMEM."""
    if smem is None:
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_8I, 1)
    else:
        info = types.SimpleNamespace(smem_capacity_bytes=smem)
    assert kernel.scale_slab_fits(tokens, info) is fits


@pytest.mark.parametrize(
    "ep,t_local,g_local,topk",
    GEOMETRIES + [pytest.param(32, 48, 16, 10, id="qwen38_ep32_g16_t48")],
)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "padded"])
@pytest.mark.parametrize("seed", [0, 1])
def test_sorted_token_gather_is_the_scattered_one(
    ep, t_local, g_local, topk, routing, seed
):
    """The sort build gives the scatter's table, from either plan builder.

    Skewed routing is the case that matters most: most runs are empty or odd,
    so the padding elements the sort carries are live on some runs and not on
    others. Padded routing puts expert id e_total on whole rows and on all of
    one shard, which the sort has to key off-shard as the plans do.
    """
    idx, kwargs = _routing_inputs(
        ep, t_local, g_local, topk, skew=routing == "skewed", seed=seed
    )
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    if routing == "padded":
        idx = _pad_half_of_each_shard(idx, ep=ep, t_local=t_local, e_total=e_total)
    assert host.token_gather_sort_fits(ep * t_local, topk, e_total=e_total, ep=ep)
    whole = host.build_routing_tables(idx, **kwargs)
    counts = jnp.asarray(_plan_counts(idx, ep=ep, t_local=t_local, e_total=e_total))
    for me in range(ep):
        shard = host.build_routing_tables_sharded(
            idx, jnp.int32(me), all_gather_rows=lambda _row: counts, **kwargs
        )
        for plan in (whole, shard):
            np.testing.assert_array_equal(
                np.asarray(
                    host.shard_token_gather_sorted(
                        idx,
                        plan,
                        jnp.int32(me),
                        e_total=e_total,
                        ep=ep,
                        shard_stride=stride,
                    )
                ),
                np.asarray(host.shard_token_gather(plan, me, shard_stride=stride)),
                err_msg=f"shard {me}",
            )


def _sender_payloads(idx, scale_bits, *, ep, t_local, topk, block):
    """Every shard's expert-sorted payload as the run-copy exchange gathers it.

    Returns the token numbers and the scale bits, [ep, pair_w] each with every
    source padded to the lane-block-aligned section width, and that width.
    """
    pair_w = host.align_up(t_local * topk, host.HIDDEN_LANE_BLOCK)
    tokens, scales = [], []
    for s in range(ep):
        rows = slice(s * t_local, (s + 1) * t_local)
        t_s, c_s = host.sender_sorted_pairs(
            jnp.asarray(idx)[rows].reshape(-1, block),
            jnp.int32(s),
            t_local=t_local,
            topk=topk,
            scale_bits=jnp.asarray(scale_bits)[rows],
        )
        tokens.append(np.pad(np.asarray(t_s), (0, pair_w - t_s.size)))
        scales.append(np.pad(np.asarray(c_s), (0, pair_w - c_s.size)))
    return np.stack(tokens), np.stack(scales), pair_w


@pytest.mark.parametrize(
    "ep,t_local,g_local,topk",
    GEOMETRIES + [pytest.param(32, 48, 16, 10, id="qwen38_ep32_g16_t48")],
)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "padded"])
def test_slab_runs_copy_out_the_scattered_tables(ep, t_local, g_local, topk, routing):
    """Copying runs of the expert-sorted payloads rebuilds both slab tables.

    The host half of the run-copy form: sender_sorted_pairs on every source,
    then slab_run_copy_runs' offsets for each shard, copied here in numpy (the
    kernel that does it on the chip is checked against this in its own test).
    The result must be the scatter's token-gather table, padding and clip
    included, and the scatter's scale slab over the whole slab: zero on the
    ROWBLK padding of every run, as the scatter leaves it.
    """
    idx, kwargs = _routing_inputs(
        ep, t_local, g_local, topk, skew=routing == "skewed", seed=7
    )
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    if routing == "padded":
        idx = _pad_half_of_each_shard(idx, ep=ep, t_local=t_local, e_total=e_total)
    T = ep * t_local
    rng = np.random.default_rng(8)
    scale_bits = (rng.random(T, dtype=np.float32) + 0.5).view(np.int32)
    tokens, scales, pair_w = _sender_payloads(
        idx,
        scale_bits,
        ep=ep,
        t_local=t_local,
        topk=topk,
        block=kwargs["block"],
    )
    counts = jnp.asarray(_plan_counts(idx, ep=ep, t_local=t_local, e_total=e_total))
    view_rows = host.act_scale_slab_rows(stride) * host.HIDDEN_LANE_BLOCK
    lo, hi = host.gather_clamp_bounds(T)
    for me in range(ep):
        plan = host.build_routing_tables_sharded(
            idx, jnp.int32(me), all_gather_rows=lambda _row: counts, **kwargs
        )
        rows = host.local_slab_rows(plan, me, shard_stride=stride)
        src, dst, length = (
            np.asarray(a)
            for a in host.slab_run_copy_runs(
                plan, jnp.int32(me), e_total=e_total, ep=ep, src_stride=pair_w
            )
        )
        assert length.size == g_local * ep
        got_tokens = np.zeros(view_rows, np.int32)
        got_scales = np.zeros(view_rows, np.int32)
        for a, b, n in zip(src, dst, length):
            got_tokens[b : b + n] = tokens.reshape(-1)[a : a + n]
            got_scales[b : b + n] = scales.reshape(-1)[a : a + n]
        np.testing.assert_array_equal(
            np.clip(got_tokens[:stride], lo, hi),
            np.asarray(
                host.shard_token_gather(plan, me, shard_stride=stride, rows=rows)
            ),
            err_msg=f"token-gather table, shard {me}",
        )
        scattered = host.slab_scatter(
            view_rows, rows, jnp.asarray(np.repeat(scale_bits, topk)), by_set=False
        )
        np.testing.assert_array_equal(
            got_scales[:stride],
            np.asarray(scattered)[:stride],
            err_msg=f"activation-scale slab, shard {me}",
        )


def test_token_gather_sort_keys_fit_the_served_shapes():
    """Qwen3.8's EP32 prefill step is far inside the int32 key range."""
    assert host.token_gather_sort_fits(16384, 10, e_total=512, ep=32)
    assert not host.token_gather_sort_fits(2**24, 10, e_total=512, ep=32)


def test_token_gather_sort_keys_stop_at_the_int32_bound():
    """The bound is exact: the off-shard key, e_total * run_span, stays < 2**31."""
    e_total, topk, ep = 2, 1, 1
    at_bound = 2**30 - (host.ROWBLK - 1)  # e_total * run_span == 2**31
    assert not host.token_gather_sort_fits(at_bound, topk, e_total=e_total, ep=ep)
    assert host.token_gather_sort_fits(at_bound - 1, topk, e_total=e_total, ep=ep)


@pytest.mark.parametrize(
    "tokens,pays", [(1536, False), (4096, False), (8192, True), (16384, True)]
)
def test_slab_run_copy_pays_at_qwen38s_measured_steps(tokens, pays):
    """At Qwen3.8's shape (top-10 of 512) the gate picks the steps measured to gain."""
    assert kernel.slab_run_copy_pays(tokens, 10, e_total=512) is pays


@pytest.mark.parametrize(
    "t_local,vmem,fits",
    [
        (256, None, True),  # Qwen3.8's 8,192-token step on TPU8i
        (512, None, True),  # and its 16,384-token step
        (1 << 16, None, False),  # ~2M tokens: past TPU8i's VMEM budget
        (512, 1 << 20, False),  # the 16,384-token step on a 1 MiB VMEM
    ],
)
def test_slab_run_copy_fits_its_vmem_budget(t_local, vmem, fits):
    """The run copy is admitted only where its streams and tables fit VMEM."""
    ep, topk, e_total, capacity = 32, 10, 512, 128
    stride = host.ragged_stride_bound(ep * t_local, topk, e_total, capacity)
    if vmem is None:
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_8I, 1)
    else:
        info = types.SimpleNamespace(vmem_capacity_bytes=vmem)
    assert host.slab_run_copy_fits(t_local, topk, ep, stride, info) is fits


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


@pytest.mark.parametrize("tile_m,rows", [(64, 1), (128, 1), (256, 2), (512, 4)])
def test_mirror_rows_per_tile_keeps_128_lanes_above_a_lane_block(tile_m, rows):
    """A tile's mirror sublane above 128 lanes is stored as rows of 128."""
    assert host.mirror_rows_per_tile(tile_m) == rows


def test_mirror_rows_per_tile_refuses_a_partial_lane_row():
    with pytest.raises(ValueError):
        host.mirror_rows_per_tile(192)


@pytest.mark.parametrize(
    "smem,t_local,topk,fits",
    [
        (None, 8192, 10, True),  # 640 KiB <= 992 KiB on TPU v7x (1 MiB SMEM)
        (None, 12288, 10, True),  # 960 KiB <= 992 KiB
        (None, 12697, 10, True),  # 1,015,760 B <= 1,015,808 B upper boundary
        (None, 12698, 10, False),  # 1,015,840 B > 1,015,808 B
        (None, 16384, 8, False),  # 1,024 KiB > 992 KiB (no headroom left)
        (None, 16384, 10, False),  # 1,280 KiB > 1 MiB (Qwen3.5 1P1D prefill)
        (512 * 1024, 4096, 10, True),  # 320 KiB <= 480 KiB on 512 KiB SMEM
        (512 * 1024, 8192, 10, False),  # 640 KiB > 480 KiB on 512 KiB SMEM
        (2 * 1024 * 1024, 16384, 10, True),  # 1,280 KiB <= 2,016 KiB on 2 MiB
    ],
)
def test_combine_tables_budget_follows_the_chips_smem(smem, t_local, topk, fits):
    """Combine scalar-prefetch tables fit the chip's SMEM less the headroom."""
    if smem is None:
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    else:
        info = types.SimpleNamespace(smem_capacity_bytes=smem)
    assert kernel.combine_tables_fit(t_local, topk, info) is fits
    expected_use = bool(kernel.combine_step_tokens(t_local)) and fits
    assert kernel.combine_uses_kernel(t_local, topk, info) is expected_use


def test_combine_uses_kernel_skips_device_lookup_below_min_tokens(monkeypatch):
    """Below COMBINE_MIN_TOKENS, combine_uses_kernel short-circuits before TPU info."""
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: (_ for _ in ()).throw(AssertionError("should not query TPU info")),
    )
    assert kernel.combine_uses_kernel(8, 10) is False


def test_combine_arrivals_dispatches_on_smem_budget(monkeypatch):
    """_combine_arrivals falls back to XLA when pos+coef exceed SMEM, else kernel."""
    from vllm_torchtpu.kernels.fused_moe.v2 import layer

    t_local = kernel.COMBINE_MIN_TOKENS
    topk, lane_blocks, lanes, tile_m = 10, 2, 128, 128
    hidden = lane_blocks * lanes
    assert kernel.combine_step_tokens(t_local) == 64
    assert kernel.combine_prefetch_bytes(t_local, topk) == 20480

    rng = np.random.default_rng(42)
    recv_rows = 64
    arrivals_np = rng.uniform(-1.0, 1.0, (recv_rows, lane_blocks, lanes)).astype(
        np.float32
    )
    scales_np = rng.uniform(0.5, 1.5, (2, tile_m)).astype(np.float32)
    pos_np = rng.integers(0, recv_rows, (t_local, topk), dtype=np.int32)
    mirror_pos_np = rng.integers(0, 2 * tile_m, (t_local, topk), dtype=np.int32)
    weights_np = rng.uniform(0.05, 0.25, (t_local, topk)).astype(np.float32)

    row_scales = scales_np[mirror_pos_np // tile_m, mirror_pos_np % tile_m]
    coef_np = row_scales * weights_np
    expected = (
        (arrivals_np[pos_np] * coef_np[:, :, None, None])
        .sum(axis=1)
        .reshape(t_local, hidden)
    )

    args = (
        jnp.asarray(arrivals_np),
        jnp.asarray(scales_np),
        jnp.asarray(pos_np),
        jnp.asarray(mirror_pos_np),
        jnp.asarray(weights_np),
        jnp.bfloat16,
    )

    # Small SMEM (48 KiB -> 16 KiB budget < 20 KiB tables): must skip kernel
    # and match the weighted sum via the XLA gather epilogue.
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(smem_capacity_bytes=48 * 1024),
    )

    def _fail_if_built(**kwargs):
        raise AssertionError(
            f"build_combine_kernel should not be called when over SMEM: {kwargs}"
        )

    monkeypatch.setattr(layer, "build_combine_kernel", _fail_if_built)
    got = layer._combine_arrivals(*args)
    assert got.shape == (t_local, hidden)
    assert got.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float32), expected, rtol=1e-2, atol=1e-2
    )

    # Real TPU v7x SMEM (1 MiB -> 992 KiB budget >= 20 KiB tables): must select
    # build_combine_kernel with the exact shape and dtype arguments.
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    recorded = {}
    sentinel = jnp.full((t_local, hidden), 7.0, dtype=jnp.bfloat16)

    def _fake_builder(**kwargs):
        recorded["kwargs"] = kwargs
        return lambda arrivals, coef, rows: sentinel

    monkeypatch.setattr(layer, "build_combine_kernel", _fake_builder)
    got_kernel = layer._combine_arrivals(*args)
    assert got_kernel is sentinel
    assert recorded["kwargs"] == {
        "t_local": t_local,
        "topk": topk,
        "hidden": hidden,
        "lane_blocks": lane_blocks,
        "wire_dtype": jnp.float32,
        "out_dtype": jnp.bfloat16,
    }


def test_build_combine_kernel_rejects_oversized_smem_and_unaligned_tokens(monkeypatch):
    """Direct calls to _build_combine_kernel raise ValueError instead of OOMing."""
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    with pytest.raises(ValueError, match=r"1310720 bytes.*1015808-byte SMEM budget"):
        kernel._build_combine_kernel(
            t_local=16384,
            topk=10,
            hidden=256,
            lane_blocks=2,
            wire_dtype=jnp.bfloat16,
            out_dtype=jnp.bfloat16,
        )
    with pytest.raises(ValueError, match=r"t_local=8"):
        kernel._build_combine_kernel(
            t_local=8,
            topk=10,
            hidden=256,
            lane_blocks=2,
            wire_dtype=jnp.bfloat16,
            out_dtype=jnp.bfloat16,
        )
