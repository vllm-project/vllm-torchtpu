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
"""Host tables of the push-dispatch fused EP MoE, checked on CPU.

The push build moves rows by the segment table and reads them back by v2's
routing plan, so the two must agree on every row: a segment's destination
row is where v2's slab places that (expert, source) run, and its source row
is where the local sort put the run. The plan builder itself must also stay
value-identical to v2's.
"""

import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.experimental.push_fused_moe import (  # noqa: E402
    host as push_host,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_backend():
    """Keep every array in this file off the accelerator."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


# Decode-sized, prefill-sized (the windowed rank pass) and a small geometry.
GEOMETRIES = [
    pytest.param(8, 64, 64, 10, id="decode_t64_ep8_g64"),
    pytest.param(8, 512, 64, 10, id="prefill_t512_ep8_g64"),
    pytest.param(4, 128, 32, 8, id="ep4_g32_topk8"),
]


def _routing(ep, t_local, g_local, topk, *, skew, seed):
    """Global top-k ids [T, topk] (some rows routed nowhere) and plan kwargs."""
    e_total = g_local * ep
    T = t_local * ep
    rng = np.random.default_rng(seed)
    if skew:
        # Low concurrency: few experts selected at all, most runs empty. A
        # token still selects an expert at most once, as top-k does.
        hot = rng.choice(e_total, size=max(topk, e_total // 12), replace=False)
        idx = np.stack([rng.choice(hot, size=topk, replace=False) for _ in range(T)])
    else:
        idx = np.stack([rng.permutation(e_total)[:topk] for _ in range(T)])
    idx[rng.random(T) < 0.05] = e_total  # rows that route nowhere
    capacity = 128
    kwargs = dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=host.routing_block(t_local, topk),
        tile_m=capacity,
        shard_stride=host.ragged_stride_bound(T, topk, e_total, capacity),
    )
    return idx.astype(np.int32), kwargs


def _counts(idx, ep, t_local, e_total):
    """rows_by_dest [ep, e_total]: each source shard's pairs per expert."""
    return np.stack(
        [
            np.bincount(
                idx[s * t_local : (s + 1) * t_local].reshape(-1),
                minlength=e_total + 1,
            )[:e_total]
            for s in range(ep)
        ]
    ).astype(np.int32)


def _gathered(idx, me, kw):
    ep, t_local, block, e_total = kw["ep"], kw["t_local"], kw["block"], kw["e_total"]
    topk = idx.shape[1]
    blocks = jnp.asarray(idx.reshape(-1, block))
    per_dest = t_local * topk // block
    local = blocks[me * per_dest : (me + 1) * per_dest]
    return host.GatheredPairs(
        expert_blocks=blocks,
        local_blocks=local,
        block_hist=host.pair_block_hist(local, e_total),
        rows_by_dest=jnp.asarray(_counts(idx, ep, t_local, e_total)),
        n_tokens=ep * t_local,
        topk=topk,
    )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_sharded_plan_matches_v2(ep, t_local, g_local, topk, skew):
    idx, kw = _routing(ep, t_local, g_local, topk, skew=skew, seed=0)
    for me in (0, ep - 1):
        gathered = _gathered(idx, me, kw)
        want = host.build_routing_tables_sharded(
            None, jnp.int32(me), gathered=gathered, **kw
        )
        got = push_host.build_routing_tables_sharded(
            jnp.int32(me),
            gathered,
            push_host.run_tables(gathered.rows_by_dest),
            **kw,
        )
        for field in want._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(got, field)),
                np.asarray(getattr(want, field)),
                err_msg=f"shard {me}, {field}",
            )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
def test_local_sort_activations(ep, t_local, g_local, topk):
    idx, kw = _routing(ep, t_local, g_local, topk, skew=True, seed=1)
    e_total = kw["e_total"]
    mine = idx[:t_local]
    counts = np.bincount(mine.reshape(-1), minlength=e_total + 1)[:e_total]
    rows = np.arange(t_local, dtype=np.float32)[:, None, None] * np.ones((1, 2, 128))
    q_sorted, seg_start = push_host.local_sort_activations(
        jnp.asarray(mine.reshape(-1)),
        jnp.asarray(counts),
        jnp.asarray(rows),
        e_total=e_total,
        topk=topk,
    )
    want_tokens = [t for e in range(e_total) for t in range(t_local) if e in mine[t]]
    n = len(want_tokens)
    np.testing.assert_array_equal(np.asarray(q_sorted)[:n, 0, 0], want_tokens)
    np.testing.assert_array_equal(np.asarray(seg_start), np.cumsum(counts) - counts)


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_segments_land_on_the_plans_slab_rows(ep, t_local, g_local, topk, skew):
    idx, kw = _routing(ep, t_local, g_local, topk, skew=skew, seed=2)
    e_total = kw["e_total"]
    rows_by_dest = _counts(idx, ep, t_local, e_total)
    runs = push_host.run_tables(jnp.asarray(rows_by_dest))
    plan = host.build_routing_tables(jnp.asarray(idx), **kw)
    expert_base = np.asarray(plan.expert_base)
    run_start = np.asarray(plan.run_start_aligned)
    visit = np.stack(
        [
            np.asarray(
                host.expert_visit_list(
                    np.asarray(runs[3]).reshape(ep, g_local)[d], g_local
                )[0]
            )
            for d in range(ep)
        ]
    )
    for me in (0, ep - 1):
        counts = rows_by_dest[me]
        seg_start = np.cumsum(counts) - counts
        seg, meta, visit_all = push_host.push_dispatch_segments(
            runs[3],
            runs[2],
            jnp.asarray(rows_by_dest),
            jnp.int32(me),
            jnp.asarray(seg_start),
            e_total=e_total,
            ep=ep,
        )
        seg, meta = np.asarray(seg), np.asarray(meta)
        np.testing.assert_array_equal(np.asarray(visit_all), visit)
        S = ep * g_local
        sent = remote = 0
        for v in range(g_local):
            for d in range(ep):
                p = v * ep + d
                a = int(seg[p])
                assert a >> 24 == d
                n = a & 65535
                e = visit[d, v]
                if e < 0 or n == 0:
                    continue
                ge = d * g_local + e
                assert (a >> 16) & 255 == e
                assert n == counts[ge]
                assert seg[S + p] == seg_start[ge]
                # The run's first row in d's own slab, as v2's plan has it.
                assert seg[2 * S + p] == (
                    expert_base[ge] - d * kw["shard_stride"] + run_start[ge, me]
                )
                sent += n
                remote += n * (d != me)
        recv = rows_by_dest[:, me * g_local : (me + 1) * g_local].sum(axis=0)
        np.testing.assert_array_equal(meta[g_local + 1 : 2 * g_local + 1], recv)
        assert meta[2 * g_local + 1] == remote
        assert meta[2 * g_local + 2] == sent == counts.sum()


def test_to_slots_splits_rows_over_replicas():
    from vllm_torchtpu.kernels.experimental.push_fused_moe.layer import _to_slots

    # Expert 0 has two copies (slots 3, 7); experts 1 and 2 one each.
    table = np.array([[3, 7], [5, -1], [0, -1]], np.int32)
    ids = np.array([[0, 1], [0, 2], [2, 0]], np.int32)
    got = np.asarray(_to_slots(jnp.asarray(ids), jnp.asarray(table), jnp.int32(4)))
    # Row r takes replica (4 + r) mod (copies of its expert).
    np.testing.assert_array_equal(got, [[3, 5], [7, 0], [0, 3]])


@pytest.mark.parametrize("replicas", [1, 2, 10])
def test_to_slots_forms_agree(replicas):
    """Both forms return exactly the table entry the definition names, at
    the served sizes (288 rows, top-6, 384 experts over 400 slots), for
    every row base parity and an out-of-range id."""
    from vllm_torchtpu.kernels.experimental.push_fused_moe.layer import _to_slots

    rng = np.random.default_rng(replicas)
    experts, rows, topk = 384, 288, 6
    n_rep = np.ones(experts, np.int32)
    n_rep[rng.choice(experts, 16, replace=False)] = rng.integers(1, replicas + 1, 16)
    perm = rng.permutation(experts + 16 * replicas)
    table = np.full((experts, replicas), -1, np.int32)
    k = 0
    for e in range(experts):
        table[e, : n_rep[e]] = perm[k : k + n_rep[e]]
        k += n_rep[e]
    ids = np.stack([rng.choice(experts, topk, replace=False) for _ in range(rows)])
    ids[3, 2] = experts  # a row's no-route sentinel
    for row_base, onehot in itertools.product((0, 5, 4 * rows), (False, True)):
        got = np.asarray(
            _to_slots(
                jnp.asarray(ids, jnp.int32),
                jnp.asarray(table),
                row_base,
                onehot=onehot,
            )
        )
        e = np.clip(ids, 0, experts - 1)
        token = row_base + np.arange(rows)[:, None]
        want = table[e, token % n_rep[e]]
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "t_local,e_total,topk",
    [
        pytest.param(64, 32, 4, id="t64_e32_k4"),
        pytest.param(288, 384, 6, id="t288_e384_k6"),
        pytest.param(513, 416, 6, id="t513_e416_k6"),
        pytest.param(1024, 384, 8, id="t1024_e384_k8"),
        pytest.param(2048, 416, 6, id="t2048_e416_k6"),
    ],
)
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_local_sort_paths_agree(monkeypatch, t_local, e_total, topk, skew):
    """The prefill argsort path and the decode count path give the same
    q_sorted (every row, padding included) and seg_start."""
    rng = np.random.default_rng(t_local + e_total + topk + skew)
    if skew:
        hot = rng.choice(e_total, size=max(topk, e_total // 12), replace=False)
        mine = np.stack(
            [rng.choice(hot, size=topk, replace=False) for _ in range(t_local)]
        )
    else:
        mine = np.stack([rng.permutation(e_total)[:topk] for _ in range(t_local)])
    mine[rng.random(t_local) < 0.05] = e_total  # rows that route nowhere
    counts = np.bincount(mine.reshape(-1), minlength=e_total + 1)[:e_total]
    rows = np.arange(t_local, dtype=np.float32)[:, None, None] * np.ones((1, 1, 128))
    args = (jnp.asarray(mine.reshape(-1)), jnp.asarray(counts), jnp.asarray(rows))
    results = []
    # count path, its bf16-count form, then the argsort path
    for threshold, bf16 in ((t_local, False), (t_local, True), (t_local - 1, False)):
        monkeypatch.setattr(push_host, "ARGSORT_MIN_TOKENS", threshold)
        results.append(
            push_host.local_sort_activations(
                *args, e_total=e_total, topk=topk, bf16_counts=bf16
            )
        )
    (q_count, s_count), (q_bf16, s_bf16), (q_sort, s_sort) = results
    np.testing.assert_array_equal(np.asarray(q_sort), np.asarray(q_count))
    np.testing.assert_array_equal(np.asarray(s_sort), np.asarray(s_count))
    np.testing.assert_array_equal(np.asarray(q_bf16), np.asarray(q_count))
    np.testing.assert_array_equal(np.asarray(s_bf16), np.asarray(s_count))
    want = [t for e in range(e_total) for t in range(t_local) if e in mine[t]]
    np.testing.assert_array_equal(np.asarray(q_sort)[: len(want), 0, 0], want)


@pytest.mark.parametrize("live", [0, 1, 2, 7, 24, 25])
def test_interleave_visits_alternates_heaviest_and_lightest(live):
    g_local = 25
    rows = np.zeros(g_local, np.int32)
    rows[np.random.default_rng(live).permutation(g_local)[:live]] = (
        np.arange(live, 0, -1) * 8
    )
    visit, n_visit = host.expert_visit_list(jnp.asarray(rows), g_local)
    got = np.asarray(push_host.interleave_visits(visit, n_visit))
    visit = np.asarray(visit)
    want = [visit[(q // 2) if q % 2 == 0 else live - 1 - q // 2] for q in range(live)]
    np.testing.assert_array_equal(got[:live], want)
    np.testing.assert_array_equal(got[live:], visit[live:])
    assert sorted(got[:live]) == sorted(visit[:live])


@pytest.mark.parametrize("n,repeats", [(4608, 6), (3840, 3), (2304, 6), (200, 6)])
def test_repeat_bits_equals_jnp_repeat(n, repeats):
    """Every i32 bit pattern (negatives, NaN and inf f32 patterns among them)
    comes back exactly, at the served scale-slab sizes and a ragged one."""
    from vllm_torchtpu.kernels.experimental.push_fused_moe.layer import _repeat_bits

    rng = np.random.default_rng(n)
    bits = rng.integers(-(2**31), 2**31, n, dtype=np.int64).astype(np.int32)
    bits[:4] = np.array([np.nan, np.inf, -np.inf, -0.0], np.float32).view(np.int32)
    got = np.asarray(_repeat_bits(jnp.asarray(bits), repeats))
    np.testing.assert_array_equal(got, np.repeat(bits, repeats))


@pytest.mark.parametrize("n_pre", [1, 2, 3, 25])
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_dispatch_tables_match_the_decoded_segments(n_pre, skew):
    """push_dispatch_tables' extra outputs equal what the layer decodes out
    of push_dispatch_segments' tables, at the DP16 geometry (ep 16, 25 slots,
    288 rows, top-6) for every rank parity."""
    ep, t_local, g_local, topk = 16, 288, 25, 6
    idx, kw = _routing(ep, t_local, g_local, topk, skew=skew, seed=5)
    e_total = kw["e_total"]
    n_pre = min(n_pre, g_local)
    rows_by_dest = jnp.asarray(_counts(idx, ep, t_local, e_total))
    runs = push_host.run_tables(rows_by_dest)
    for me in (0, 1, ep - 1):
        counts = np.asarray(rows_by_dest)[me]
        seg_start = jnp.asarray(np.cumsum(counts) - counts)
        args = (runs[3], runs[2], rows_by_dest, jnp.int32(me), seg_start)
        seg0, meta0, visit0 = (
            np.asarray(a)
            for a in push_host.push_dispatch_segments(*args, e_total=e_total, ep=ep)
        )
        seg, meta, visit_all, my_visit, counts_sc = (
            np.asarray(a)
            for a in push_host.push_dispatch_tables(
                *args, e_total=e_total, ep=ep, n_pre=n_pre
            )
        )
        np.testing.assert_array_equal(seg, seg0)
        np.testing.assert_array_equal(visit_all, visit0)
        np.testing.assert_array_equal(my_visit, visit0[me])
        S = ep * g_local
        seg_n = (seg0[:S] & 65535).reshape(g_local, ep)
        remote = seg_n * (np.arange(ep)[None, :] != me)
        want_meta = meta0.copy()
        want_meta[2 * g_local + 1] = remote[n_pre:].sum()
        np.testing.assert_array_equal(meta, want_meta)
        recv = meta0[g_local + 1 : 2 * g_local + 1]
        rows_at_v = recv[visit0[me][:n_pre]]
        want_sc = [
            rows_at_v[0::2].sum(),
            rows_at_v[1::2].sum(),
            remote[0:n_pre:2].sum(),
            remote[1:n_pre:2].sum(),
            0,
            0,
            0,
            0,
        ]
        np.testing.assert_array_equal(counts_sc, want_sc)


@pytest.mark.parametrize("n_pre", [1, 2, 25])
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_table_kernel_matches_the_xla_tables(n_pre, skew):
    """plan_kernel.build_push_tables (interpret mode) gives exactly the XLA
    dispatch and shard tables at the DP16 geometry, for several ranks."""
    from vllm_torchtpu.kernels.experimental.push_fused_moe.plan_kernel import (
        build_push_tables,
    )

    ep, t_local, g_local, topk = 16, 288, 25, 6
    idx, kw = _routing(ep, t_local, g_local, topk, skew=skew, seed=11)
    e_total = kw["e_total"]
    rows_by_dest = jnp.asarray(_counts(idx, ep, t_local, e_total))
    runs = push_host.run_tables(rows_by_dest)
    plan = host.build_routing_tables(jnp.asarray(idx), **kw)
    for me in (0, 3, ep - 1):
        counts = np.asarray(rows_by_dest)[me]
        seg_start = jnp.asarray(np.cumsum(counts) - counts)
        want_seg, want_meta, _, want_visit, want_sc = push_host.push_dispatch_tables(
            runs[3], runs[2], rows_by_dest, jnp.int32(me), seg_start,
            e_total=e_total, ep=ep, n_pre=n_pre,
        )  # fmt: skip
        blocks = host.shard_transport_tables_in_blocks(plan, me, e_total=e_total, ep=ep)
        rows_t = host.shard_push_tables_in_rows(plan, me, e_total=e_total, ep=ep)
        want_er, want_sb = host.shard_expert_slabs(plan, me, e_total=e_total, ep=ep)
        want_counts = host.shard_count_vector(plan, want_er, me, e_total=e_total, ep=ep)
        got = build_push_tables(
            rows_by_dest, seg_start, jnp.int32(me), ep=ep, n_pre=n_pre,
            tile_m=kw["tile_m"], interpret=True,
        )  # fmt: skip
        seg, meta, visit, sc, kt, er, sb, cnt, _ = (
            jax.tree.map(np.asarray, x) for x in got
        )
        np.testing.assert_array_equal(seg, np.asarray(want_seg))
        np.testing.assert_array_equal(meta, np.asarray(want_meta))
        np.testing.assert_array_equal(visit, np.asarray(want_visit))
        np.testing.assert_array_equal(sc, np.asarray(want_sc))
        for k, want in enumerate((*blocks[:3], blocks[5], rows_t[0])):
            np.testing.assert_array_equal(kt[k], np.asarray(want), err_msg=str(k))
        np.testing.assert_array_equal(er, np.asarray(want_er))
        np.testing.assert_array_equal(sb, np.asarray(want_sb))
        np.testing.assert_array_equal(cnt, np.asarray(want_counts))


@pytest.mark.parametrize("n_pre", [1, 2, 3])
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_light_prepush_only_reorders_the_visits(n_pre, skew):
    """With light_prepush the table kernel visits the n_pre lightest experts
    first (lightest at 0), then the rest most rows first; the shard tables
    the expert kernel reads per expert are unchanged, and the pre-push
    counts follow the new first positions."""
    from vllm_torchtpu.kernels.experimental.push_fused_moe.plan_kernel import (
        build_push_tables,
    )

    ep, t_local, g_local, topk = 16, 288, 25, 6
    idx, kw = _routing(ep, t_local, g_local, topk, skew=skew, seed=13)
    e_total = kw["e_total"]
    rows_by_dest = jnp.asarray(_counts(idx, ep, t_local, e_total))
    for me in (0, 3, ep - 1):
        counts = np.asarray(rows_by_dest)[me]
        seg_start = jnp.asarray(np.cumsum(counts) - counts)
        base, light = (
            [
                jax.tree.map(np.asarray, x)
                for x in build_push_tables(
                    rows_by_dest, seg_start, jnp.int32(me), ep=ep, n_pre=n_pre,
                    tile_m=kw["tile_m"], interpret=True, light_prepush=lp,
                )  # fmt: skip
            ]
            for lp in (False, True)
        )
        for k in (4, 5, 6, 7, 8):
            jax.tree.map(np.testing.assert_array_equal, light[k], base[k])
        recv = np.asarray(rows_by_dest)[:, me * g_local : (me + 1) * g_local].sum(0)
        n_live = int((recv > 0).sum())
        lp = min(n_pre, n_live)
        order = list(base[2][:n_live])
        want = order[n_live - lp :][::-1] + order[: n_live - lp]
        np.testing.assert_array_equal(light[2][:n_live], want)
        assert light[3][0] + light[3][1] == recv[want[:lp]].sum()


@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "skewed"])
def test_plan_from_kernel_tables_matches_the_sharded_plan(skew):
    """routing_pairs_from_tables, fed the table kernel's plan tables
    (interpret mode), gives the sharded plan's pos, mirror_pos and
    local_slab_rows exactly, at the DP16 geometry."""
    from vllm_torchtpu.kernels.experimental.push_fused_moe.plan_kernel import (
        build_push_tables,
    )

    ep, t_local, g_local, topk = 16, 288, 25, 6
    idx, kw = _routing(ep, t_local, g_local, topk, skew=skew, seed=13)
    for me in (0, 7, ep - 1):
        gathered = _gathered(idx, me, kw)
        rows_by_dest = gathered.rows_by_dest
        want = push_host.build_routing_tables_sharded(
            jnp.int32(me), gathered, push_host.run_tables(rows_by_dest), **kw
        )
        want_slab = host.local_slab_rows(want, me, shard_stride=kw["shard_stride"])
        counts = np.asarray(rows_by_dest)[me]
        seg_start = jnp.asarray(np.cumsum(counts) - counts)
        tables = build_push_tables(
            rows_by_dest, seg_start, jnp.int32(me), ep=ep, n_pre=2,
            tile_m=kw["tile_m"], interpret=True,
        )[8]  # fmt: skip
        pos, mirror_pos, slab_row = push_host.routing_pairs_from_tables(
            jnp.int32(me),
            gathered,
            *tables,
            e_total=kw["e_total"],
            ep=ep,
            t_local=t_local,
            block=kw["block"],
            shard_stride=kw["shard_stride"],
        )
        np.testing.assert_array_equal(np.asarray(pos), np.asarray(want.pos))
        np.testing.assert_array_equal(
            np.asarray(mirror_pos), np.asarray(want.mirror_pos)
        )
        np.testing.assert_array_equal(np.asarray(slab_row), np.asarray(want_slab))


@pytest.mark.parametrize("count_at", [2176, 2000, 300])
def test_relane_counts_matches_the_reshape(count_at):
    """The exchange-tables kernel's re-laning of the exchanged count words
    (interpret mode) equals reshaping them to [s, d, g] at ep 16, g 25, for
    lane-aligned and unaligned count offsets."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.experimental.push_fused_moe.plan_kernel import (
        LANES,
        _relane_counts,
    )

    ep, g = 16, 25
    r_lo = count_at // LANES
    n_cnt = (count_at + ep * g - 1) // LANES - r_lo + 1
    rng = np.random.default_rng(count_at)
    words = rng.integers(0, 1000, (ep, (r_lo + n_cnt) * LANES)).astype(np.int32)
    cnt = words.reshape(ep, r_lo + n_cnt, LANES)[:, r_lo:]

    def kernel(c_ref, o_ref):
        _relane_counts(c_ref[...], o_ref, count_at=count_at - r_lo * LANES, g=g)

    got = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((ep, ep, LANES), jnp.int32),
        interpret=True,
    )(jnp.asarray(cnt))
    want = words[:, count_at : count_at + ep * g].reshape(ep, ep, g)
    np.testing.assert_array_equal(np.asarray(got)[:, :, :g], want)
    np.testing.assert_array_equal(np.asarray(got)[:, :, g:], 0)
