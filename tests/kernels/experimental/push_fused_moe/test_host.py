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
