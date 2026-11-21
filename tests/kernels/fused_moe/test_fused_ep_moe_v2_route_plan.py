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
"""The route-plan kernel returns the XLA path's tables, bit for bit.

Every table the fused kernel and the combine read is compared shard by shard
with what layer.compute_local builds from the XLA routing all-gather on the
sharded plan: the transport/push tables, the token-gather table, the expert
rows and slab bases, the activation scale slab, the visit list, the count table
and the arrival/mirror positions.

On a TPU host this runs the kernel natively over the first `ep` devices (the
`multichip` marker). On CPU it runs it in Pallas' TPU interpret mode, which
emulates the remote DMAs and semaphores; that needs at least `ep` CPU devices,
e.g. XLA_FLAGS=--xla_force_host_platform_device_count=32.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax import lax  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v2 import host, layer, route_plan  # noqa: E402

# (ep, g_local, t_local, topk): Qwen3.8's 16-chip and 4-chip decode layouts,
# and a smaller one with an expert count that is not a lane-block multiple.
GEOMETRIES = [
    pytest.param(32, 24, 16, 10, id="ep32_g24_t16"),
    pytest.param(8, 24, 16, 10, id="ep8_g24_t16"),
    # The 12-token MTP draft bucket: 3 rows per replica, aligned to 4.
    pytest.param(8, 24, 4, 10, id="ep8_g24_t4"),
    pytest.param(32, 24, 4, 10, id="ep32_g24_t4"),
    pytest.param(4, 16, 8, 8, id="ep4_g16_t8"),
]
ROUTINGS = ["uniform", "skewed", "padded", "all_mine"]


def _selections(ep, g_local, t_local, topk, routing, seed):
    """[ep * t_local, topk] distinct expert ids per row; e_total marks padding."""
    e_total = ep * g_local
    rng = np.random.default_rng(seed)
    T = ep * t_local
    if routing == "skewed":
        pool = rng.choice(e_total, size=max(topk, e_total // 12), replace=False)
    elif routing == "all_mine":
        # Every row picks shard 1's experts (and a few of shard 0's): one
        # shard owns nearly every pair, the worst case for its placement loop.
        pool = np.arange(g_local, min(e_total, 2 * g_local + topk))
    else:
        pool = np.arange(e_total)
    idx = np.stack([rng.choice(pool, size=topk, replace=False) for _ in range(T)])
    if routing in ("padded", "uniform"):
        # The layer's replica padding rows route nowhere: every slot e_total.
        for d in range(ep):
            lo = d * t_local + (t_local * 3) // 4
            idx[lo : (d + 1) * t_local] = e_total
        if routing == "padded":
            idx[t_local : 2 * t_local] = e_total  # shard 1 entirely padding
    return idx.astype(np.int32)


def _payloads(idx, ep, g_local, t_local, topk, seed):
    """Each shard's packed payload, as layer.select_local builds it."""
    e_total = ep * g_local
    block = host.routing_block(t_local, topk)
    layout = layer._exchange_layout(t_local, topk, e_total, True)
    rng = np.random.default_rng(seed + 1)
    scales = rng.uniform(1e-3, 2.0, size=(ep * t_local,)).astype(np.float32)
    rows = []
    for d in range(ep):
        blocks = jnp.asarray(idx[d * t_local : (d + 1) * t_local]).reshape(-1, block)
        hist = host.pair_block_hist(blocks, e_total)
        bits = lax.bitcast_convert_type(
            jnp.asarray(scales[d * t_local : (d + 1) * t_local]), jnp.int32
        )
        rows.append(
            layer._pack_routing_exchange(blocks.reshape(-1), bits, hist.sum(0), layout)
        )
    return jnp.stack(rows), layout, block


def _reference(packed_l, rank_l, *, ax, ep, e_total, t_local, topk, block, cap, stride, layout):
    """layer.compute_local's sharded-plan tables from the XLA all-gather."""
    me = rank_l[0, 0]
    packed = packed_l[0]
    exchange = lax.all_gather(packed, ax, axis=0, tiled=True).reshape(ep, layout[-1])
    local_blocks = packed[: t_local * topk].reshape(-1, block)
    block_hist = host.pair_block_hist(local_blocks, e_total)
    T = ep * t_local
    pairs_g, scale_bits_g, rows_by_dest = layer._unpack_routing_exchange(
        exchange, layout, t_local, topk, e_total
    )
    gathered = host.GatheredPairs(
        expert_blocks=pairs_g.reshape(T * topk // block, block),
        local_blocks=local_blocks,
        block_hist=block_hist,
        rows_by_dest=rows_by_dest,
        n_tokens=T,
        topk=topk,
    )
    routing = host.build_routing_tables_sharded(
        None, me, gathered=gathered, e_total=e_total, ep=ep, t_local=t_local,
        block=block, tile_m=cap, shard_stride=stride,
    )
    bt = host.shard_transport_tables_in_blocks(routing, me, e_total=e_total, ep=ep)
    rt = host.shard_push_tables_in_rows(routing, me, e_total=e_total, ep=ep)
    slab_row = host.local_slab_rows(routing, me, shard_stride=stride)
    tg = host.shard_token_gather(routing, me, shard_stride=stride, rows=slab_row)
    tg = jnp.pad(tg, (0, host.token_gather_window_rows(cap)))
    er, sb = host.shard_expert_slabs(routing, me, e_total=e_total, ep=ep)
    scale_rows = host.act_scale_slab_rows(stride)
    scale = host.slab_scatter(
        scale_rows * host.HIDDEN_LANE_BLOCK, slab_row, jnp.repeat(scale_bits_g, topk), by_set=False
    ).reshape(scale_rows, host.HIDDEN_LANE_BLOCK)
    visit, _ = host.expert_visit_list(er, e_total // ep)
    counts = host.shard_count_vector(routing, er, me, e_total=e_total, ep=ep)
    out = (*bt[:3], bt[5], rt[0], tg, er, sb, scale, visit, counts, routing.pos, routing.mirror_pos)
    return tuple(o[None] for o in out)


def _kernel(packed_l, rank_l, *, ep, e_total, t_local, topk, block, cap, stride, layout):
    fn = route_plan.build_route_plan(
        ep=ep, e_total=e_total, t_local=t_local, topk=topk, block=block,
        tile_m=cap, shard_stride=stride, layout=layout, quantized=True,
    )
    rp = fn(packed_l[0], rank_l)
    out = (
        *rp.kernel_tables, rp.token_gather, rp.expert_rows, rp.slab_base,
        lax.bitcast_convert_type(rp.scale_slab, jnp.int32), rp.visit, rp.counts,
        rp.pos, rp.mirror_pos,
    )
    return tuple(o[None] for o in out)


NAMES = (
    "run_start", "aligned_rows", "contrib_off", "push_dst", "true_rows", "token_gather",
    "expert_rows", "slab_base", "scale_slab_bits", "visit", "counts", "pos", "mirror_pos",
)


def _run(ep, g_local, t_local, topk, routing, seed):
    tpu = jax.devices()[0].platform == "tpu"
    devices = jax.devices() if tpu else jax.devices("cpu")
    if len(devices) < ep:
        pytest.skip(f"needs {ep} devices, have {len(devices)}")
    mesh = Mesh(np.array(devices[:ep]), ("ep",))
    e_total = ep * g_local
    cap = 128
    idx = _selections(ep, g_local, t_local, topk, routing, seed)
    packed, layout, block = _payloads(idx, ep, g_local, t_local, topk, seed)
    stride = host.ragged_stride_bound(ep * t_local, topk, e_total, cap)
    kw = dict(ep=ep, e_total=e_total, t_local=t_local, topk=topk, block=block,
              cap=cap, stride=stride, layout=layout)
    assert route_plan.route_plan_supported(
        ep=ep, e_total=e_total, t_local=t_local, topk=topk, block=block,
        tile_m=cap, shard_stride=stride,
    )
    rank = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)

    def sm(f, **extra):
        return jax.jit(
            jax.shard_map(
                lambda p, r: f(p, r, **extra, **kw),
                mesh=mesh,
                in_specs=(P("ep"), P("ep")),
                out_specs=P("ep"),
                check_vma=False,
            )
        )

    want = sm(_reference, ax="ep")(packed, rank)
    if tpu:
        got = sm(_kernel)(packed, rank)
    else:
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
            got = sm(_kernel)(packed, rank)
    for name, w, g in zip(NAMES, want, got):
        w, g = np.asarray(w), np.asarray(g)
        assert w.shape == g.shape, (name, w.shape, g.shape)
        for d in range(ep):
            bad = np.argwhere(w[d] != g[d])
            assert bad.size == 0, (
                f"{name} shard {d}: {len(bad)} mismatches, first at {bad[:4].tolist()}: "
                f"want {w[d][tuple(bad[0])]} got {g[d][tuple(bad[0])]}"
            )


@pytest.mark.multichip
@pytest.mark.parametrize("ep,g_local,t_local,topk", GEOMETRIES)
@pytest.mark.parametrize("routing", ROUTINGS)
def test_route_plan_matches_the_xla_tables(ep, g_local, t_local, topk, routing):
    _run(ep, g_local, t_local, topk, routing, seed=7)
