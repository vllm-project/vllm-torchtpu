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
with what layer.xla_route_plan -- the layer's own XLA plan -- builds from the
routing all-gather on the sharded plan: the transport/push tables, the
token-gather table, the expert rows and slab bases, the activation scale slab,
the visit list, the count table and the arrival/mirror positions. The XLA plan
builds the slab tables as scatter-adds.

On a TPU host with at least `ep` devices this runs the kernel natively over the
first `ep` of them (the `multichip` marker). Otherwise it runs it in Pallas' TPU
interpret mode on CPU, which emulates the remote DMAs and semaphores and needs
`ep` CPU devices: in-process when this process already has them (e.g.
XLA_FLAGS=--xla_force_host_platform_device_count=64), else in a child process
with JAX_PLATFORMS=cpu and that many host devices. On an 8-core host an
interpreted case takes about 1 minute at EP8, 2.5 at EP32 and 5 at EP64.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax import lax  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v2 import host, layer, route_plan  # noqa: E402

# (ep, g_local, t_local, topk)
GEOMETRIES = [
    pytest.param(32, 24, 16, 10, id="ep32_g24_t16"),
    pytest.param(8, 24, 16, 10, id="ep8_g24_t16"),
    # The 12-token MTP draft bucket: 3 rows per replica, aligned to 4.
    pytest.param(8, 24, 4, 10, id="ep8_g24_t4"),
    pytest.param(32, 24, 4, 10, id="ep32_g24_t4"),
    # 320 pairs per shard: past 256, which the old pair-count bound refused.
    pytest.param(8, 24, 32, 10, id="ep8_g24_t32"),
    pytest.param(32, 24, 32, 10, id="ep32_g24_t32"),
    # Past 128 tokens the per-token row scales fill more than one lane block,
    # and tokens 128.. read theirs from the second. The padded routing pads
    # those rows away; skewed and all_mine route them.
    pytest.param(8, 24, 160, 10, id="ep8_g24_t160"),
    # Local expert counts that are not a multiple of 8: 768 experts at EP64,
    # and per-shard counts of 12 and 6 on meshes a host can fit.
    pytest.param(8, 12, 16, 10, id="ep8_g12_t16"),
    pytest.param(32, 6, 16, 10, id="ep32_g6_t16"),
    pytest.param(64, 12, 16, 10, id="ep64_g12_t16"),
    # Pair counts that are not a multiple of 8: one and two rows per replica,
    # what small draft buckets split to (10 and 20 pairs).
    pytest.param(8, 24, 1, 10, id="ep8_g24_t1"),
    pytest.param(8, 24, 2, 10, id="ep8_g24_t2"),
    pytest.param(4, 16, 8, 8, id="ep4_g16_t8"),
]
ROUTINGS = ["skewed", "padded", "all_mine"]

# The chip the SMEM budget is checked against off TPU, where get_tpu_info()
# has no chip to read: v7x, where the budget's headroom was measured.
V7X_INFO = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
SMEM_BUDGET = V7X_INFO.smem_capacity_bytes - route_plan.SMEM_HEADROOM_BYTES
# Per case, above the multichip lane's --timeout=600 and the child process's
# own limit, so a slow interpreted case fails on the child's timeout with its
# output rather than on pytest's.
CASE_TIMEOUT_S = 1500
CHILD_TIMEOUT_S = 1200


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
    if routing == "padded":
        # The layer's replica padding rows route nowhere: every slot e_total.
        for d in range(ep):
            lo = d * t_local + (t_local * 3) // 4
            idx[lo : (d + 1) * t_local] = e_total
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


def _flat(rp):
    """A RoutePlan as one [1, ...] array per table, scale slab as its bits."""
    out = (
        *rp.kernel_tables,
        rp.token_gather,
        rp.expert_rows,
        rp.slab_base,
        lax.bitcast_convert_type(rp.scale_slab, jnp.int32),
        rp.visit,
        rp.counts,
        rp.pos,
        rp.mirror_pos,
    )
    return tuple(o[None] for o in out)


def _reference(
    packed_l,
    rank_l,
    *,
    ax,
    ep,
    e_total,
    t_local,
    topk,
    block,
    cap,
    stride,
    layout,
):
    """layer.xla_route_plan over the XLA all-gather of the same payloads."""
    packed = packed_l[0]
    exchange = lax.all_gather(packed, ax, axis=0, tiled=True).reshape(ep, layout[-1])
    local_blocks = packed[: t_local * topk].reshape(-1, block)
    rp, _ = layer.xla_route_plan(
        exchange,
        local_blocks,
        host.pair_block_hist(local_blocks, e_total),
        rank_l[0, 0],
        ep=ep,
        e_total=e_total,
        t_local=t_local,
        topk=topk,
        block=block,
        tile_m=cap,
        shard_stride=stride,
        layout=layout,
        quantized=True,
        sharded_plan=True,
    )
    return _flat(rp)


def _kernel(
    packed_l, rank_l, *, ep, e_total, t_local, topk, block, cap, stride, layout
):
    fn = route_plan.build_route_plan(
        ep=ep,
        e_total=e_total,
        t_local=t_local,
        topk=topk,
        block=block,
        tile_m=cap,
        shard_stride=stride,
        layout=layout,
        quantized=True,
    )
    return _flat(fn(packed_l[0], rank_l))


NAMES = (
    "run_start",
    "aligned_rows",
    "contrib_off",
    "push_dst",
    "true_rows",
    "token_gather",
    "expert_rows",
    "slab_base",
    "scale_slab_bits",
    "visit",
    "counts",
    "pos",
    "mirror_pos",
)


_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"


def _run_in_cpu_child(ep, g_local, t_local, topk, routing, seed):
    """Run one case in a child process with `ep` interpreted CPU devices.

    The host device count is fixed when JAX starts its CPU backend, and on a
    TPU host this process holds the chips, so the case needs a fresh process.
    """
    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={ep}"]),
    )
    args = [ep, g_local, t_local, topk, routing, seed]
    proc = subprocess.run(
        [sys.executable, __file__, *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-5000:]


def _run(ep, g_local, t_local, topk, routing, seed):
    tpu = jax.devices()[0].platform == "tpu"
    devices = jax.devices() if tpu else jax.devices("cpu")
    if len(devices) < ep:
        _run_in_cpu_child(ep, g_local, t_local, topk, routing, seed)
        return
    mesh = Mesh(np.array(devices[:ep]), ("ep",))
    e_total = ep * g_local
    cap = 128
    idx = _selections(ep, g_local, t_local, topk, routing, seed)
    packed, layout, block = _payloads(idx, ep, g_local, t_local, topk, seed)
    stride = host.ragged_stride_bound(ep * t_local, topk, e_total, cap)
    kw = dict(
        ep=ep,
        e_total=e_total,
        t_local=t_local,
        topk=topk,
        block=block,
        cap=cap,
        stride=stride,
        layout=layout,
    )
    assert route_plan.route_plan_supported(
        ep=ep,
        e_total=e_total,
        t_local=t_local,
        topk=topk,
        block=block,
        tile_m=cap,
        shard_stride=stride,
        info=None if tpu else V7X_INFO,
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
                f"{name} shard {d}: {len(bad)} mismatches, "
                f"first at {bad[:4].tolist()}: "
                f"want {w[d][tuple(bad[0])]} got {g[d][tuple(bad[0])]}"
            )


@pytest.mark.parametrize(
    "ep,t_local,supported",
    [
        (32, 16, True),
        (32, 128, True),
        (32, 242, True),
        # One token more and the second lane block of row scales per source
        # (16 KiB at EP32) no longer fits.
        (32, 243, False),
        # The 1024-token bucket over TP4 at EP32: its SMEM scratch overflowed
        # (5.2 KiB over 1 MiB) and failed the compile.
        (32, 256, False),
        (64, 64, True),
        (64, 128, False),
        (128, 32, True),
        (128, 64, False),
        (8, 256, True),
    ],
)
def test_route_plan_refuses_shapes_over_the_smem_budget(ep, t_local, supported):
    e_total, topk, cap = 768, 10, 128
    stride = host.ragged_stride_bound(ep * t_local, topk, e_total, cap)
    smem = route_plan.route_plan_smem_bytes(
        ep=ep, e_total=e_total, t_local=t_local, topk=topk, shard_stride=stride
    )
    assert (smem <= SMEM_BUDGET) == supported, smem
    assert (
        route_plan.route_plan_supported(
            ep=ep,
            e_total=e_total,
            t_local=t_local,
            topk=topk,
            block=host.routing_block(t_local, topk),
            tile_m=cap,
            shard_stride=stride,
            info=V7X_INFO,
        )
        == supported
    )


@pytest.mark.parametrize(
    "topk,supported",
    [
        # 256 tokens x 16 = 4096 pairs: the bitmask's 128 words, full.
        (16, True),
        # 4352 pairs need 136 words; the SMEM budget alone would admit it.
        (17, False),
    ],
)
def test_route_plan_refuses_pairs_past_the_bitmask(topk, supported):
    ep, e_total, t_local, cap = 2, 768, 256, 128
    stride = host.ragged_stride_bound(ep * t_local, topk, e_total, cap)
    smem = route_plan.route_plan_smem_bytes(
        ep=ep, e_total=e_total, t_local=t_local, topk=topk, shard_stride=stride
    )
    assert smem <= SMEM_BUDGET
    assert (
        route_plan.route_plan_supported(
            ep=ep,
            e_total=e_total,
            t_local=t_local,
            topk=topk,
            block=host.routing_block(t_local, topk),
            tile_m=cap,
            shard_stride=stride,
            info=V7X_INFO,
        )
        == supported
    )


@pytest.mark.multichip
@pytest.mark.timeout(CASE_TIMEOUT_S)
@pytest.mark.parametrize("ep,g_local,t_local,topk", GEOMETRIES)
@pytest.mark.parametrize("routing", ROUTINGS)
def test_route_plan_matches_the_xla_tables(ep, g_local, t_local, topk, routing):
    _run(ep, g_local, t_local, topk, routing, seed=7)


if __name__ == "__main__":
    # The child process of _run_in_cpu_child: ep g_local t_local topk routing seed.
    ep, g_local, t_local, topk, routing, seed = sys.argv[1:]
    _run(int(ep), int(g_local), int(t_local), int(topk), routing, int(seed))
