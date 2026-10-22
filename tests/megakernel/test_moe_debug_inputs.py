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
"""`MoEBlock` debug inputs: forced routing (`rbias`) and the row mask (`moe_rows`).

The flags live in `DebugOptions.blocks["moe"]` (`MoEDebug`), not in `Tiles`, so
they leave the default-tile VMEM fit on; each adds a block extra input that
`kernel.make_step` and `cache.make_pooled_step` pass after the states.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import cache, kernel, planner, vmem
from vllm_torchtpu.megakernel.blocks import MoEBlock, MoEDebug
from vllm_torchtpu.megakernel.blocks.moe import MoETunables
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES

R = 2
_BOTH = MoEDebug(forced_routing=True, row_mask=True)


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _debug(flags: MoEDebug | None = None, skip: frozenset[str] = frozenset()) -> Any:
    blocks = {MoEBlock.name: flags} if flags is not None else {}
    return mk.DebugOptions(skip=skip, blocks=blocks)


def test_moe_debug_flags_declare_inputs_and_keep_the_default_tile_fit():
    spec = qwen35.qwen35_spec(qwen35.FULL_EP)
    base = mk.ExecSpec(B=64, T=1, R=8, ctx=4096)
    ex = dataclasses.replace(base, debug=_debug(_BOTH))
    assert planner.extra_inputs(spec, base) == []
    assert [
        (x.name, x.memory_space, x.shape) for x in planner.extra_inputs(spec, ex)
    ] == [
        ("rbias", "hbm", ()),
        ("moe_rows", "smem", (64,)),
    ]
    # The flags leave `tiles=None`, so the VMEM fit still runs (and shrinks a
    # ring at this bucket) and the fitted plan stays in budget.
    assert ex.tiles is None
    fitted = vmem.fit_default_tiles(spec, ex)
    assert fitted.tiles is not None and fitted.tiles != mk.default_tiles(64, R=8)
    assert fitted.tiles.block(MoEBlock.name) == MoETunables()
    core_plan, _ = vmem.build_scratch(spec, fitted)
    assert core_plan.vmem_bytes <= fitted.tiles.vmem_mib * (1 << 20)
    with pytest.raises(TypeError, match="MoEDebug"):
        MoEBlock().debug_flags(
            dataclasses.replace(base, debug=_debug(MoETunables(pipe=False)))  # type: ignore[arg-type]
        )


def _geom() -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4,
    )  # fmt: skip


def _exec(B: int, paged: bool) -> mk.ExecSpec:
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10).with_blocks(
        {"gqa": {"kv_blk": 256}}
    )
    return mk.ExecSpec(
        R=R,
        B=B,
        T=1,
        ctx=256,
        stack_e=True,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=96) if paged else None,
    )


@pytest.mark.slow
@pytest.mark.parametrize("paged", [False, True], ids=["make_step", "pooled_step"])
def test_moe_forced_routing_and_row_mask_inputs_cpu_interpret(paged: bool):
    """Forced routing and the row mask through both step builders (bitwise).

    - `rbias` picks each row's routed experts: two biases that pick different
      experts give different outputs;
    - all rows masked off equals skipping the routed experts;
    - with disjoint experts per row, masking row 1 drops its experts only: row 0
      matches the unmasked run, row 1 the all-masked run.
    """
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices")
    mesh = Mesh(np.array(devs[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = _geom()
    B = 2
    ex = _exec(B, paged)
    spec = qwen35.qwen35_spec(g)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(7), g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=paged
    )
    if paged:
        pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
        pools = tuple(
            jax.device_put(np.asarray(r).copy(), tp)
            for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
        )

    def run(dbg: Any, *extra: Any) -> np.ndarray:
        e = dataclasses.replace(ex, debug=dbg)
        if paged:
            step, _ = cache.make_pooled_step(mesh, spec, e)
            out, _ = step(pos, hid_mk, w_mk, pools, pt, ckpt, roff, *extra)
        else:
            step, _ = kernel.make_step(mesh, spec, e)
            out, _ = step(pos, hid_mk, w_mk, st_mk, *extra)
        return np.asarray(mk.unfold(out, R)[0, :B]).view(np.uint8)

    def rbias(*row_experts: list[int]) -> jax.Array:
        """Forced routing of row `b` to `row_experts[b]` on every layer."""
        r = np.zeros((R * g.L, ex.M, g.E), np.float32)
        for b, experts in enumerate(row_experts):
            r[:, b, experts] = 1.0
        return jnp.asarray(r)

    on = jnp.ones((B,), jnp.int32)
    off = jnp.zeros((B,), jnp.int32)
    dbg = _debug(_BOTH)
    disjoint = rbias([0, 1], [2, 3])
    h_a = run(dbg, disjoint, on)
    h_b = run(dbg, rbias([4, 5], [6, 7]), on)
    assert not np.array_equal(h_a[0], h_b[0]) and not np.array_equal(h_a[1], h_b[1])
    h_off = run(dbg, disjoint, off)
    np.testing.assert_array_equal(h_off, run(_debug(skip=frozenset({"experts"}))))
    # Row 1 masked off: its experts (2, 3) leave the routed list, row 0's stay.
    h_mix = run(dbg, disjoint, jnp.asarray([1, 0], jnp.int32))
    np.testing.assert_array_equal(h_mix[0], h_a[0])
    np.testing.assert_array_equal(h_mix[1], h_off[1])
