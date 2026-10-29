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
"""A Qwen3.5 pooled step over a pool that stores the conv state as f32.

The step reads and writes the conv window as f32 (the dtype the serving
framework's stock GDN kernel uses), on the split pool I/O path; it must match
the reference step and keep the kernel's conv state exactly (no bf16 round
trip), where a bf16-conv pool rounds it.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _cpu_mesh(R: int) -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _tiny_geom() -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4, moe_tp=False, e_slots=0,
    )  # fmt: skip


def _tiny_exec(R: int, B: int, T: int) -> mk.ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10)
    return mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=256,
        stack_e=True,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=96),
    )


def _with_conv_dtype(spec, dt):
    lay = replace(spec.cache_backend.layout, conv_dtype=dt)
    return replace(spec, cache_backend=spec.cache_backend.with_layout(lay)), lay


@pytest.mark.parametrize("T,B", [(1, 2), (2, 4)])
def test_f32_conv_pooled_step_matches_reference_and_keeps_conv_exact(T: int, B: int):
    mesh = _cpu_mesh(2)
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom()
    ex = _tiny_exec(R=2, B=B, T=T)
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(70 + T), g, ex)
    conv = np.asarray(st_l["conv"], np.float32)
    assert not np.array_equal(
        conv, np.asarray(st_l["conv"].astype(jnp.bfloat16), np.float32)
    )

    ref_out, ref_st = qwen35.reference_step(
        g, ex, pos, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    _, hid_mk, w_mk, st_mk_paged = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    _, mk_st = qwen35.make_step(mesh, g, ex, donate_states=False)[0](
        pos, hid_mk, w_mk, st_mk_paged
    )

    spec = qwen35.qwen35_spec(g)
    errs = {}
    for name, dt in (("f32", jnp.float32), ("bf16", jnp.bfloat16)):
        spec_dt, lay = _with_conv_dtype(spec, dt)
        lay.validate(ex)
        pt, ckpt, roff, min_np = pool.allocate_test_layout(lay, ex)
        assert mk.cache_exec(ex).num_pages >= min_np
        st_in = (
            st_l
            if name == "f32"
            else dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
        )
        raw_pools = tuple(
            jax.device_put(r, tp)
            for r in pool.states_to_pool(lay, ex, st_in, pt, ckpt, roff)
        )
        step, _ = mk.PallasWriter().make_step(mesh, spec_dt, ex, donate=False)
        out_hid, raw_out = step(pos, hid_mk, w_mk, raw_pools, pt, ckpt, roff)
        out_b = mk.unfold(out_hid, ex.R)[0, :B].astype(jnp.float32)
        assert jnp.allclose(out_b, ref_out.astype(jnp.float32), atol=3e-2, rtol=3e-2)
        st_after = pool.pool_to_states(lay, ex, raw_out, pt, ckpt)

        QKV = g.QKV(ex.R)
        if T == 1:
            kern = (
                mk.unfold(mk_st["conv"], ex.R)[:, :, : 3 * B, :]
                .reshape(ex.R, g.NGDN, 3, B, QKV)
                .swapaxes(2, 3)
            )
            got = st_after["conv"][:, :, :, 0]
            ref = ref_st["conv"][:, :, : 3 * B, :].reshape(ex.R, g.NGDN, 3, B, QKV)
            ref = ref.swapaxes(2, 3)
        else:
            kern = mk.unfold(mk_st["conv_ckpt"], ex.R)[
                :, :, :, 1 + T - 1 : 4 + T - 1, :
            ]
            got = st_after["conv"][:, :, :, T - 1]
            ref = ref_st["conv_ckpt"][:, :, :, 1 + T - 1 : 4 + T - 1, :]
        if name == "f32":
            # The pool keeps the kernel's f32 conv state bit-for-bit.
            np.testing.assert_array_equal(np.asarray(got), np.asarray(kern))
        errs[name] = float(jnp.max(jnp.abs(got - ref)))
    assert errs["f32"] <= errs["bf16"], errs
