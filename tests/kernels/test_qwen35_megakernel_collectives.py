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
"""Unit tests for Qwen3.5 megakernel collectives (`PallasAllReduce`) and TP mesh topology."""

from __future__ import annotations

import dataclasses
import inspect
import sys
from types import SimpleNamespace

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen35_megakernel import (
    MKBuildContext,
    PallasAllReduce,
    TINY,
    TINY_TILES,
    megakernel as K,
    mtp_adapter,
    vllm_adapter as VA,
)
from vllm_torchtpu.kernels.qwen35_megakernel.megakernel import (
    BF16,
    COLLECTIVE_ID,
    F32,
    I32,
    rnd,
)
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import _build_tp_rank_mesh

from qwen35_megakernel_test_utils import setup_8_cpu_devices


@pytest.mark.parametrize(
    "hier,rsag",
    [
        (0, False),
        (0, True),
        (1, False),
        (2, False),
        (3, False),
    ],
)
def test_standalone_pallas_allreduce_interpret(hier: int, rsag: bool):
    """Test PallasAllReduce standalone in pl.pallas_call(interpret=True) vs jnp.sum."""
    devs = setup_8_cpu_devices()
    r_ranks = 8 if len(devs) >= 8 else 1
    if r_ranks == 1 and (hier > 0 or rsag):
        ctx8 = MKBuildContext(
            dataclasses.replace(TINY, R=8, B=4, H=1024),
            dataclasses.replace(TINY_TILES, hier=hier, rsag=rsag),
            interpret=True,
        )
        ar8 = PallasAllReduce(ctx8)
        assert ar8.ctx.HIER == hier

    c = dataclasses.replace(
        TINY,
        R=r_ranks,
        NKV=min(TINY.NKV, r_ranks),
        B=4,
        T=1,
        L=1,
        att_every=1,
        H=1024,
        paged_kv=False,
    )
    tiles = dataclasses.replace(TINY_TILES, hier=hier, rsag=rsag, war=1)
    ctx = MKBuildContext(c, tiles, interpret=True)
    ar_s_names = [
        k
        for k in (
            "res",
            "send",
            "recv",
            "ag_send",
            "ag_recv",
            "s2_send",
            "s2_recv",
            "ar_ssem",
            "ar_rsem",
        )
        if k in ctx.scratch
    ]

    def ar_kernel(res_in_ref, part_in_ref, out_ref, *scr_refs):
        ctx.scr = dict(zip(ar_s_names, scr_refs))
        ctx.me = lax.axis_index("tp").astype(I32) if ctx.R > 1 else jnp.int32(0)
        ctx.row_m = lax.broadcasted_iota(I32, (ctx.MB, 1), 0)
        ctx.live = ctx.row_m < ctx.B
        ar = PallasAllReduce(ctx)
        ctx.scr["res"][0 : ctx.MB] = res_in_ref[0 : ctx.MB]
        ar.add_residual(ar.allreduce(0, part_in_ref[...], jnp.bool_(True)))
        if ctx.comm:
            for rnd_i in ctx.AR_ROUNDS:
                for d in ar.ar_copies(0, rnd_i):
                    d.wait_send()
        out_ref[0 : ctx.MB] = ctx.scr["res"][0 : ctx.MB]

    pcall = pl.pallas_call(
        ar_kernel,
        out_shape=jax.ShapeDtypeStruct((ctx.MB, c.H), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[ctx.scratch[k] for k in ar_s_names],
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID if r_ranks > 1 else None
        ),
        interpret=pltpu.InterpretParams(),
    )
    mesh = Mesh(np.array(devs[:r_ranks]), ("tp",))
    res_in = jax.random.normal(
        jax.random.PRNGKey(0), (ctx.MB, c.H), BF16
    ).astype(F32)
    res_in = jnp.where(jnp.arange(ctx.MB)[:, None] < c.B, res_in, 0.0)
    parts = jax.random.normal(
        jax.random.PRNGKey(1), (r_ranks, ctx.MB, c.H), BF16
    ).astype(F32)
    parts = jnp.where(jnp.arange(ctx.MB)[None, :, None] < c.B, parts, 0.0)

    out = jax.shard_map(
        lambda r, p: pcall(r, p[0]),
        mesh=mesh,
        in_specs=(P(), P("tp")),
        out_specs=P(),
        check_vma=False,
    )(res_in, parts)

    ref = rnd(res_in[: c.B] + jnp.sum(parts[:, : c.B].astype(BF16).astype(F32), axis=0))
    np.testing.assert_allclose(
        np.asarray(out[: c.B]), np.asarray(ref), rtol=2e-2, atol=1e-1
    )


def test_build_tp_rank_mesh_preserves_rank_order(monkeypatch) -> None:
    """Verify `_build_tp_rank_mesh` preserves `tpu7x` rank order."""
    tpu7x_order = [0, 1, 4, 5, 6, 7, 2, 3]
    fake_devices = np.array(
        [SimpleNamespace(id=d) for d in tpu7x_order], dtype=object
    )
    fake_mesh = SimpleNamespace(devices=fake_devices, axis_names=("tp",))
    fake_pallas = SimpleNamespace(get_pallas_mesh=lambda axis_names: fake_mesh)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas", fake_pallas)

    mesh = _build_tp_rank_mesh(8)
    assert mesh is fake_mesh
    assert [int(d.id) for d in mesh.devices.flat] == tpu7x_order


def test_megakernel_meshes_are_rank_ordered():
    """Every megakernel mesh comes from `_build_tp_rank_mesh` (device-id sorted meshes put
    `axis_index('tp')` != tp_rank on tpu7x)."""
    for mod in (VA, mtp_adapter):
        assert not hasattr(mod, "build_ep_mesh")
        assert "build_ep_mesh(" not in inspect.getsource(mod)
    for cls in (VA.Qwen35Megakernel, mtp_adapter.Qwen35MtpMegakernel):
        assert "_build_tp_rank_mesh(self.tp_size)" in inspect.getsource(cls.__init__)


def test_default_tiles_r16_only_change_the_all_reduce():
    all_verify = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
    for b in all_verify:
        for base in (K.FULL_EP, K.FULL):
            t8 = K.default_tiles(dataclasses.replace(base, B=b))
            t16 = K.default_tiles(dataclasses.replace(base, R=16, B=b))
            assert t16.hier == 0, b
            assert dataclasses.replace(t16, hier=t8.hier) == t8, b
            assert t16.rsag == (b > 4), b
