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
"""Qwen3.8-shaped (`KH=16 < R=32`, `IS=2048 < 128*R`) hybrid composition at 32 ranks."""

from __future__ import annotations

from dataclasses import replace as dc_replace

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel as mk
from tests.megakernel.megakernel_fakes import FakeGeometry
from vllm_torchtpu.megakernel import comm, kernel, vmem
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import (
    qwen35,
)
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.ops import BF16, F32, I32, rnd
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec, Module, WeightDecl

R = 32
TOL = 3e-2
CTX = 2048
POS0 = 1800
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 540

QWEN38_FULL_HF_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 8192,
        "num_hidden_layers": 92,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 128,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 2048,
        "shared_expert_intermediate_size": 2048,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}


def _tiny_qwen38_geometry() -> qwen35.Qwen35Geometry:
    """Tiny Qwen3.8-shaped geometry (`KH=16 < R=32`, `IS=2048 < 128*R`) at R=32."""
    return qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=1024,
        NH=32,
        NKV=4,
        HD=128,
        ROT=32,
        KH=16,
        VH=32,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=2048,
        moe_tp=False,
    )


def _tiny_qwen38_r8_geometry() -> qwen35.Qwen35Geometry:
    """8-rank geometry with identical replication factors (`kh_rep=2`, `sh_rep=2`)."""
    return qwen35.Qwen35Geometry(
        L=4,
        att_every=4,
        H=1024,
        NH=8,
        NKV=2,
        HD=128,
        ROT=32,
        KH=4,
        VH=8,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=512,
        moe_tp=False,
    )


def _rms(x) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.square(x.astype(F32)))))


def _report(name: str, got, exp) -> None:
    err = jnp.abs(got.astype(F32) - exp.astype(F32))
    print(
        f"{name}: max_err={float(err.max()):.3e} rms_err={_rms(err):.3e} "
        f"rms_ref={_rms(exp):.3e}"
    )


def _check(name: str, got, exp, controls) -> None:
    _report(name, got, exp)
    assert jnp.allclose(got, exp, atol=TOL, rtol=TOL), name
    for label, bad in controls:
        _report(f"{name} vs {label}", bad, exp)
        assert not jnp.allclose(bad, exp, atol=TOL, rtol=TOL), f"{name}: {label}"


def _swap(x, a: int, b: int):
    return x.at[a].set(x[b]).at[b].set(x[a])


pytestmark = pytest.mark.tpu_test


def test_tpu_hier_allreduce_8ranks_silicon():
    """Two-level hierarchical all-reduce (hier=4, R=8) on TPU silicon vs f32 ref."""

    class _TwoTileModule(Module):
        name = "two_tile"
        reduction = "allreduce"

        def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
            return [
                WeightDecl("w1", K=256, N=128, dtype=BF16, stream_ring="static"),
                WeightDecl("w2", K=128, N=256, dtype=BF16, stream_ring="static"),
            ]

        def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
            return {"acc": 256, "xb2": 128}

    class _SecondTileModule(Module):
        name = "second_tile"
        reduction = "allreduce"

    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip(f"Requires >= 8 TPU cores, got {len(devs)}")
    R = 8
    H = 1024
    is_cpu = devs[0].platform == "cpu"
    tiles = dc_replace(TINY_TILES, hier=4, hier_group=4, rsag=False, war=1, mb8=2)
    ex = ExecSpec(B=4, T=1, R=R, ctx=128, tiles=tiles, interpret=is_cpu)
    g = FakeGeometry(L=2, H=H)
    ms = ModelSpec(
        "ar_tpu_smoke",
        config=g,
        layer_type=LayerType("lt", slots=(_TwoTileModule(), _SecondTileModule())),
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    s_names = list(scratch.keys())

    def pallas_body(x_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        ar = comm.AllReduceRing(cp.comm, scr, me)

        barrier = pltpu.get_barrier_semaphore()
        for off in range(1, R):
            pl.semaphore_signal(
                barrier,
                1,
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(barrier, R - 1)

        scr["res"][...] = jnp.zeros((ex.MB, g.H), F32)

        def layer_fn(lyr, _):
            first = lyr == 0
            tot0 = ar.allreduce(0, x_ref[...], first)
            ar.add_residual(tot0)
            tot1 = ar.allreduce(1, x_ref[...], first)
            ar.add_residual(tot1)
            return 0

        lax.fori_loop(0, 2, layer_fn, 0)
        ar.drain_sends()
        out_ref[...] = scr["res"][...]

    call = pl.pallas_call(
        pallas_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, g.H), F32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams() if is_cpu else False,
    )

    mesh = Mesh(devs[:R], ("tp",))
    step = jax.jit(
        jax.shard_map(
            lambda x: call(x[0])[None],
            mesh=mesh,
            in_specs=(P("tp"),),
            out_specs=P("tp"),
            check_vma=False,
        )
    )

    x_all = jax.random.normal(
        jax.random.PRNGKey(42), (R, ex.MB, g.H), dtype=BF16
    ).astype(F32)
    out_all = step(x_all)

    def _expected_res(x_in):
        exp_one = jnp.sum(x_in[:, : ex.B].astype(BF16).astype(F32), axis=0)
        exp = jnp.zeros_like(exp_one)
        for _ in range(4):
            exp = rnd(exp + rnd(exp_one))
        return exp

    expected = _expected_res(x_all)
    group0_only = _expected_res(x_all[:4])
    for r in range(1, R):
        assert jnp.array_equal(out_all[r, : ex.B], out_all[0, : ex.B])
    assert jnp.allclose(out_all[0, : ex.B], expected, atol=2e-1, rtol=2e-2)
    assert not jnp.allclose(out_all[0, : ex.B], group0_only, atol=2e-1, rtol=2e-2)


def test_tpu_qwen38_tiny_8ranks_silicon():
    """Qwen3.8 TINY R=8 make_step matches reference_step on TPU silicon."""
    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip(f"Requires >= 8 TPU cores, got {len(devs)}")
    g = _tiny_qwen38_geometry()
    ex8 = ExecSpec(
        B=2,
        T=1,
        R=8,
        ctx=CTX,
        tiles=dc_replace(TINY_TILES, bank=256 << 10, hier=0, mb8=1, inplace=False),
        interpret=devs[0].platform == "cpu",
    )
    _, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.PRNGKey(77), g, ex8)
    pos = POS0 + jnp.arange(ex8.NS, dtype=I32)
    ref_hid, _ = qwen35.reference_step(g, ex8, pos, hid_l, w_l, st_l)

    # Swapped-rank negative control
    perm = jnp.array([4, 1, 2, 3, 0, 5, 6, 7], dtype=jnp.int32)
    w_swapped = dict(w_l, lin_in=w_l["lin_in"][perm])
    ref_swapped, _ = qwen35.reference_step(g, ex8, pos, hid_l, w_swapped, st_l)

    mesh = Mesh(devs[: ex8.R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex8, mesh=mesh
    )
    step_fn, _ = kernel.make_step(
        mesh, g.to_model_spec(R=ex8.R), ex8, donate_states=True
    )
    out_hid, _ = step_fn(pos_mk, hid_mk, w_mk, st_mk)
    hid_r = kernel.unfold(out_hid, ex8.R)[:, : ex8.B]
    _check(
        "tpu silicon qwen38 tiny R=8 make_step vs reference_step",
        hid_r[0],
        ref_hid,
        [("swapped rank 0/4 weights", ref_swapped), ("input hidden", hid_l)],
    )
