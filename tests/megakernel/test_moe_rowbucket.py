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
"""Routed-row bucketing (`MoETunables.row_bucket`) vs the dense PIPE_EP expert loop.

Runs one EP MoE layer (R=2, PIPE_EP: e13 in 2 tiles, e2 in 1) in Pallas
interpret mode with `row_bucket` off and on and requires bitwise-equal hidden
outputs, at the c32 decode row counts and with one hot expert holding most
rows (the chunked path).

Each comparison runs in a subprocess with `--xla_allow_excess_precision=false`
(and FMA off, `--xla_cpu_max_isa=AVX`): by default XLA:CPU drops some of the
kernel's f32 -> bf16 -> f32 roundings (`rnd`) depending on the graph around
them (a `pl.when` branch is enough), which changes either loop's output
independently of the row bucketing. Mosaic keeps every rounding.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.megakernel.megakernel_helpers import TINY_TILES, FakeGeometry, MiniCtx
from vllm_torchtpu.megakernel import comm, planner, stream, vmem
from vllm_torchtpu.megakernel.blocks import MoEBlock
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec

R = 2
H, E, TOPK, IR, ISR = 256, 128, 8, 128, 128
HOT = 5  # expert id the hot case routes (almost) every row to (rank 0)


def _run(
    B: int, row_bucket: bool, hot: bool = False, direct0: bool = False, **interp
):
    """Hidden rows after one MoE layer (+ residual), all ranks, and the RB flags."""
    devs = jax.devices("cpu")
    assert len(devs) >= R
    tiles = dataclasses.replace(
        TINY_TILES, bank=512 << 10, nbd=8, nbe=6, ebank=32 << 10, war=1
    ).with_blocks(
        {
            "moe": {
                "pipe_ep": True,
                "moe_unroll": True,
                "r2_direct_store": direct0,
                "topk_lanes": True,
                "row_bucket": row_bucket,
            }
        }
    )
    g = FakeGeometry(
        L=1, att_every=1, H=H, NH=2, NKV=1, HD=128, E=E, TOPK=TOPK, I=IR,
        IS=ISR * R, moe_tp=False, e_slots=0,
    )
    moe = MoEBlock()
    ms = ModelSpec("moe_rb", config=g, layer_type=LayerType("lt", slots=(moe,)))
    ex = ExecSpec(
        B=B, T=1, R=R, ctx=128, stack_e=True, tiles=tiles,
        interpret=pltpu.InterpretParams(**interp),
    )
    cp, scratch = vmem.build_scratch(ms, ex)
    mp = cp.block("moe")
    assert mp.PIPE_EP and (mp.T13, mp.T2) == (2, 1)
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    keys = jax.random.split(jax.random.PRNGKey(7 + B), 14)
    hid = jax.random.normal(keys[0], (ex.M, H), dtype=F32) * 0.5
    router = jax.random.normal(keys[2], (H, E), dtype=F32) * 0.1
    if hot:
        # Every row leans along `v`, and expert HOT's router column is `v`: it wins
        # the top-k of (almost) every row.
        v = jnp.sign(jax.random.normal(keys[12], (H,), dtype=F32))
        hid = hid * 0.2 + v[None, :]
        router = router.at[:, HOT].set(v * 0.5)
    hid = hid.astype(BF16)
    router = router.astype(BF16)
    post_norm = jax.random.uniform(keys[1], (H,), dtype=F32, minval=-0.2, maxval=0.2)
    sh_gate = jax.random.normal(keys[3], (H,), dtype=F32) * 0.2
    sh13 = (jax.random.normal(keys[4], (R, H, 2 * ISR), dtype=F32) * 0.4).astype(FP8)
    sh13_s = jax.random.uniform(keys[5], (R, 2 * ISR), dtype=F32, minval=0.05, maxval=0.15)
    sh2 = (jax.random.normal(keys[6], (R, ISR, H), dtype=F32) * 0.4).astype(FP8)
    sh2_s = jax.random.uniform(keys[7], (R, H), dtype=F32, minval=0.05, maxval=0.15)
    EW = E // R
    e13 = (jax.random.normal(keys[8], (R, EW, H, 2 * IR), dtype=F32) * 0.4).astype(FP8)
    e13_s = jax.random.uniform(keys[9], (R, EW, 2 * IR), dtype=F32, minval=0.05, maxval=0.15)
    e2 = (jax.random.normal(keys[10], (R, EW, IR, H), dtype=F32) * 0.4).astype(FP8)
    e2_s = jax.random.uniform(keys[11], (R, EW, H), dtype=F32, minval=0.05, maxval=0.15)

    lay = p.smalls_layout
    sm = jnp.zeros((R, 1, 1, p.smalls_width), dtype=F32)
    for name, val in [
        ("post_norm", jnp.broadcast_to(post_norm[None, :], (R, H))),
        ("sh13_s", sh13_s),
        ("sh2_s", sh2_s),
        ("sh_gate", jnp.broadcast_to(sh_gate[None, :], (R, H))),
    ]:
        o, w = lay[name]
        sm = sm.at[:, 0, 0, o : o + w].set(val)
    w_by_name = {
        "router": jnp.broadcast_to(router[None, ...], (R, 1, H, E)).reshape(
            R, *wsh["router"][0]
        ),
        "sh13": sh13.reshape(R, *wsh["sh13"][0]),
        "sh2": sh2.reshape(R, *wsh["sh2"][0]),
        "e13": e13.reshape(R, *wsh["e13"][0]),
        "e2": e2.reshape(R, *wsh["e2"][0]),
        "es": jnp.concatenate([e13_s, e2_s], axis=-1).reshape(R, *wsh["es"][0]),
        "smalls": sm,
        "final_norm": jnp.zeros((R, *wsh["final_norm"][0]), dtype=F32),
    }

    def pallas_body(pos_ref, hid_ref, *rest):
        n_w, n_o = len(p.w_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        O = dict(zip(p.o_names, rest[n_w : n_w + n_o]))  # noqa: E741
        scr = dict(zip(s_names, rest[n_w + n_o :]))
        me = lax.axis_index("tp").astype(I32)
        barrier = pltpu.get_barrier_semaphore()
        for off in range(1, R):
            pl.semaphore_signal(
                barrier, 1, device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, R - 1)
        ring = stream.TwoRing(cp.stream, W, scr, me, interpret=True)
        ar = comm.AllReduceRing(cp.comm, scr, me)
        ctx = MiniCtx(
            g=g, ex=ex, core_plan=cp, pos_ref=pos_ref, W=W, ST={}, O=O, scr=scr,
            me=me, stream=ring, comm=ar,
        )
        for i in range(cp.stream.NBD):
            ring.fetch(jnp.int32(i))
        pltpu.make_async_copy(W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).start()
        moe.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        moe.init_buffers(ctx)
        pltpu.make_async_copy(W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).wait()
        moe.on_layer_start(ctx, jnp.int32(0), jnp.int32(0))
        ring.begin_layer(jnp.int32(0))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        ar.add_residual(ar.allreduce(0, part_moe, jnp.bool_(True)))
        ar.drain_sends()
        O["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            O["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, H), BF16)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
    ] + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names]
    call = pl.pallas_call(
        pallas_body,
        out_shape=[jax.ShapeDtypeStruct(*osh[k]) for k in p.o_names],
        in_specs=in_specs,
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM) for _ in p.o_names],
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=ex.interpret,
    )
    mesh = Mesh(devs[:R], ("tp",))
    step = jax.jit(
        jax.shard_map(
            lambda pos_in, hid_in, *ws: call(pos_in, hid_in[0], *[w[0] for w in ws])[0][
                None
            ],
            mesh=mesh,
            in_specs=(P(), P("tp"), *(P("tp") for _ in p.w_names)),
            out_specs=P("tp"),
            check_vma=False,
        )
    )
    pos = jnp.arange(1, B + 1, dtype=I32)
    hid_rep = jnp.broadcast_to(hid[None, ...], (R, ex.M, H))
    out = jax.block_until_ready(step(pos, hid_rep, *[w_by_name[k] for k in p.w_names]))
    # Routed-row count of the hottest expert (router on the raw rows; the norm only
    # rescales rows / channels by positive factors close to 1).
    logits = hid[:B].astype(F32) @ router.astype(F32)
    top = jnp.argsort(-logits, axis=-1)[:, :TOPK]
    hot_rows = int(jnp.max(jnp.bincount(top.reshape(-1), length=E)))
    return np.asarray(out[:, :B]).copy(), mp, hot_rows


def _compare(B: int, hot: bool = False, direct0: bool = False) -> int:
    """Dense vs row-bucketed outputs (this process); returns the hottest expert's rows."""
    dense, mp0, hot_rows = _run(B, False, hot, direct0)
    rb, mp1, _ = _run(B, True, hot, direct0)
    assert mp0.RB is False and mp1.RB is (B > 16)
    assert np.isfinite(dense.astype(np.float32)).all()
    np.testing.assert_array_equal(rb.view(np.uint16), dense.view(np.uint16))
    return hot_rows


def _check(B: int, hot: bool = False, direct0: bool = False) -> int:
    """`_compare` in a fresh process: every bf16 rounding kept, no FMA contraction."""
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=(
            f"{flags} --xla_allow_excess_precision=false --xla_cpu_max_isa=AVX"
        ).strip(),
    )
    code = (
        "from tests.megakernel.test_moe_rowbucket import _compare; "
        f"print('HOT_ROWS', _compare({B}, {hot}, {direct0}))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        timeout=1800,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-6000:]
    return int(r.stdout.split("HOT_ROWS")[-1].split()[0])


def test_row_bucket_resolution():
    """RB resolves only on PIPE_EP above one 16-row block; chunks tile MB."""
    g = FakeGeometry(
        L=1, att_every=1, H=H, NH=2, NKV=1, HD=128, E=E, TOPK=TOPK, I=IR,
        IS=ISR * R, moe_tp=False, e_slots=0,
    )
    moe = MoEBlock()
    ms = ModelSpec("moe_rb", config=g, layer_type=LayerType("lt", slots=(moe,)))
    for B, pipe_ep, want, chunk in [
        (16, True, False, None),
        (32, True, True, 32),
        (48, True, True, 16),
        (64, True, True, 32),
        (96, True, True, 32),
        (64, False, False, None),
    ]:
        tiles = dataclasses.replace(
            TINY_TILES, bank=512 << 10, nbd=8, nbe=6, ebank=32 << 10
        ).with_blocks(
            {"moe": {"pipe_ep": pipe_ep, "moe_unroll": pipe_ep, "row_bucket": True}}
        )
        ex = ExecSpec(B=B, T=1, R=R, ctx=128, stack_e=True, tiles=tiles)
        cp, scratch = vmem.build_scratch(ms, ex)
        mp = cp.block("moe")
        assert mp.RB is want, (B, pipe_ep)
        assert ("rbcs" in scratch) is want
        if want:
            assert mp.rb_small == 16 and mp.rb_chunk == chunk
            assert ex.MB % mp.rb_chunk == 0


@pytest.mark.slow
@pytest.mark.parametrize("B", [32, 64, 96])
def test_row_bucket_bitwise_vs_dense(B: int):
    """c32 decode sizes: experts hold 1..~15 rows (16-row block, a few 32-row chunks)."""
    _check(B)


@pytest.mark.slow
def test_row_bucket_b16_is_dense():
    """b16: RB stays off (one 16-row block is the dense loop); outputs unchanged."""
    _check(16)


@pytest.mark.slow
@pytest.mark.parametrize("B,direct0", [(64, False), (96, True), (48, False)])
def test_row_bucket_hot_expert_bitwise(B: int, direct0: bool):
    """One expert routed by (almost) every row: the chunked path (2-3 chunks)."""
    hot_rows = _check(B, hot=True, direct0=direct0)
    assert hot_rows > 32, hot_rows
