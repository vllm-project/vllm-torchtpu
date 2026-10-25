# SPDX-License-Identifier: Apache-2.0
"""ix-unified-port: VLLM_TPU_QWEN35_MK_SH_AT (hostov stack a83d4ad9f / e4244f756 / 6aa2cabb8).

SH_AT moves the megakernel's shared expert (SH_PRE path) into route()'s hit-list copy (1) or
after the first expert DMAs (2). Unset keeps config.SH_AT = 0 (the unified kernel). The knob
applies only to buckets of at most VLLM_TPU_QWEN35_MK_SH_AT_MAX_ROWS rows (default 8). With it
on, the MoE block's output must be bit-identical to SH_AT=0 and to the pure-JAX reference, and
the traced kernel must actually differ (otherwise the identity check proves nothing)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))

from qwen35_megakernel_test_utils import (  # noqa: E402
    make_test_ctx_and_inputs,
    ref_dmatmul,
    rms_ref,
    setup_8_cpu_devices,
)
from vllm_torchtpu.kernels.qwen35_megakernel import MoEBlock, WeightRings  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel import config as mk_config  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel.config import TINY, TINY_TILES  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel.context import MKBuildContext  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel.megakernel import BF16, F32, I32, rnd  # noqa: E402


def _ctx(rows):
    return MKBuildContext(dataclasses.replace(TINY, B=rows), TINY_TILES, interpret=True)


@pytest.mark.parametrize("rows,expect", [(4, 2), (8, 2), (16, 0)])
def test_sh_at_env_only_on_small_buckets(monkeypatch, rows, expect):
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_SH_AT", "2")
    monkeypatch.delenv("VLLM_TPU_QWEN35_MK_SH_AT_MAX_ROWS", raising=False)
    assert _ctx(rows).SH_AT == expect


def test_sh_at_max_rows_override(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_SH_AT", "2")
    monkeypatch.setenv("VLLM_TPU_QWEN35_MK_SH_AT_MAX_ROWS", "16")
    assert _ctx(16).SH_AT == 2


def test_sh_at_unset_keeps_config_default(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_QWEN35_MK_SH_AT", raising=False)
    assert int(mk_config.SH_AT) == 0
    assert _ctx(4).SH_AT == 0


# ---- bit identity of the MoE block under SH_AT 0 / 1 / 2 (interpret mode, SH_PRE engaged) ----


def _lazy_setup(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_QWEN35_MK_SH_AT", raising=False)
    c, tiles, _, w_sh, st_sh, lay, w_vals, st_vals, pos, hid = make_test_ctx_and_inputs()
    ctx = MKBuildContext(c, dataclasses.replace(tiles, lazy=True), interpret=True)
    assert ctx.LAZY_MOE and ctx.nt["sh13"] + ctx.nt["sh2"] <= 4, "SH_PRE path not engaged"
    return c, ctx, w_sh, st_sh, lay, w_vals, st_vals, pos, hid


def _moe_call(ctx, w_sh, st_sh, w_vals, st_vals, pos, hid, sh_at):
    ctx.SH_AT = sh_at
    seg_r = ctx.nt["lin_in"] + ctx.nt["lin_out"]

    def moe_kernel(*refs):
        ctx.bind_refs(refs)
        rings = WeightRings(ctx)
        moe_blk = MoEBlock(ctx, rings)
        scr = ctx.scr
        scr["hitv"][...] = jnp.zeros((ctx.MR, 128), I32)
        zero_seen = pltpu.make_async_copy(scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"])
        zero_seen.start()
        zero_seen.wait()
        for i in range(ctx.N_MOE):
            rings.fetch(jnp.int32(seg_r + i))
        cp_sm = pltpu.make_async_copy(ctx.W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0])
        cp_sm.start()
        cp_sm.wait()
        scr["res"][...] = jnp.where(ctx.live, ctx.hid_ref[: ctx.MB].astype(F32), 0.0)
        part = moe_blk(jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False))
        ctx.out_ref[: ctx.MB] = part.astype(BF16)

    call = pl.pallas_call(
        moe_kernel,
        out_shape=ctx.out_shape,
        in_specs=ctx.in_specs,
        out_specs=ctx.out_specs,
        scratch_shapes=[ctx.scratch[k] for k in ctx.s_names],
        interpret=pltpu.InterpretParams(),
    )
    args = (pos, hid, *[w_vals[k] for k in w_sh], *[st_vals[k] for k in st_sh])
    mesh1 = Mesh(np.array(setup_8_cpu_devices()[:1]), ("tp",))
    f = jax.shard_map(
        lambda *a: call(*a),
        mesh=mesh1,
        in_specs=tuple(P() for _ in args),
        out_specs=tuple(P() for _ in ctx.out_shape),
        check_vma=False,
    )
    return f, args


def _reference(c, ctx, lay, w_vals, hid):
    def sm_val(l_idx, name, width=None):
        o, n = lay[name]
        w = n if width is None else width
        return w_vals["smalls"][l_idx, :, o : o + w]

    x_moe = rms_ref(hid[: c.B].astype(F32), sm_val(0, "post_norm"), c.eps)
    xb_moe = x_moe.astype(BF16)
    logits = ref_dmatmul(xb_moe, w_vals["router"][0], ctx.MB)
    probs = jax.nn.softmax(logits, axis=-1)
    top_vals, top_ids = lax.top_k(probs, c.TOPK)
    wn = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)
    sg = jax.nn.sigmoid(jnp.sum(x_moe * sm_val(0, "sh_gate"), axis=-1, keepdims=True))
    g13 = ref_dmatmul(xb_moe, w_vals["sh13"][0], ctx.MB) * sm_val(0, "sh13_s")
    mid_sh = rnd(rnd(jax.nn.silu(rnd(g13[:, : c.ISR]))) * rnd(g13[:, c.ISR :])).astype(BF16)
    part_ref = ref_dmatmul(mid_sh, w_vals["sh2"][0], ctx.MB) * sm_val(0, "sh2_s") * sg
    for b in range(c.B):
        for k_i in range(c.TOPK):
            e = int(top_ids[b, k_i])
            es_e = w_vals["es"][0, e, 0]
            h_e = ref_dmatmul(xb_moe[b : b + 1], w_vals["e13"][0, e], ctx.MB) * es_e[: 2 * c.IR]
            mid_e = rnd(rnd(jax.nn.silu(rnd(h_e[:, : c.IR]))) * rnd(h_e[:, c.IR :])).astype(BF16)
            y_e = rnd(ref_dmatmul(mid_e, w_vals["e2"][0, e], ctx.MB) * es_e[2 * c.IR :])
            part_ref = part_ref.at[b : b + 1].add(wn[b, k_i] * y_e)
    return part_ref.astype(BF16)


@pytest.fixture(scope="module")
def moe_outputs():
    mp = pytest.MonkeyPatch()
    try:
        c, ctx, w_sh, st_sh, lay, w_vals, st_vals, pos, hid = _lazy_setup(mp)
        outs, jaxprs = {}, {}
        for v in (0, 1, 2):
            f, args = _moe_call(ctx, w_sh, st_sh, w_vals, st_vals, pos, hid, v)
            jaxprs[v] = str(jax.make_jaxpr(f)(*args))
            outs[v] = np.asarray(f(*args)[0][: c.B].astype(F32))
        ref = np.asarray(_reference(c, ctx, lay, w_vals, hid).astype(F32))
        return outs, jaxprs, ref
    finally:
        mp.undo()


def test_sh_at_0_matches_reference(moe_outputs):
    outs, _, ref = moe_outputs
    assert np.array_equal(outs[0], ref)


@pytest.mark.parametrize("v", [1, 2])
def test_sh_at_bit_identical_to_0(moe_outputs, v):
    outs, _, ref = moe_outputs
    assert np.array_equal(outs[v], outs[0])
    assert np.array_equal(outs[v], ref)


@pytest.mark.parametrize("v", [1, 2])
def test_sh_at_changes_the_trace(moe_outputs, v):
    _, jaxprs, _ = moe_outputs
    assert jaxprs[v] != jaxprs[0]
