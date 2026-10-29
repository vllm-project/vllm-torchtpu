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
"""Pallas CPU interpret and TPU v7x zero-spill tests for `megakernel.ops`."""

from __future__ import annotations

import glob
import importlib.util
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import topologies
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import ops
from vllm_torchtpu.megakernel.ops import (
    BF16,
    F32,
    FP8,
    U8,
    U32,
    combine_bf16_lse,
    gdn_recurrence_step,
    kda_step,
    nvfp4_dot_subtiled,
    online_softmax_lse_merge,
    online_softmax_stats_merge,
    split_bf16_lse,
    unpack_e8m0_scale,
    unpack_mxfp4_u32_bf16,
    unpack_mxfp4_u32_fp8,
    unpack_nvfp4_tile,
    unpack_nvfp4_tile_fp8,
)


def test_ops_backwards_compatible_reexports() -> None:
    """All legacy `ops.py` symbols remain importable from `megakernel.ops`."""
    expected = {
        "BF16",
        "F32",
        "I32",
        "U32",
        "U8",
        "FP8",
        "rup",
        "rnd",
        "bswitch",
        "rmsnorm",
        "rms_zc",
        "rope_partial",
        "silu_mul",
        "scope_fn",
    }
    assert expected <= set(dir(ops))


def _pack_nvfp4_host(
    w_f4: jax.Array, bscale_f8: jax.Array
) -> tuple[jax.Array, np.ndarray]:
    """Pack `[2 * tk_raw, n]` float4 + `[tk_raw // 8, n]` float8 scales into `uint8`."""
    k_logical, n_logical = w_f4.shape
    tk_raw = k_logical // 2
    w_pairs = w_f4.reshape(tk_raw, 2, n_logical).transpose(0, 2, 1)
    packed_u8 = lax.bitcast_convert_type(w_pairs, U8)
    bs_u8 = lax.bitcast_convert_type(bscale_f8, U8).reshape(tk_raw, n_logical // 8)
    n_raw = ops.rup(n_logical + n_logical // 8, 128)
    pad = jnp.zeros((tk_raw, n_raw - n_logical - n_logical // 8), dtype=U8)
    tile_u8 = jnp.concatenate([packed_u8, bs_u8, pad], axis=-1)
    ref = (
        np.asarray(w_f4.astype(F32), dtype=np.float64).reshape(
            tk_raw // 8, 16, n_logical
        )
        * np.asarray(bscale_f8.astype(F32), dtype=np.float64)[:, None, :]
    ).reshape(k_logical, n_logical)
    return tile_u8, ref


def test_nvfp4_pallas_subtiled_dot_parity() -> None:
    """Verify `unpack_nvfp4_tile*` and `nvfp4_dot_subtiled` on MoE expert geometry."""
    tk_raw, n_logical, m_rows = 256, 256, 16
    k_logical = 2 * tk_raw
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(5), 3)
    w_f4 = jax.random.uniform(
        k0, (k_logical, n_logical), minval=-3.0, maxval=3.0
    ).astype(jnp.float4_e2m1fn)
    bs_f8 = jax.random.uniform(
        k1, (tk_raw // 8, n_logical), minval=0.125, maxval=2.0
    ).astype(FP8)
    tile_u8, w_ref_f64 = _pack_nvfp4_host(w_f4, bs_f8)
    x = jax.random.normal(k2, (m_rows, k_logical), dtype=BF16)

    def kernel(x_ref, w_ref, w_out_ref, fp8_deq_ref, dot_out_ref):
        w_bf16 = unpack_nvfp4_tile(w_ref[...], n_logical, interpret=True)
        w_fp8, w_s = unpack_nvfp4_tile_fp8(w_ref[...], n_logical, interpret=True)
        w_out_ref[...] = w_bf16
        fp8_deq_ref[...] = w_fp8.astype(F32) * w_s
        dot_out_ref[...] = nvfp4_dot_subtiled(
            x_ref, w_ref, n_logical, subtile_raw_rows=64, interpret=True
        )

    w_got, fp8_deq, dot_got = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((k_logical, n_logical), BF16),
            jax.ShapeDtypeStruct((k_logical, n_logical), F32),
            jax.ShapeDtypeStruct((m_rows, n_logical), F32),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=tuple(pl.BlockSpec(memory_space=pltpu.VMEM) for _ in range(3)),
        interpret=pltpu.InterpretParams(),
    )(x, tile_u8)

    u_bf16, u_fp8 = 2.0**-8, 2.0**-3
    assert np.max(np.abs(np.asarray(w_got, dtype=np.float64) - w_ref_f64)) <= (
        u_bf16 * np.max(np.abs(w_ref_f64))
    )
    assert np.max(np.abs(np.asarray(fp8_deq, dtype=np.float64) - w_ref_f64)) <= (
        (u_fp8 + u_bf16) * np.max(np.abs(w_ref_f64))
    )
    dot_ref = np.asarray(x, dtype=np.float64) @ np.asarray(w_got, dtype=np.float64)
    row_l1 = np.max(np.sum(np.abs(np.asarray(x, dtype=np.float64)), axis=-1))
    assert np.max(np.abs(np.asarray(dot_got, dtype=np.float64) - dot_ref)) <= (
        4.0 * (2.0**-24) * row_l1 * np.max(np.abs(w_ref_f64))
    )
    sub_w, _ = unpack_nvfp4_tile_fp8(tile_u8[:64], n_logical, interpret=True)
    print(
        f"nvfp4_subtile_fp8_vregs: {(sub_w.size * sub_w.dtype.itemsize) // 4096} vreg"
    )


def test_mxfp4_u32_e8m0_pallas_unpack_parity() -> None:
    """Verify `unpack_e8m0_scale` and `unpack_mxfp4_u32_*` on K3 expert geometry."""
    k_logical, n_logical, group_size = 512, 128, 32
    k_u32 = k_logical // 8
    k_groups = k_logical // group_size
    k0, k1 = jax.random.split(jax.random.PRNGKey(11))
    w_f4 = jax.random.uniform(
        k0, (k_logical, n_logical), minval=-3.0, maxval=3.0
    ).astype(jnp.float4_e2m1fn)
    packed_u32 = lax.bitcast_convert_type(
        w_f4.reshape(k_u32, 8, n_logical).transpose(0, 2, 1), U32
    )
    exp_u8 = jax.random.randint(
        k1, (k_groups, n_logical), minval=124, maxval=130, dtype=U8
    )

    def kernel(p_ref, s_ref, bf16_ref, fp8_ref):
        bf16_ref[...] = unpack_mxfp4_u32_bf16(
            p_ref[...], s_ref[...], group_size=group_size, interpret=True
        )
        w_fp8, col_s = unpack_mxfp4_u32_fp8(
            p_ref[...], s_ref[...], group_size=group_size, interpret=True
        )
        fp8_ref[...] = w_fp8.astype(F32) * col_s

    got_bf16, got_fp8 = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((k_logical, n_logical), BF16),
            jax.ShapeDtypeStruct((k_logical, n_logical), F32),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        interpret=pltpu.InterpretParams(),
    )(packed_u32, exp_u8)

    scales_ref = np.ldexp(1.0, np.asarray(exp_u8, dtype=np.int32) - 127)
    np.testing.assert_array_equal(
        np.asarray(unpack_e8m0_scale(exp_u8), dtype=np.float64), scales_ref
    )
    w_ref = (
        np.asarray(w_f4.astype(F32), dtype=np.float64).reshape(
            k_groups, group_size, n_logical
        )
        * scales_ref[:, None, :]
    ).reshape(k_logical, n_logical)
    assert np.max(np.abs(np.asarray(got_bf16, dtype=np.float64) - w_ref)) == 0.0
    assert np.max(np.abs(np.asarray(got_fp8, dtype=np.float64) - w_ref)) == 0.0


def test_gdn_and_kda_pallas_steps_against_f64() -> None:
    """Verify `gdn_recurrence_step` (softplus) and `kda_step` (bounded key) vs F64."""
    kh, vh, kd, vd = 2, 4, 128, 128
    keys = jax.random.split(jax.random.PRNGKey(23), 9)
    st = jax.random.normal(keys[0], (vh, kd, vd), dtype=BF16) * 0.25
    q = jax.random.normal(keys[1], (kh, kd), dtype=F32)
    k = jax.random.normal(keys[2], (kh, kd), dtype=F32)
    v = jax.random.normal(keys[3], (vh, vd), dtype=F32)
    beta_raw = jax.random.normal(keys[4], (vh,), dtype=F32)
    a_head = jax.random.normal(keys[5], (vh,), dtype=F32)
    alog_head = jax.random.uniform(keys[6], (vh,), minval=-2.0, maxval=0.5, dtype=F32)
    dtb_head = jax.random.normal(keys[7], (vh,), dtype=F32) * 0.1
    gnw = jnp.ones((vd,), dtype=BF16)
    z_raw = jax.random.normal(keys[8], (vh, vd), dtype=F32)

    def gdn_kernel(
        s_ref,
        q_ref,
        k_ref,
        v_ref,
        b_ref,
        a_ref,
        al_ref,
        dt_ref,
        g_ref,
        z_ref,
        ns_ref,
        o_ref,
    ):
        ns, out = gdn_recurrence_step(
            s_ref[...],
            q_ref[...],
            k_ref[...],
            v_ref[...],
            b_ref[...],
            a_ref[...],
            al_ref[...],
            dt_ref[...],
            gnorm_weight=g_ref[...],
            z_raw=z_ref[...],
        )
        ns_ref[...] = ns
        o_ref[...] = out

    ns_gdn, out_gdn = pl.pallas_call(
        gdn_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((vh, kd, vd), BF16),
            jax.ShapeDtypeStruct((vh, vd), F32),
        ),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM) for _ in range(10)],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        interpret=pltpu.InterpretParams(),
    )(st, q, k, v, beta_raw, a_head, alog_head, dtb_head, gnw, z_raw)

    qf = np.asarray(q, dtype=np.float64)
    kf = np.asarray(k, dtype=np.float64)
    qf = np.repeat(
        qf / np.sqrt(np.sum(qf * qf, axis=-1, keepdims=True) + 1e-6) * (kd**-0.5),
        vh // kh,
        axis=0,
    )
    kf = np.repeat(
        kf / np.sqrt(np.sum(kf * kf, axis=-1, keepdims=True) + 1e-6), vh // kh, axis=0
    )
    beta_f = 1.0 / (1.0 + np.exp(-np.asarray(beta_raw, dtype=np.float64)))
    g_in = np.asarray(a_head, dtype=np.float64) + np.asarray(dtb_head, dtype=np.float64)
    eg_f = np.exp(
        -np.exp(np.asarray(alog_head, dtype=np.float64)) * np.log1p(np.exp(g_in))
    )[:, None, None]
    s_f = np.asarray(st, dtype=np.float64) * eg_f
    pred_f = np.sum(s_f * kf[:, :, None], axis=1)
    s_next_f = (
        s_f
        + kf[:, :, None]
        * ((np.asarray(v, dtype=np.float64) - pred_f) * beta_f[:, None])[:, None, :]
    )
    out_raw_f = np.sum(s_next_f * qf[:, :, None], axis=1)
    norm_f = out_raw_f / np.sqrt(
        np.mean(out_raw_f * out_raw_f, axis=-1, keepdims=True) + 1e-6
    )
    zf = np.asarray(z_raw, dtype=np.float64)
    out_ref_f = norm_f * (zf / (1.0 + np.exp(-zf)))
    u_bf16 = 2.0**-8
    tol_bound = 8.0 * u_bf16 * max(1.0, float(np.max(np.abs(out_ref_f))))
    assert np.max(np.abs(np.asarray(ns_gdn, dtype=np.float64) - s_next_f)) <= tol_bound
    assert (
        np.max(np.abs(np.asarray(out_gdn, dtype=np.float64) - out_ref_f)) <= tol_bound
    )

    a_key = jax.random.normal(keys[5], (vh * kd,), dtype=F32)
    alog_key = jax.random.uniform(
        keys[6], (vh * kd,), minval=-2.0, maxval=0.5, dtype=F32
    )
    dtb_key = jax.random.normal(keys[7], (vh * kd,), dtype=F32) * 0.1
    ns_kda, out_kda = kda_step(
        st, q, k, v, beta_raw, a_key, alog_key, dtb_key, gnorm_weight=gnw, z_raw=z_raw
    )
    gk_in = np.asarray(a_key, dtype=np.float64) + np.asarray(dtb_key, dtype=np.float64)
    rk = np.exp(np.asarray(alog_key, dtype=np.float64))
    eg_k = np.exp((-8.0 / (1.0 + np.exp(-(rk * gk_in)))).reshape(vh, kd))[:, :, None]
    sk_f = np.asarray(st, dtype=np.float64) * eg_k
    pk_f = np.sum(sk_f * kf[:, :, None], axis=1)
    sk_next = (
        sk_f
        + kf[:, :, None]
        * ((np.asarray(v, dtype=np.float64) - pk_f) * beta_f[:, None])[:, None, :]
    )
    ok_raw = np.sum(sk_next * qf[:, :, None], axis=1)
    nk_f = ok_raw / np.sqrt(np.mean(ok_raw * ok_raw, axis=-1, keepdims=True) + 1e-6)
    ok_ref = nk_f * (1.0 / (1.0 + np.exp(-zf)))
    tol_kda = 8.0 * u_bf16 * max(1.0, float(np.max(np.abs(ok_ref))))
    assert np.max(np.abs(np.asarray(ns_kda, dtype=np.float64) - sk_next)) <= tol_kda
    assert np.max(np.abs(np.asarray(out_kda, dtype=np.float64) - ok_ref)) <= tol_kda


def test_online_softmax_lse_and_stats_merge_with_empty_rows() -> None:
    """Verify split-BF16 MLA LSE merge, GQA `(acc, m, l)` merge, and `-inf` safety."""
    G, rows, tok_per_peer, dim = 8, 8, 16, 128
    k0, k1 = jax.random.split(jax.random.PRNGKey(31))
    scores = np.asarray(
        jax.random.normal(k0, (G, rows, tok_per_peer), dtype=F32), dtype=np.float64
    )
    vals = np.asarray(
        jax.random.normal(k1, (G, rows, tok_per_peer, dim), dtype=F32), dtype=np.float64
    )
    scores[:, -1, :] = -np.inf  # Last row has 0 valid KV tokens across all peers

    part_outs, lse_his, lse_los, accs, maxs, sums = [], [], [], [], [], []
    for g in range(G):
        sg, vg = scores[g], vals[g]
        mg = np.max(sg, axis=-1, keepdims=True)
        safe_mg = np.where(np.isfinite(mg), mg, 0.0)
        wg = np.where(np.isfinite(sg), np.exp(sg - safe_mg), 0.0)
        lg = np.sum(wg, axis=-1, keepdims=True)
        ag = np.einsum("rt,rtd->rd", wg, vg)
        og = np.where(lg > 0.0, ag / np.where(lg > 0.0, lg, 1.0), 0.0)
        lseg = np.where(lg > 0.0, mg + np.log(np.maximum(lg, 1e-38)), -np.inf)
        hi, lo = split_bf16_lse(jnp.asarray(lseg, dtype=F32))
        part_outs.append(jnp.asarray(og, dtype=F32))
        lse_his.append(hi)
        lse_los.append(lo)
        accs.append(jnp.asarray(ag, dtype=F32))
        maxs.append(jnp.asarray(mg, dtype=F32))
        sums.append(jnp.asarray(lg, dtype=F32))

    recombined = combine_bf16_lse(lse_his[0], lse_los[0])
    assert jnp.all(jnp.isneginf(recombined[-1]))

    merged_mla, merged_lse = online_softmax_lse_merge(
        jnp.stack(part_outs, axis=0),
        jnp.stack(lse_his, axis=0),
        lse_lo=jnp.stack(lse_los, axis=0),
    )
    merged_gqa, _, _ = online_softmax_stats_merge(accs, maxs, sums, normalize=True)

    all_s = np.transpose(scores, (1, 0, 2)).reshape(rows, G * tok_per_peer)
    all_v = np.transpose(vals, (1, 0, 2, 3)).reshape(rows, G * tok_per_peer, dim)
    m_glob = np.max(all_s, axis=-1, keepdims=True)
    safe_glob = np.where(np.isfinite(m_glob), m_glob, 0.0)
    w_glob = np.where(np.isfinite(all_s), np.exp(all_s - safe_glob), 0.0)
    l_glob = np.sum(w_glob, axis=-1, keepdims=True)
    ref_out = np.where(
        l_glob > 0.0,
        np.einsum("rt,rtd->rd", w_glob, all_v) / np.where(l_glob > 0.0, l_glob, 1.0),
        0.0,
    )

    assert not np.any(np.isnan(np.asarray(merged_mla)))
    assert not np.any(np.isnan(np.asarray(merged_gqa)))
    assert np.all(np.asarray(merged_mla[-1]) == 0.0)
    assert np.isneginf(float(merged_lse[-1, 0]))
    u_bf16, u_f32 = 2.0**-8, 2.0**-24
    tol_mla = (
        G
        * (u_bf16**2)
        * max(1.0, float(np.max(np.abs(all_s[:-1]))))
        * max(1.0, float(np.max(np.abs(ref_out[:-1]))))
    )
    tol_gqa = 2.0 * G * u_f32 * max(1.0, float(np.max(np.abs(ref_out))))
    assert (
        np.max(np.abs(np.asarray(merged_mla[:-1], dtype=np.float64) - ref_out[:-1]))
        <= tol_mla
    )
    assert np.max(np.abs(np.asarray(merged_gqa, dtype=np.float64) - ref_out)) <= tol_gqa


def _run_llo_zero_spill_child(llo_dir: str) -> None:
    """Compile combined Layer 1 kernel for TPU v7x and assert 0 VREG spills."""
    if jax.default_backend() == "tpu":
        tpu_devs = np.asarray(jax.devices("tpu")[:1])
    else:
        topo = topologies.get_topology_desc(topology_name="tpu7x_2x2x1")
        tpu_devs = np.asarray(topo.devices)[:1]
    mesh = Mesh(tpu_devs, ("tp",))

    def kernel(
        x_ref,
        w_ref,
        mx_w_ref,
        mx_s_ref,
        st_ref,
        q_ref,
        k_ref,
        v_ref,
        b_ref,
        a_ref,
        al_ref,
        dt_ref,
        part_ref,
        lse_ref,
        o_ref,
        ns_ref,
    ):
        dot = nvfp4_dot_subtiled(
            x_ref, w_ref, 256, subtile_raw_rows=64, interpret=False
        )
        mx_fp8, mx_s = unpack_mxfp4_u32_fp8(
            mx_w_ref[...], mx_s_ref[...], group_size=32, interpret=False
        )
        mx_dot = (
            lax.dot_general(
                x_ref[:, :256].astype(FP8),
                mx_fp8,
                (((1,), (0,)), ((), ())),
                preferred_element_type=F32,
            )
            * mx_s
        )
        ns, rec_out = kda_step(
            st_ref[...],
            q_ref[...][:1, :],
            k_ref[...][:1, :],
            v_ref[...][:1, :],
            b_ref[...][0, :1],
            a_ref[...][:1, :],
            al_ref[...][:1, :],
            dt_ref[...][:1, :],
        )
        lse_tile = lse_ref[...]
        hi0, lo0 = split_bf16_lse(lse_tile[0, :, 0])
        hi1, lo1 = split_bf16_lse(lse_tile[1, :, 0])
        merged, _ = online_softmax_lse_merge(
            (part_ref[0], part_ref[1]), (hi0, hi1), lse_lo=(lo0, lo1)
        )
        ns_ref[...] = ns
        rec_pad = jnp.concatenate([rec_out, jnp.zeros((7, 128), F32)], axis=0)
        o_ref[...] = dot[:8, :128] + mx_dot[:8, :] + merged + rec_pad

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8, 128), F32),
            jax.ShapeDtypeStruct((1, 128, 128), BF16),
        ),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM) for _ in range(14)],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        name="layer1_substrate_probe",
    )
    fn = jax.jit(
        jax.shard_map(
            lambda *xs: tuple(y[None] for y in call(*(x[0] for x in xs))),
            mesh=mesh,
            in_specs=tuple(P("tp") for _ in range(14)),
            out_specs=(P("tp"), P("tp")),
            check_vma=False,
        )
    )
    compiled = fn.lower(
        jax.ShapeDtypeStruct((1, 16, 512), BF16),
        jax.ShapeDtypeStruct((1, 256, 384), U8),
        jax.ShapeDtypeStruct((1, 32, 128), U32),
        jax.ShapeDtypeStruct((1, 8, 128), U8),
        jax.ShapeDtypeStruct((1, 1, 128, 128), BF16),
        jax.ShapeDtypeStruct((1, 8, 128), F32),
        jax.ShapeDtypeStruct((1, 8, 128), F32),
        jax.ShapeDtypeStruct((1, 8, 128), F32),
        jax.ShapeDtypeStruct((1, 8, 128), F32),
        jax.ShapeDtypeStruct((1, 8, 128), F32),
        jax.ShapeDtypeStruct((1, 8, 128), F32),
        jax.ShapeDtypeStruct((1, 8, 128), F32),
        jax.ShapeDtypeStruct((1, 2, 8, 128), F32),
        jax.ShapeDtypeStruct((1, 2, 8, 128), F32),
    ).compile()
    assert compiled is not None
    bundles = [
        p
        for p in glob.glob(os.path.join(llo_dir, "*-final_bundles.txt"))
        if "layer1_substrate_probe" in os.path.basename(p)
        and "schedule-analysis" not in os.path.basename(p)
        and "-TLP-" not in os.path.basename(p)
    ]
    assert bundles, f"no *-final_bundles.txt for layer1_substrate_probe in {llo_dir}"
    for bf in bundles:
        with open(bf, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                assert "spill" not in line.lower(), f"VREG spill in {bf}: {line}"
    print(f"layer1_substrate_probe_spills: 0 spills ({len(bundles)} bundles)")


@pytest.mark.tpu_test
def test_tpu_v7x_layer1_substrate_zero_vreg_spills(tmp_path) -> None:
    """Compile Layer 1 kernels for TPU v7x and assert 0 post-RA VREG spills."""
    has_libtpu = (
        jax.default_backend() == "tpu"
        or os.path.isfile(os.environ.get("TPU_LIBRARY_PATH", ""))
        or importlib.util.find_spec("libtpu") is not None
    )
    if not has_libtpu and os.environ.get("VLLM_REQUIRE_LLO_DUMP") != "1":
        pytest.skip("requires TPU backend or libtpu AOT compiler")
    llo_dir = tmp_path / "llo"
    llo_dir.mkdir(parents=True, exist_ok=True)
    llo_flags = (
        f"--xla_jf_dump_to={llo_dir} --xla_jf_dump_hlo_text=false "
        "--xla_jf_dump_llo_proto=false "
        "--xla_jf_dump_category_filter=codegen,ra,-utilization"
    )
    env = dict(os.environ, LIBTPU_INIT_ARGS=llo_flags)
    out = subprocess.run(
        [sys.executable or "python3", __file__, "_llo_zero_spill", str(llo_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
    assert "0 spills" in out.stdout
    print(out.stdout, end="")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_llo_zero_spill":
        _run_llo_zero_spill_child(sys.argv[2])
