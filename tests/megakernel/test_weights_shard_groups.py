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
"""Unit tests for rank-group weight sharing (`shard_groups` and `split_groups`)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.comm.replication import (
    derive_kh_rep,
    derive_sh_rep,
    replicate_groups,
    shard_count,
    split_sharded_width,
)
from vllm_torchtpu.megakernel.decls import HFParamDecl, ParamTransform, VLLMAlias
from vllm_torchtpu.megakernel.modules.transforms import (
    _shard_and_transform_single,
    apply_hf_transforms,
)
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8
from vllm_torchtpu.megakernel.weights.ops import (
    JaxArrayOps,
    slice_alias_targets,
    target_axis_widths,
)


def test_target_axis_widths_and_slice_alias_targets_with_shard_groups():
    """Verify `shard_groups < R` shares slices across consecutive ranks."""
    decl = HFParamDecl(
        name="sh13",
        hf_name="mlp.shared_expert.gate_up_proj.weight",
        shape=(64, 2048),
        dtype=BF16,
        transform=ParamTransform(shard_axis=-1, shard_groups=16),
    )
    assert target_axis_widths(decl, R=32, split_axis=-1) == (2048, 128)
    assert target_axis_widths(decl, R=16, split_axis=-1) == (2048, 128)
    assert target_axis_widths(decl, R=8, split_axis=-1) == (2048, 256)

    ops = JaxArrayOps()
    alias = VLLMAlias(
        source="shared_gate_up",
        targets=("sh13",),
        shard_axis=-1,
        already_sharded=False,
        already_fp8=False,
    )
    w_buf = jnp.arange(64 * 2048, dtype=jnp.float32).reshape(64, 2048).astype(BF16)
    slices = [
        slice_alias_targets(
            ops,
            alias,
            w_buf,
            None,
            (decl,),
            {},
            R=32,
            tp_rank=r,
        )["sh13"]
        for r in range(32)
    ]
    assert slices[0].shape == (64, 128)
    for g_idx in range(16):
        assert jnp.array_equal(slices[2 * g_idx], slices[2 * g_idx + 1])
        if g_idx + 1 < 16:
            assert not jnp.array_equal(slices[2 * g_idx], slices[2 * (g_idx + 1)])
    assert jnp.array_equal(slices[0], slices[1])
    assert jnp.array_equal(slices[2], slices[3])
    assert jnp.array_equal(slices[4], slices[5])
    # Negative control: distinct rank groups receive distinct column slices.
    assert not jnp.array_equal(slices[0], slices[2])
    assert not jnp.array_equal(slices[2], slices[4])


def test_shard_and_transform_single_with_shard_groups_and_split_groups():
    """Verify grouped shards and mixed `split_groups` channel splits."""
    decl_grouped = HFParamDecl(
        name="sh13",
        hf_name="sh13",
        shape=(16, 2048),
        dtype=BF16,
        transform=ParamTransform(shard_axis=-1, shard_groups=16),
    )
    arr = jnp.arange(2 * 16 * 2048, dtype=jnp.float32).reshape(2, 16, 2048).astype(BF16)
    out = _shard_and_transform_single(arr, decl_grouped, R=32, known_names={"sh13"})
    assert out.shape == (32, 2, 16, 128)
    assert jnp.array_equal(out[0], out[1])
    assert jnp.array_equal(out[30], out[31])
    assert not jnp.array_equal(out[0], out[2])

    decl_splits = HFParamDecl(
        name="lin_in",
        hf_name="lin_in",
        shape=(16, 1536),
        dtype=BF16,
        transform=ParamTransform(
            channel_splits=(512, 1024),
            split_groups=(16, None),
        ),
    )
    arr_sp = (
        jnp.arange(2 * 16 * 1536, dtype=jnp.float32).reshape(2, 16, 1536).astype(BF16)
    )
    out_sp = _shard_and_transform_single(
        arr_sp, decl_splits, R=32, known_names={"lin_in"}
    )
    # Split 0: 512 // 16 = 32 (paired ranks); split 1: 1024 // 32 = 32 (per rank).
    assert out_sp.shape == (32, 2, 16, 64)
    assert jnp.array_equal(out_sp[0, ..., :32], out_sp[1, ..., :32])
    assert not jnp.array_equal(out_sp[0, ..., :32], out_sp[2, ..., :32])
    assert not jnp.array_equal(out_sp[0, ..., 32:], out_sp[1, ..., 32:])


def test_target_axis_widths_and_slice_alias_targets_with_split_groups():
    """Verify `split_groups` matches `_shard_and_transform_single` on vLLM aliases."""
    decl_conv = HFParamDecl(
        name="conv",
        hf_name="conv1d.weight",
        shape=(4, 8192),
        hf_shape=(8192, 1, 4),
        dtype=F32,
        transform=ParamTransform(
            shard_axis=-1,
            channel_splits=(2048, 2048, 4096),
            split_groups=(16, 16, None),
            flatten_tail=True,
        ),
    )
    assert target_axis_widths(decl_conv, R=32, split_axis=-1) == (8192, 384)
    assert target_axis_widths(decl_conv, R=16, split_axis=-1) == (8192, 512)
    assert target_axis_widths(decl_conv, R=8, split_axis=-1) == (8192, 1024)
    assert target_axis_widths(decl_conv, R=1, split_axis=-1) == (8192, 8192)

    raw_ch_major = jnp.arange(8192 * 4, dtype=F32).reshape(8192, 1, 4)
    arr_mk = jnp.transpose(raw_ch_major[:, 0, :], (1, 0))
    hf_out = _shard_and_transform_single(arr_mk, decl_conv, R=32, known_names={"conv"})
    assert hf_out.shape == (32, 1536)

    ops = JaxArrayOps()
    alias_sharded = VLLMAlias(
        source="conv1d.weight",
        targets=("conv",),
        shard_axis=-1,
        already_sharded=True,
        already_fp8=False,
    )
    alias_unsharded = VLLMAlias(
        source="conv1d.weight",
        targets=("conv",),
        shard_axis=-1,
        already_sharded=False,
        already_fp8=False,
    )
    for r in range(32):
        g_idx = r // 2
        q_part = raw_ch_major[g_idx * 128 : (g_idx + 1) * 128]
        k_part = raw_ch_major[2048 + g_idx * 128 : 2048 + (g_idx + 1) * 128]
        v_part = raw_ch_major[4096 + r * 128 : 4096 + (r + 1) * 128]
        rank_buf = jnp.concatenate([q_part, k_part, v_part], axis=0)
        assert rank_buf.shape == (384, 1, 4)

        got_sharded = slice_alias_targets(
            ops, alias_sharded, rank_buf, None, (decl_conv,), {}, R=32, tp_rank=r
        )["conv"]
        got_unsharded = slice_alias_targets(
            ops, alias_unsharded, raw_ch_major, None, (decl_conv,), {}, R=32, tp_rank=r
        )["conv"]
        assert got_sharded.shape == (1536,)
        assert jnp.array_equal(got_sharded, hf_out[r])
        assert jnp.array_equal(got_unsharded, hf_out[r])


def test_replication_helpers_boundary_values():
    """Verify `shard_count`, `derive_kh_rep`, `derive_sh_rep`, `replicate_groups`."""
    assert shard_count(32, None) == 32
    assert shard_count(32, 16) == 16
    assert shard_count(8, 16) == 8
    with pytest.raises(ValueError, match="positive"):
        shard_count(32, 0)
    with pytest.raises(ValueError, match="divisible"):
        shard_count(32, 6)
    assert derive_kh_rep(32, 16) == 2
    assert derive_kh_rep(8, 16) == 1
    assert derive_sh_rep(32, 2048) == 2
    assert derive_sh_rep(8, 2048) == 1
    x = jnp.arange(8, dtype=jnp.int32).reshape(4, 2)
    rep = replicate_groups(x, 2)
    assert jnp.array_equal(rep[0], x[0]) and jnp.array_equal(rep[1], x[0])
    assert jnp.array_equal(rep[2], x[2]) and jnp.array_equal(rep[3], x[2])


def test_shard_count_and_split_width_reject_invalid_groups():
    """Verify `shard_count`, `split_sharded_width`, and transforms fail closed."""
    assert shard_count(32, 1) == 1
    assert shard_count(32, 32) == 32
    assert shard_count(32, 64) == 32
    assert split_sharded_width(32, (2048, 2048, 4096), (16, 16, None)) == 384

    bad_sc_args: tuple[tuple[Any, Any], ...] = (
        (32, 0),
        (32, -4),
        (32, True),
        (True, 16),
    )
    for r_arg, g_arg in bad_sc_args:
        with pytest.raises(ValueError, match="positive int"):
            shard_count(r_arg, g_arg)
    with pytest.raises(ValueError, match="multiple of R"):
        shard_count(32, 48)
    with pytest.raises(ValueError, match="divisible"):
        shard_count(32, 6)
    with pytest.raises(ValueError, match="divisible"):
        split_sharded_width(32, (100,), (16,))

    bad_sgs: tuple[Any, ...] = (0, True)
    for bad_sg in bad_sgs:
        with pytest.raises(ValueError, match="shard_groups"):
            ParamTransform(shard_axis=-1, shard_groups=bad_sg)
    with pytest.raises(ValueError, match="channel_splits"):
        ParamTransform(split_groups=(16,))
    with pytest.raises(ValueError, match=r"len\(split_groups\)"):
        ParamTransform(channel_splits=(128, 128), split_groups=(16,))
    bad_sps: tuple[Any, ...] = ((16, -2), (16, False))
    for bad_sp in bad_sps:
        with pytest.raises(ValueError, match="split_groups entries"):
            ParamTransform(channel_splits=(128, 128), split_groups=bad_sp)


def test_multi_target_fp8_in_proj_qkvz_and_boundary_groups_all_32_ranks():
    """Verify bit-exact HF transforms == alias slicing across all 32 ranks."""
    ops = JaxArrayOps()
    r_total = 32
    h_dim = 64

    # 1. Boundary groups in (1, 32, 64) across 32 ranks and 2 outlier seeds
    for seed in (0, 42):
        rng = np.random.default_rng(seed)
        raw_np = rng.standard_normal((h_dim, 2048), dtype=np.float32)
        raw_np[0, ::64] += 12.0
        w_buf = jnp.asarray(raw_np, dtype=BF16)
        w_layers = w_buf[None, :, :]
        for grp in (1, 16, 32, 64):
            decl = HFParamDecl(
                name="sh13",
                hf_name="sh13",
                shape=(h_dim, 2048),
                dtype=BF16,
                transform=ParamTransform(shard_axis=-1, shard_groups=grp),
            )
            hf_sharded = _shard_and_transform_single(
                w_layers, decl, R=r_total, known_names={"sh13"}
            )
            alias = VLLMAlias(
                source="sh13_src",
                targets=("sh13",),
                shard_axis=-1,
                already_sharded=False,
                already_fp8=False,
            )
            for r in range(r_total):
                got = slice_alias_targets(
                    ops, alias, w_buf, None, (decl,), {}, R=r_total, tp_rank=r
                )["sh13"]
                assert np.array_equal(np.asarray(got), np.asarray(hf_sharded[r, 0]))

    # 2. Multi-target FP8 in_proj_qkvz (KH=16, VH=32) across all 32 ranks
    kq_dim, v_dim = 256, 512
    qkv_full = 2 * kq_dim + v_dim
    qkvz_alias = (
        VLLMAlias(
            source="in_proj_qkvz",
            targets=("lin_q", "lin_k", "lin_v", "lin_z"),
            scale_targets=("lin_q_s", "lin_k_s", "lin_v_s", "lin_z_s"),
            scale_source="in_proj_qkvz.weight_scale",
            shard_axis=-1,
            already_sharded=False,
            already_fp8=True,
            flipped=True,
        ),
    )

    def _mk_decl(
        name: str, out_w: int, grp: int | None, order: int
    ) -> tuple[HFParamDecl, HFParamDecl]:
        return (
            HFParamDecl(
                name=name,
                hf_name=f"{name}.weight",
                shape=(h_dim, out_w),
                hf_shape=(out_w, h_dim),
                dtype=FP8,
                transform=ParamTransform(
                    fuse_into="lin_in",
                    fuse_order=order,
                    fuse_axis=-1,
                    transpose=(1, 0),
                    quantize="fp8",
                    shard_axis=-1,
                    shard_groups=grp,
                ),
                vllm_aliases=qkvz_alias if order == 0 else (),
            ),
            HFParamDecl(
                name=f"{name}_s",
                hf_name=f"{name}.scale",
                shape=(out_w,),
                hf_shape=(out_w,),
                dtype=F32,
                transform=ParamTransform(
                    fuse_into="lin_in_s",
                    fuse_order=order,
                    fuse_axis=-1,
                    scale_for=name,
                    shard_axis=-1,
                    shard_groups=grp,
                ),
            ),
        )

    dq, dqs = _mk_decl("lin_q", kq_dim, 16, 0)
    dk, dks = _mk_decl("lin_k", kq_dim, 16, 1)
    dv, dvs = _mk_decl("lin_v", v_dim, None, 2)
    dz, dzs = _mk_decl("lin_z", v_dim, None, 3)
    t_decls = (dq, dk, dv, dz)
    s_decls = {"lin_q_s": dqs, "lin_k_s": dks, "lin_v_s": dvs, "lin_z_s": dzs}
    all_decls = [*t_decls, *s_decls.values()]

    fg = {
        "lin_in": ("lin_q", "lin_k", "lin_v", "lin_z"),
        "lin_in_s": ("lin_q_s", "lin_k_s", "lin_v_s", "lin_z_s"),
    }
    t_keys = ("lin_q", "lin_k", "lin_v", "lin_z")
    s_keys = ("lin_q_s", "lin_k_s", "lin_v_s", "lin_z_s")
    for seed in (7, 99):
        rng = np.random.default_rng(seed)
        q_mk = jnp.asarray(
            rng.standard_normal((h_dim, kq_dim), dtype=np.float32), dtype=FP8
        )
        k_mk = jnp.asarray(
            rng.standard_normal((h_dim, kq_dim), dtype=np.float32), dtype=FP8
        )
        v_mk = jnp.asarray(
            rng.standard_normal((h_dim, v_dim), dtype=np.float32), dtype=FP8
        )
        z_mk = jnp.asarray(
            rng.standard_normal((h_dim, v_dim), dtype=np.float32), dtype=FP8
        )
        qs = jnp.asarray(
            np.abs(rng.standard_normal((kq_dim,), dtype=np.float32)) + 0.05, dtype=F32
        )
        ks = jnp.asarray(
            np.abs(rng.standard_normal((kq_dim,), dtype=np.float32)) + 0.05, dtype=F32
        )
        vs = jnp.asarray(
            np.abs(rng.standard_normal((v_dim,), dtype=np.float32)) + 0.05, dtype=F32
        )
        zs = jnp.asarray(
            np.abs(rng.standard_normal((v_dim,), dtype=np.float32)) + 0.05, dtype=F32
        )

        hf_in = {
            "lin_q": q_mk[None, ...],
            "lin_k": k_mk[None, ...],
            "lin_v": v_mk[None, ...],
            "lin_z": z_mk[None, ...],
            "lin_q_s": qs[None, ...],
            "lin_k_s": ks[None, ...],
            "lin_v_s": vs[None, ...],
            "lin_z_s": zs[None, ...],
        }
        hf_out = apply_hf_transforms(all_decls, hf_in, R=r_total)

        fused_unsharded = jnp.concatenate([q_mk, k_mk, v_mk, z_mk], axis=-1)
        scale_unsharded = jnp.concatenate([qs, ks, vs, zs], axis=0)
        assert fused_unsharded.shape == (h_dim, qkv_full + v_dim)

        alias_sharded = VLLMAlias(
            source="in_proj_qkvz",
            targets=("lin_q", "lin_k", "lin_v", "lin_z"),
            scale_targets=("lin_q_s", "lin_k_s", "lin_v_s", "lin_z_s"),
            scale_source="in_proj_qkvz.weight_scale",
            shard_axis=-1,
            already_sharded=True,
            already_fp8=True,
            flipped=True,
        )
        for r in range(r_total):
            got_un = slice_alias_targets(
                ops,
                qkvz_alias[0],
                fused_unsharded,
                scale_unsharded,
                t_decls,
                s_decls,
                R=r_total,
                tp_rank=r,
                fuse_groups=fg,
            )
            g_idx = r // 2
            q_sl = slice(g_idx * 16, (g_idx + 1) * 16)
            v_sl = slice(r * 16, (r + 1) * 16)
            rank_w = jnp.concatenate(
                [q_mk[:, q_sl], k_mk[:, q_sl], v_mk[:, v_sl], z_mk[:, v_sl]], axis=-1
            )
            rank_s = jnp.concatenate([qs[q_sl], ks[q_sl], vs[v_sl], zs[v_sl]], axis=0)
            got_sh = slice_alias_targets(
                ops,
                alias_sharded,
                rank_w,
                rank_s,
                t_decls,
                s_decls,
                R=r_total,
                tp_rank=r,
                fuse_groups=fg,
            )
            got_sh_split = slice_alias_targets(
                ops,
                alias_sharded,
                rank_w,
                rank_s,
                t_decls,
                s_decls,
                R=r_total,
                tp_rank=r,
            )
            un_w = jnp.concatenate([got_un[k] for k in t_keys], axis=-1)
            un_s = jnp.concatenate([got_un[k] for k in s_keys], axis=-1)
            sh_sp_w = jnp.concatenate([got_sh_split[k] for k in t_keys], axis=-1)
            sh_sp_s = jnp.concatenate([got_sh_split[k] for k in s_keys], axis=-1)
            exp_w = np.asarray(hf_out["lin_in"][r, 0])
            exp_s = np.asarray(hf_out["lin_in_s"][r, 0])
            assert np.array_equal(np.asarray(un_w), exp_w)
            assert np.array_equal(np.asarray(got_sh["lin_in"]), exp_w)
            assert np.array_equal(np.asarray(sh_sp_w), exp_w)
            assert np.array_equal(np.asarray(un_s), exp_s)
            assert np.array_equal(np.asarray(got_sh["lin_in_s"]), exp_s)
            assert np.array_equal(np.asarray(sh_sp_s), exp_s)

    # 3. `pad` spec with `shard_groups=16` across all 32 ranks
    decl_pad = HFParamDecl(
        name="lin_b",
        hf_name="lin_b",
        shape=(h_dim, 32),
        dtype=BF16,
        transform=ParamTransform(
            shard_axis=-1, shard_groups=16, pad=((0, 0), (0, 128))
        ),
    )
    raw_b = jnp.arange(h_dim * 32, dtype=F32).reshape(h_dim, 32).astype(BF16)
    hf_b = _shard_and_transform_single(
        raw_b[None, ...], decl_pad, R=r_total, known_names={"lin_b"}
    )
    alias_b = VLLMAlias(
        source="lin_b",
        targets=("lin_b",),
        shard_axis=-1,
        already_sharded=False,
        already_fp8=False,
    )
    for r in range(r_total):
        got_b = slice_alias_targets(
            ops, alias_b, raw_b, None, (decl_pad,), {}, R=r_total, tp_rank=r
        )["lin_b"]
        assert got_b.shape == (h_dim, 128)
        assert np.array_equal(np.asarray(got_b), np.asarray(hf_b[r, 0]))
