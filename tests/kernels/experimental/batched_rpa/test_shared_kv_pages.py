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
"""CPU contract tests for shared-KV pages (`shared_kv_rotary_dims`)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    kernel,
    prologue,
    schedule,
    wrapper,
)

_FP8, _BF16, _F32, _I32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32, jnp.int32
_LANE = configs.KVLayout.SEQ_ALONG_LANE
_MESH = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
_SDS = jax.ShapeDtypeStruct


def _rows(x):
    return np.asarray(x).T.reshape(x.shape[1] // 4, 4, x.shape[0])


def _cfgs(rot, mode=configs.RpaCase.DECODE, bkv=1536, batch=8):
    return configs.RpaConfigs(
        block=configs.BlockSizes(1, 1, bkv, batch),
        model=configs.ModelConfigs(8, 1, 512, mask_value=-1e30),
        serve=configs.ServingConfigs(
            64,
            256,
            16,
            64 * 640,
            dtype_q=_BF16,
            dtype_kv=_FP8,
            dtype_out=_BF16,
            kv_layout=_LANE,
            shared_kv_rotary_dims=rot,
        ),
        mode=mode,
        vmem_limit_bytes=64 << 20,
    )


def test_prepare_inputs_writes_the_value_and_the_rotated_key_rows():
    tokens, d, rot = 40, 512, 64
    key = jax.random.key(0)
    q = jax.random.normal(key, (tokens, 8, d)).astype(_BF16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (tokens, 1, d)).astype(_FP8)
    v = jax.random.normal(jax.random.fold_in(key, 2), (tokens, 1, d)).astype(_FP8)
    with use_abstract_mesh(_MESH):
        q_hbm, new_kv = wrapper.prepare_inputs(
            q,
            k,
            v,
            _BF16,
            _FP8,
            kv_layout=_LANE,
            page_size=256,
            shared_kv_rotary_dims=rot,
        )
        q_std, _ = wrapper.prepare_inputs(
            q, k, v, _BF16, _FP8, kv_layout=_LANE, page_size=256
        )
    assert new_kv.shape == (2, d // 4, 4, 256)
    bits = lambda x: np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))  # noqa: E731
    np.testing.assert_array_equal(bits(q_hbm), bits(q_std))
    nk = bits(new_kv)
    np.testing.assert_array_equal(nk[0, :, :, :tokens], _rows(bits(v[:, 0])))
    k8 = bits(k[:, 0])
    rotated = np.concatenate([k8[:, :rot], k8[:, d // 2 : d // 2 + rot]], axis=1)
    np.testing.assert_array_equal(nk[1, : 2 * rot // 4, :, :tokens], _rows(rotated))
    assert not nk[1, 2 * rot // 4 :].any() and not nk[:, :, :, tokens:].any()


def test_kernels_move_the_value_and_rotated_key_rows_only():
    with use_abstract_mesh(_MESH):
        shared, std = _cfgs(64), _cfgs(0)
        assert (shared.kv_rows, std.kv_rows) == (160, 256)
        assert (shared.kv_bytes_per_token, std.kv_bytes_per_token) == (640, 1024)
        assert shared.kv_vmem_shape == (8, 1, 160, 4, 1792)
        assert std.kv_vmem_shape == (8, 2, 128, 4, 1792)
        sizes = [
            wrapper.calculate_block_sizes(c.model, c.serve, 64 << 20)[0]
            for c in (shared, std)
        ]
    assert sizes[0].bkv_sz > sizes[1].bkv_sz


@pytest.mark.parametrize(
    "rot,heads,d,layout,per_token,match",
    [
        (64, 2, 512, _LANE, False, "one KV head"),
        (64, 1, 512, configs.KVLayout.HEAD_ALONG_SUBLANE, False, "one KV head"),
        (48, 1, 512, _LANE, False, "tiles"),
        (512, 1, 512, _LANE, False, "is not in"),
        (64, 1, 512, _LANE, True, "per-token"),
    ],
)
def test_shared_kv_refusals(rot, heads, d, layout, per_token, match):
    with use_abstract_mesh(_MESH), pytest.raises(ValueError, match=match):
        wrapper.validate_shared_kv(rot, heads, d, layout, per_token)


def test_shared_kv_kernels_lower_for_tpu7x():
    with use_abstract_mesh(_MESH):
        kv_sh = wrapper.get_kv_cache_shape(64, 256, 1, 512, _FP8, kv_layout=_LANE)
        args = [
            _SDS((1024, 8, 512), _BF16),
            _SDS((1024, 1, 512), _FP8),
            _SDS((1024, 1, 512), _FP8),
            _SDS(kv_sh, _FP8),
            _SDS((8,), _I32),
            _SDS((128,), _I32),
            _SDS((9,), _I32),
            _SDS((3,), _I32),
        ]
        f = lambda *a: wrapper.ragged_paged_attention(  # noqa: E731
            *a,
            sm_scale=512**-0.5,
            k_scale=1.0,
            v_scale=1.0,
            kv_layout=_LANE,
            fp8_compute=True,
            shared_kv_rotary_dims=64,
        )
        text = jax.jit(f).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert "RPAd" in text and "RPAm" in text


def test_shared_kv_attention_matches_dense_reference():
    tokens, d, rot, k_scale, v_scale = 32, 128, 32, 0.5, 0.75
    key = jax.random.key(11)
    q_raw = jax.random.normal(key, (tokens, 8, d), _F32)
    v_fp8 = jax.random.normal(jax.random.fold_in(key, 1), (tokens, 1, d)).astype(_FP8)
    k_rot = jax.random.normal(jax.random.fold_in(key, 2), (tokens, 1, d)).astype(_FP8)
    wk = (1.0 + 0.2 * jax.random.normal(jax.random.fold_in(key, 3), (d,))).astype(_BF16)
    q = (
        q_raw * prologue.shared_kv_query_factor(wk, rot, k_scale, v_scale)[:, None, :]
    ).astype(_BF16)

    v_deq, k_rot_deq, wk_f = (
        v_fp8[:, 0].astype(_F32) * v_scale,
        k_rot[:, 0].astype(_F32) * k_scale,
        wk.astype(_F32)[None, :],
    )
    k_eff = jnp.concatenate(
        [
            k_rot_deq[:, :rot],
            v_deq[:, rot : d // 2] * wk_f[:, rot : d // 2],
            k_rot_deq[:, d // 2 : d // 2 + rot],
            v_deq[:, d // 2 + rot :] * wk_f[:, d // 2 + rot :],
        ],
        axis=-1,
    )
    logits = jnp.where(
        jnp.tril(jnp.ones((tokens, tokens), dtype=bool))[None],
        jnp.einsum("thd,sd->hts", q_raw.astype(_BF16).astype(_F32), k_eff) * (d**-0.5),
        -1e30,
    )
    ref = jnp.einsum("hts,sd->thd", jax.nn.softmax(logits, axis=-1), v_deq).astype(
        _BF16
    )

    if jax.devices()[0].platform == "tpu":
        kv_sh = wrapper.get_kv_cache_shape(4, 256, 1, d, _FP8, kv_layout=_LANE)
        out_tpu, _ = wrapper.ragged_paged_attention(
            q,
            k_rot,
            v_fp8,
            jnp.zeros(kv_sh, _FP8),
            jnp.array([tokens], _I32),
            jnp.array([0, 1, 2, 3], _I32),
            jnp.array([0, tokens], _I32),
            jnp.array([0, 1, 1], _I32),
            sm_scale=d**-0.5,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_layout=_LANE,
            shared_kv_rotary_dims=rot,
        )
        np.testing.assert_allclose(
            np.asarray(out_tpu, np.float32), np.asarray(ref, np.float32), atol=3e-2
        )

    with use_abstract_mesh(_MESH):
        cfgs = configs.RpaConfigs(
            block=configs.BlockSizes(tokens, tokens, 256, 1),
            model=configs.ModelConfigs(8, 1, d, sm_scale=d**-0.5, mask_value=-1e30),
            serve=configs.ServingConfigs(
                1,
                256,
                tokens,
                4,
                dtype_q=_BF16,
                dtype_kv=_FP8,
                dtype_out=_BF16,
                scale_k=k_scale,
                scale_v=v_scale,
                kv_layout=_LANE,
                shared_kv_rotary_dims=rot,
                mask_invalid_v=True,
            ),
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=64 << 20,
        )
        q_hbm, new_kv = wrapper.prepare_inputs(
            q,
            k_rot,
            v_fp8,
            _BF16,
            _FP8,
            kv_layout=_LANE,
            page_size=256,
            shared_kv_rotary_dims=rot,
        )
        kv_vmem = (
            jnp.zeros(cfgs.kv_vmem_shape, _FP8)
            .at[0, 0, :, :, :256]
            .set(jnp.concatenate([new_kv[0], new_kv[1, : 2 * rot // 4]], axis=0))
            .at[0, 0, :, :, tokens:]
            .set(jnp.nan)
        )
        sched_tmpl = schedule.RpaSchedule.create_shape_dtype(cfgs)
        sched_init = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, x.dtype), sched_tmpl
        )
        sched_init = dataclasses.replace(
            sched_init,
            is_last_k=dataclasses.replace(
                sched_init.is_last_k, data=sched_init.is_last_k.data.at[0].set(1)
            ),
        )
        vs = pl.BlockSpec(memory_space=pltpu.VMEM)
        ss = jax.tree_util.tree_map(
            lambda _: pl.BlockSpec(memory_space=pltpu.SMEM), sched_tmpl
        )

        def run_shared_kv(q_ref, kv_ref, sched_ref, o_ref, m_ref, l_ref, acc_ref):
            m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, _F32)
            l_ref[...], acc_ref[...] = (
                jnp.zeros(l_ref.shape, _F32),
                jnp.zeros(acc_ref.shape, _F32),
            )
            kernel._attend_and_store(
                _I32(0),
                q_ref,
                kv_ref,
                o_ref,
                None,
                sched_ref,
                m_ref,
                l_ref,
                acc_ref,
                stitch_meta=kernel.StitchMetadata([_I32(0)], [_I32(tokens)], [_I32(0)]),
                step_meta=kernel.ComputeMetadata([_I32(0)]),
                cfgs=cfgs,
                masked=True,
            )

        lm_s = _SDS(cfgs.lm_scratch_shape, _F32)
        o_vmem = pl.pallas_call(
            run_shared_kv,
            in_specs=(vs, vs, ss),
            out_specs=(vs,) * 4,
            out_shape=[
                _SDS(cfgs.o_vmem_shape, _BF16),
                lm_s,
                lm_s,
                _SDS(cfgs.acc_scratch_shape, _F32),
            ],
            interpret=True,
        )(q_hbm[None], kv_vmem, sched_init)[0]
        out = wrapper.prepare_outputs(o_vmem[0])[0]

    assert jnp.isfinite(out).all()
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(ref, np.float32), atol=3e-2
    )
