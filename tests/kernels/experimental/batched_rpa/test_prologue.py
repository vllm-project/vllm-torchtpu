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
"""CPU interpret and TPU lowering tests for prologue.qkv_to_rpa_inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, prologue
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB

_EPS = 1e-6
_LAYOUTS = {
    "nhd": {},
    "hnd": dict(kv_layout=configs.KVLayout.SEQ_ALONG_LANE, page_size=128),
}


def _make_qkv_case(tokens, nq, nkv, d, rotary_dim, seed, angle_base=10000.0):
    key = jax.random.key(seed)
    qkv = (
        jax.random.normal(key, (tokens, (nq + 2 * nkv) * d), jnp.float32) * 2
    ).astype(jnp.bfloat16)
    inv = 1.0 / (angle_base ** (jnp.arange(rotary_dim // 2) / (rotary_dim // 2)))
    angle = jnp.arange(tokens)[:, None] * inv[None]
    cos, sin = jnp.cos(angle).astype(jnp.bfloat16), jnp.sin(angle).astype(jnp.bfloat16)
    wq = (1 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (d,))).astype(
        jnp.bfloat16
    )
    wk = (1 + 0.1 * jax.random.normal(jax.random.fold_in(key, 2), (d,))).astype(
        jnp.bfloat16
    )
    return qkv, cos, sin, wq, wk


def _reference(
    qkv, cos, sin, wq, wk, *, nq, nkv, d, k_scale, v_scale, q_factor=None, **layout
):
    tokens = qkv.shape[0]
    q, k, v = jnp.split(qkv, [nq * d, (nq + nkv) * d], axis=-1)

    def rms(x, w, heads):
        x32 = x.reshape(tokens, heads, d).astype(jnp.float32)
        x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + _EPS)
        return x32 if w is None else x32 * w.astype(jnp.float32)

    def rotary(y):
        c, s = cos.astype(jnp.float32)[:, None], sin.astype(jnp.float32)[:, None]
        h = c.shape[-1]
        y1, y2, rest = y[..., :h], y[..., h : 2 * h], y[..., 2 * h :]
        return jnp.concatenate([y1 * c - y2 * s, y2 * c + y1 * s, rest], axis=-1)

    q = (rotary(rms(q, wq, nq)) * (1.0 if q_factor is None else q_factor)).astype(
        jnp.bfloat16
    )
    k = rotary(rms(k, wk, nkv)).astype(jnp.bfloat16).astype(jnp.float32)
    v = rms(v, None, nkv)
    dev = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=dev)
    ):
        return KB.prepare_inputs(
            q,
            jnp.clip(k / k_scale, -448.0, 448.0).astype(jnp.float8_e4m3fn),
            jnp.clip(v / v_scale, -448.0, 448.0).astype(jnp.float8_e4m3fn),
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            **layout,
        )


def _ordered_steps(out: jax.Array, ref: jax.Array) -> np.ndarray:
    assert out.shape == ref.shape and out.dtype == ref.dtype
    out_f32, ref_f32 = (
        np.asarray(out.astype(jnp.float32)),
        np.asarray(ref.astype(jnp.float32)),
    )
    np.testing.assert_array_equal(np.isnan(out_f32), np.isnan(ref_f32))
    np.testing.assert_array_equal(np.isinf(out_f32), np.isinf(ref_f32))
    dt = jnp.dtype(out.dtype)
    bits_dtype, sign, mag = (
        (jnp.uint16, 0x8000, 0x7FFF) if dt == jnp.bfloat16 else (jnp.uint8, 0x80, 0x7F)
    )
    a_u = np.asarray(out.view(bits_dtype)).astype(np.int32)
    b_u = np.asarray(ref.view(bits_dtype)).astype(np.int32)
    return np.abs(
        np.where(a_u & sign, -(a_u & mag), a_u)
        - np.where(b_u & sign, -(b_u & mag), b_u)
    )


def _assert_within_one_step(out: jax.Array, ref: jax.Array) -> None:
    steps = _ordered_steps(out, ref)
    assert steps.max() <= 1 and np.count_nonzero(steps) <= max(1, out.size // 1000)


def _assert_same_or_one_tpu_step(out: jax.Array, ref: jax.Array) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(out, jnp.uint8)),
            np.asarray(jax.lax.bitcast_convert_type(ref, jnp.uint8)),
        )
        return
    steps = _ordered_steps(out, ref)
    if jnp.dtype(out.dtype) == jnp.bfloat16:
        abs_diff = np.abs(
            np.asarray(out.astype(jnp.float32)) - np.asarray(ref.astype(jnp.float32))
        )
        assert not np.any((steps > 1) & (abs_diff > 1e-5))
        return
    assert steps.max() <= 1


def _rpa_in(*pos, nq, nkv=1, d=256, k_scale=1.0, v_scale=1.0, interpret=True, **kw):
    return prologue.qkv_to_rpa_inputs(
        *pos,
        num_q_heads=nq,
        num_kv_heads=nkv,
        head_dim=d,
        eps=_EPS,
        k_scale=k_scale,
        v_scale=v_scale,
        interpret=interpret,
        **kw,
    )


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,rotary_dim,block_tokens,k_scale,v_scale",
    [
        (64, 4, 2, 256, 256, 32, 1.0, 1.0),
        (64, 4, 1, 512, 512, 32, 1.0, 1.0),
        (64, 16, 2, 512, 512, 64, 0.5, 2.0),
        (32, 8, 4, 256, 128, None, 1.0, 1.0),
        (40, 4, 2, 256, 256, None, 1.0, 1.0),
    ],
)
def test_matches_norm_rotary_quantize_prepare(
    tokens, nq, nkv, d, rotary_dim, block_tokens, k_scale, v_scale, layout=None
):
    layout = layout or {}
    pos = _make_qkv_case(tokens, nq, nkv, d, rotary_dim, tokens + nq + d)
    scales = dict(nq=nq, nkv=nkv, d=d, k_scale=k_scale, v_scale=v_scale, **layout)
    with jax.default_matmul_precision("highest"):
        ref_q, ref_kv = _reference(*pos, **scales)
    q, kv = _rpa_in(*pos, block_tokens=block_tokens, **scales)
    assert q.shape == ref_q.shape and kv.shape == ref_kv.shape
    _assert_within_one_step(q, ref_q)
    _assert_within_one_step(kv, ref_kv)


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,rotary_dim,block_tokens,page_size",
    [
        (16, 8, 1, 512, 512, None, 256),
        (128, 8, 4, 256, 256, 128, 256),
        (40, 4, 2, 256, 128, None, 128),
        (128, 4, 1, 512, 512, 128, 128),
    ],
)
def test_seq_along_lane_matches_prepare(
    tokens, nq, nkv, d, rotary_dim, block_tokens, page_size
):
    lay = dict(kv_layout=configs.KVLayout.SEQ_ALONG_LANE, page_size=page_size)
    test_matches_norm_rotary_quantize_prepare(
        tokens, nq, nkv, d, rotary_dim, block_tokens, 1.0, 1.0, lay
    )


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,block_tokens",
    [
        (512, 8, 4, 256, 128),
        (1024, 8, 1, 512, 512),
        (1040, 4, 1, 512, None),
        (512, 32, 4, 512, None),
    ],
)
def test_seq_along_lane_is_the_default_layout_relaid_out(
    tokens, nq, nkv, d, block_tokens
):
    qkv, cos, sin, w, _ = _make_qkv_case(tokens, nq, nkv, d, d, tokens + d)
    pos = (qkv, cos, sin, w, w)
    q, kv = _rpa_in(*pos, nq=nq, nkv=nkv, d=d, block_tokens=block_tokens)
    q_lane, kv_lane = _rpa_in(
        *pos,
        nq=nq,
        nkv=nkv,
        d=d,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        page_size=256,
    )
    rows = kv.reshape(tokens, -1, d)[:, : 2 * nkv]
    want = jnp.pad(
        rows.transpose(1, 2, 0).reshape(2 * nkv, d // 4, 4, tokens),
        ((0, 0),) * 3 + ((0, kv_lane.shape[-1] - tokens),),
    )
    _assert_same_or_one_tpu_step(q_lane, q)
    _assert_same_or_one_tpu_step(kv_lane, want)
    if tokens < kv_lane.shape[-1]:
        assert int(jnp.count_nonzero(kv_lane[..., tokens:].view(jnp.uint8))) == 0


def test_seq_along_lane_one_block_needs_the_vmem_budget():
    one_block = prologue._seq_along_lane_one_block
    assert one_block(16, 10 * 512, 8, 1, 512, False, 256)
    assert one_block(16, 16 * 256, 8, 2, 256, False, 256)
    assert not one_block(512, 40 * 512, 32, 2, 512, False, 512)
    assert not one_block(1024, 10 * 512, 8, 1, 512, False, 1024)


@pytest.mark.parametrize(
    "tokens,nq,d,rot,k_scale,v_scale,page_size",
    [
        (16, 8, 512, 64, 1.0, 1.0, 256),
        (128, 8, 512, 64, 0.5, 2.0, 128),
        (40, 4, 256, 32, 1.0, 1.0, 128),
    ],
)
def test_shared_kv_matches_prepare(tokens, nq, d, rot, k_scale, v_scale, page_size):
    qkv, cos, sin, wq, wk = _make_qkv_case(tokens, nq, 1, d, d, tokens + d + rot)
    wk = (wk * 0.06).astype(jnp.bfloat16)
    pos = (qkv, cos, sin, wq, wk)
    kw = dict(
        nq=nq,
        d=d,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        page_size=page_size,
        shared_kv_rotary_dims=rot,
    )
    factor = prologue.shared_kv_query_factor(wk, rot, k_scale, v_scale)
    with jax.default_matmul_precision("highest"):
        ref_q, ref_kv = _reference(*pos, nkv=1, q_factor=factor, **kw)
    q, kv = _rpa_in(*pos, **kw)
    _assert_within_one_step(q, ref_q)
    _assert_within_one_step(kv, ref_kv)
    fp8 = dict(fp8_queries=True) if nq % 4 == 0 else {}
    q_dma, kv_dma, *q_fp8 = _rpa_in(*pos, dma_stores=True, **fp8, **kw)
    _assert_same_or_one_tpu_step(q_dma, q)
    _assert_same_or_one_tpu_step(kv_dma, kv)
    if tokens < kv_dma.shape[-1]:
        assert int(jnp.count_nonzero(kv_dma[..., tokens:].view(jnp.uint8))) == 0
    for q8 in q_fp8:
        _assert_same_or_one_tpu_step(q8, KB.quantize_queries(q_dma, jnp.float8_e4m3fn))


@pytest.mark.parametrize(
    ("nq", "extra_kw", "match"),
    [
        (4, dict(shared_kv_rotary_dims=32), "SEQ_ALONG_LANE pages of one KV head"),
        (4, dict(kv_layout=configs.KVLayout.SEQ_ALONG_LANE), "page_size"),
        (2, dict(dma_stores=True, fp8_queries=True), "fp8_queries"),
        (3, {}, "even number"),
    ],
)
def test_prologue_validation_errors(nq, extra_kw, match):
    qkv = jnp.zeros((32, (nq + 2) * 256), jnp.bfloat16)
    cs, w = jnp.zeros((32, 128), jnp.bfloat16), jnp.ones((256,), jnp.bfloat16)
    with pytest.raises(ValueError, match=match):
        _rpa_in(qkv, cs, cs, w, w, nq=nq, **extra_kw)


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,rotary_dim,block_tokens,fp8_q",
    [
        (64, 16, 8, 256, 256, 16, False),
        (64, 16, 2, 512, 128, 32, True),
        (48, 8, 1, 256, 256, 16, True),
        (48, 4, 1, 128, 64, 16, False),
        (40, 4, 2, 256, 256, None, False),
        (1024, 8, 1, 512, 512, 512, False),
    ],
)
@pytest.mark.parametrize("layout", ["nhd", "hnd"])
def test_dma_stores_and_fp8_queries_match(
    tokens, nq, nkv, d, rotary_dim, block_tokens, fp8_q, layout
):
    qkv, cos, sin, wq, wk = _make_qkv_case(
        tokens, nq, nkv, d, rotary_dim, tokens * nq + d
    )
    kw = dict(
        num_q_heads=nq,
        num_kv_heads=nkv,
        head_dim=d,
        eps=_EPS,
        k_scale=0.5,
        v_scale=1.0,
        block_tokens=block_tokens,
        interpret=True,
        **_LAYOUTS[layout],
    )
    q, kv = prologue.qkv_to_rpa_inputs(qkv, cos, sin, wq, wk, **kw)
    q_dma, kv_dma, *q_fp8 = prologue.qkv_to_rpa_inputs(
        qkv, cos, sin, wq, wk, dma_stores=True, fp8_queries=fp8_q, **kw
    )
    _assert_same_or_one_tpu_step(q_dma, q)
    _assert_same_or_one_tpu_step(kv_dma, kv)
    if fp8_q:
        ref = KB.quantize_queries(q_dma, jnp.float8_e4m3fn)
        assert q_fp8[0].shape == ref.shape and q_fp8[0].dtype == ref.dtype
        _assert_same_or_one_tpu_step(q_fp8[0], ref)


def test_block_tokens_fit_vmem_and_tile():
    assert prologue.choose_block_tokens(10240, 64 * 256, 32, 8, 256) == 256
    assert prologue.choose_block_tokens(10240, 40 * 512, 32, 2, 512) == 256
    assert prologue.choose_block_tokens(16384, 32 * 256, 16, 4, 256) == 512
    assert prologue.choose_block_tokens(8320, 32 * 256, 16, 4, 256) == 128
    with pytest.raises(ValueError, match="no token block"):
        prologue.choose_block_tokens(24, 512, 1, 1, 256)


@pytest.mark.parametrize(
    "nq,nkv,d,rpa_kwargs",
    [
        pytest.param(2, 1, 256, {}, id="bf16-queries"),
        pytest.param(
            4,
            1,
            512,
            dict(fp8_compute=True, prologue_dma_stores=True, prologue_fp8_queries=True),
            id="fp8-queries-from-the-prologue",
        ),
        pytest.param(
            8,
            2,
            256,
            dict(prologue_dma_stores=True, output_quant_dtype=jnp.float8_e4m3fn),
            id="dma-stores-quantized-output",
        ),
        pytest.param(
            8,
            1,
            512,
            dict(kv_layout=configs.KVLayout.SEQ_ALONG_LANE, fp8_compute=True),
            id="seq-along-lane-pages",
        ),
        pytest.param(
            8,
            4,
            256,
            dict(
                kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                prologue_dma_stores=True,
                output_quant_dtype=jnp.float8_e4m3fn,
            ),
            id="seq-along-lane-dma-stores-quantized-output",
        ),
    ],
)
def test_sharded_attention_from_qkv_lowers_for_tpu(nq, nkv, d, rpa_kwargs):
    from vllm_torchtpu.layers.core import attention_interface

    tokens = 16
    dev = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    mesh = jax.sharding.AbstractMesh(
        (1,), ("model",), axis_types=(jax.sharding.AxisType.Auto,), abstract_device=dev
    )
    spec = jax.ShapeDtypeStruct
    layout = rpa_kwargs.get("kv_layout", configs.KVLayout.HEAD_ALONG_SUBLANE)
    page_size = 128 if layout == configs.KVLayout.SEQ_ALONG_LANE else 16
    with jax.sharding.use_abstract_mesh(mesh):
        cache = KB.get_kv_cache_shape(
            128, page_size, nkv, d, jnp.float8_e4m3fn, kv_layout=layout
        )
        args = (
            spec((tokens, (nq + 2 * nkv) * d), jnp.bfloat16),
            spec((tokens, d // 2), jnp.bfloat16),
            spec((tokens, d // 2), jnp.bfloat16),
            spec((d,), jnp.bfloat16),
            spec((d,), jnp.bfloat16),
            spec(cache, jnp.float8_e4m3fn),
            spec((16,), jnp.int32),
            spec((128,), jnp.int32),
            spec((17,), jnp.int32),
            spec((3,), jnp.int32),
        )

        def attend(*a):
            return attention_interface.sharded_ragged_paged_attention_from_qkv(
                mesh,
                *a,
                num_q_heads=nq,
                num_kv_heads=nkv,
                head_dim=d,
                eps=_EPS,
                sm_scale=d**-0.5,
                k_scale=1.0,
                v_scale=1.0,
                **rpa_kwargs,
            )

        out, new_cache = jax.eval_shape(attend, *args)
        text = jax.jit(attend).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    if "output_quant_dtype" in rpa_kwargs:
        q, scale = out
        assert q.shape == (tokens, nq * d) and scale.shape == (tokens, 1)
        assert "rpa_output_quantize" in text
    else:
        assert out.shape == (tokens, nq, d)
    assert new_cache.shape == cache and "qkv_to_rpa_inputs" in text
    if rpa_kwargs.get("prologue_fp8_queries"):
        casts = [
            ln
            for ln in text.splitlines()
            if "stablehlo.convert" in ln
            and "bf16>) -> tensor<" in ln
            and "f8E4M3FN>" in ln
        ]
        assert "RPAm" in text and not casts, casts
