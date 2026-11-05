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
"""qkv_to_rpa_inputs against norm + rotary + quantize_kv + prepare_inputs.

Runs on CPU in Pallas interpret mode. The reference feeds the norms' f32 output
to rotary unrounded, as XLA does on TPU. The CPU backend sums the norms' squares
in a shape-dependent order, so an output may land one rounding step away; the
check allows that on a few elements and nothing more.
"""

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
        half = c.shape[-1]
        y1, y2, rest = y[..., :half], y[..., half : 2 * half], y[..., 2 * half :]
        return jnp.concatenate([y1 * c - y2 * s, y2 * c + y1 * s, rest], axis=-1)

    def quantize(x, scale):
        return jnp.clip(x / scale, -448.0, 448.0).astype(jnp.float8_e4m3fn)

    q = rotary(rms(q, wq, nq))
    q = (q if q_factor is None else q * q_factor).astype(jnp.bfloat16)
    k = rotary(rms(k, wk, nkv)).astype(jnp.bfloat16).astype(jnp.float32)
    v = rms(v, None, nkv)
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    with jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=device)
    ):
        return KB.prepare_inputs(
            q,
            quantize(k, k_scale),
            quantize(v, v_scale),
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            **layout,
        )


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,rotary_dim,block_tokens,k_scale,v_scale",
    [
        (64, 4, 2, 256, 256, 32, 1.0, 1.0),  # sliding, TP8 per rank
        (64, 4, 1, 512, 512, 32, 1.0, 1.0),  # global: one KV head, padded slots
        (64, 16, 2, 512, 512, 64, 0.5, 2.0),  # four q packs per KV head, KV scales
        (32, 8, 4, 256, 128, None, 1.0, 1.0),  # partial rotary, chosen block
        (40, 4, 2, 256, 256, None, 1.0, 1.0),  # odd bucket: padded to 48 rows
    ],
)
def test_matches_norm_rotary_quantize_prepare(
    tokens, nq, nkv, d, rotary_dim, block_tokens, k_scale, v_scale, layout=None
):
    layout = layout or {}
    key = jax.random.key(tokens + nq + d)
    qkv = jax.random.normal(key, (tokens, (nq + 2 * nkv) * d), jnp.float32) * 2
    qkv = qkv.astype(jnp.bfloat16)
    inv = 1.0 / (10000.0 ** (jnp.arange(rotary_dim // 2) / (rotary_dim // 2)))
    angle = jnp.arange(tokens)[:, None] * inv[None]
    cos, sin = jnp.cos(angle).astype(jnp.bfloat16), jnp.sin(angle).astype(jnp.bfloat16)
    wq = (1 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (d,))).astype(
        jnp.bfloat16
    )
    wk = (1 + 0.1 * jax.random.normal(jax.random.fold_in(key, 2), (d,))).astype(
        jnp.bfloat16
    )
    heads = dict(nq=nq, nkv=nkv, d=d, k_scale=k_scale, v_scale=v_scale)
    with jax.default_matmul_precision("highest"):
        ref_q, ref_kv = _reference(qkv, cos, sin, wq, wk, **heads, **layout)
    q, kv = prologue.qkv_to_rpa_inputs(
        qkv,
        cos,
        sin,
        wq,
        wk,
        num_q_heads=nq,
        num_kv_heads=nkv,
        head_dim=d,
        eps=_EPS,
        k_scale=k_scale,
        v_scale=v_scale,
        block_tokens=block_tokens,
        interpret=True,
        **layout,
    )
    assert q.shape == ref_q.shape and kv.shape == ref_kv.shape
    _assert_within_one_step(q, ref_q, jnp.int16)
    _assert_within_one_step(kv, ref_kv, jnp.int8)


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,rotary_dim,block_tokens,page_size",
    [
        (16, 8, 1, 512, 512, None, 256),  # Gemma4-26B TP2 global decode bucket
        (128, 8, 4, 256, 256, 128, 256),  # Gemma4-26B TP2 sliding: padded to a page
        (40, 4, 2, 256, 128, None, 128),  # odd bucket, partial rotary
        (128, 4, 1, 512, 512, 128, 128),  # page-sized: no padding
    ],
)
def test_seq_along_lane_matches_prepare(
    tokens, nq, nkv, d, rotary_dim, block_tokens, page_size
):
    """SEQ_ALONG_LANE K/V are prepare_inputs' [2 * kv_heads, d / 4, 4, padded
    tokens], zero past the tokens."""
    layout = dict(kv_layout=configs.KVLayout.SEQ_ALONG_LANE, page_size=page_size)
    test_matches_norm_rotary_quantize_prepare(
        tokens, nq, nkv, d, rotary_dim, block_tokens, 1.0, 1.0, layout
    )


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,block_tokens",
    [
        (512, 8, 4, 256, 128),  # four token blocks
        (1024, 8, 1, 512, 512),  # two token blocks, one KV head
        (1040, 4, 1, 512, None),  # past one block: padded to whole pages
        (512, 32, 4, 512, None),  # TP1-wide heads: one block exceeds VMEM
    ],
)
def test_seq_along_lane_is_the_default_layout_relaid_out(
    tokens, nq, nkv, d, block_tokens
):
    """Across token blocks, the SEQ_ALONG_LANE K/V hold the HEAD_ALONG_SUBLANE
    K/V's bits in prepare_inputs' SEQ_ALONG_LANE order (zero past the tokens),
    and q is the same."""
    key = jax.random.key(tokens + d)
    qkv = jax.random.normal(key, (tokens, (nq + 2 * nkv) * d)) * 2
    qkv = qkv.astype(jnp.bfloat16)
    angle = jnp.arange(tokens)[:, None] / (100.0 ** jnp.arange(d // 2))[None]
    cos, sin = jnp.cos(angle).astype(jnp.bfloat16), jnp.sin(angle).astype(jnp.bfloat16)
    w = (1 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (d,))).astype(
        jnp.bfloat16
    )
    kwargs = dict(num_q_heads=nq, num_kv_heads=nkv, head_dim=d, eps=_EPS, k_scale=1.0)
    kwargs.update(v_scale=1.0, interpret=True)
    q, kv = prologue.qkv_to_rpa_inputs(
        qkv, cos, sin, w, w, block_tokens=block_tokens, **kwargs
    )
    lane = configs.KVLayout.SEQ_ALONG_LANE
    q_lane, kv_lane = prologue.qkv_to_rpa_inputs(
        qkv, cos, sin, w, w, kv_layout=lane, page_size=256, **kwargs
    )
    rows = kv.reshape(tokens, -1, d)[:, : 2 * nkv]  # [tokens, k v of each head, d]
    want = rows.transpose(1, 2, 0).reshape(2 * nkv, d // 4, 4, tokens)
    want = jnp.pad(want, ((0, 0),) * 3 + ((0, kv_lane.shape[-1] - tokens),))
    bits = lambda x: jax.lax.bitcast_convert_type(x, jnp.uint8)  # noqa: E731
    np.testing.assert_array_equal(bits(q_lane), bits(q))
    np.testing.assert_array_equal(bits(kv_lane), bits(want))


def test_seq_along_lane_one_block_needs_the_vmem_budget():
    """Buckets of up to 512 tokens take one block (K/V across all their padded
    lanes) only where it fits VMEM, as the token-major path's blocks must."""
    one_block = prologue._seq_along_lane_one_block
    # Gemma4-26B TP2 global and sliding decode buckets: one block.
    assert one_block(16, 10 * 512, 8, 1, 512, False, 256)
    assert one_block(16, 16 * 256, 8, 2, 256, False, 256)
    # Gemma4-31B TP1 global (32 q, 4 KV heads, head_dim 512): 512 tokens do not.
    assert not one_block(512, 40 * 512, 32, 2, 512, False, 512)
    assert not one_block(1024, 10 * 512, 8, 1, 512, False, 1024)


@pytest.mark.parametrize(
    "tokens,nq,d,rot,k_scale,v_scale,page_size",
    [
        (16, 8, 512, 64, 1.0, 1.0, 256),  # Gemma4-26B TP2 global decode bucket
        (128, 8, 512, 64, 0.5, 2.0, 128),  # static KV scales in the query factor
        (40, 4, 256, 32, 1.0, 1.0, 128),  # odd bucket
    ],
)
def test_shared_kv_matches_prepare(tokens, nq, d, rot, k_scale, v_scale, page_size):
    """Shared-KV rows are prepare_inputs(shared_kv_rotary_dims)'s of the same K
    and V, and the queries carry shared_kv_query_factor."""
    key = jax.random.key(tokens + d + rot)
    qkv = jax.random.normal(key, (tokens, (nq + 2) * d), jnp.float32) * 2
    qkv = qkv.astype(jnp.bfloat16)
    angle = jnp.arange(tokens)[:, None] * (0.5 ** jnp.arange(d // 2))[None]
    cos, sin = jnp.cos(angle).astype(jnp.bfloat16), jnp.sin(angle).astype(jnp.bfloat16)
    wq = (1 + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (d,))).astype(
        jnp.bfloat16
    )
    wk = (0.06 + 0.01 * jax.random.normal(jax.random.fold_in(key, 2), (d,))).astype(
        jnp.bfloat16
    )
    layout = dict(
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        page_size=page_size,
        shared_kv_rotary_dims=rot,
    )
    factor = prologue.shared_kv_query_factor(wk, rot, k_scale, v_scale)
    heads = dict(nq=nq, nkv=1, d=d, k_scale=k_scale, v_scale=v_scale)
    with jax.default_matmul_precision("highest"):
        ref_q, ref_kv = _reference(
            qkv, cos, sin, wq, wk, q_factor=factor, **heads, **layout
        )
    kwargs = dict(num_q_heads=nq, num_kv_heads=1, head_dim=d, eps=_EPS)
    kwargs.update(k_scale=k_scale, v_scale=v_scale, interpret=True, **layout)
    q, kv = prologue.qkv_to_rpa_inputs(qkv, cos, sin, wq, wk, **kwargs)
    assert q.shape == ref_q.shape and kv.shape == ref_kv.shape
    _assert_within_one_step(q, ref_q, jnp.int16)
    _assert_within_one_step(kv, ref_kv, jnp.int8)
    fp8 = dict(fp8_queries=True) if nq % 4 == 0 else {}
    q_dma, kv_dma, *q_fp8 = prologue.qkv_to_rpa_inputs(
        qkv, cos, sin, wq, wk, dma_stores=True, **fp8, **kwargs
    )
    bits = lambda x: jax.lax.bitcast_convert_type(x, jnp.uint8)  # noqa: E731
    np.testing.assert_array_equal(bits(q_dma), bits(q))
    np.testing.assert_array_equal(bits(kv_dma), bits(kv))
    for q8 in q_fp8:
        ref = KB.quantize_queries(q, jnp.float8_e4m3fn)
        np.testing.assert_array_equal(bits(q8), bits(ref))


def test_shared_kv_needs_seq_along_lane_pages_of_one_head():
    qkv = jnp.zeros((16, 6 * 256), jnp.bfloat16)
    rows = jnp.zeros((16, 128), jnp.bfloat16)
    w = jnp.ones((256,), jnp.bfloat16)
    kwargs = dict(num_q_heads=4, num_kv_heads=1, head_dim=256, eps=_EPS)
    kwargs.update(k_scale=1.0, v_scale=1.0, interpret=True, shared_kv_rotary_dims=32)
    with pytest.raises(ValueError, match="SEQ_ALONG_LANE pages of one KV head"):
        prologue.qkv_to_rpa_inputs(qkv, rows, rows, w, w, **kwargs)


def test_seq_along_lane_needs_page_size():
    qkv = jnp.zeros((128, 6 * 256), jnp.bfloat16)
    rows = jnp.zeros((128, 128), jnp.bfloat16)
    w = jnp.ones((256,), jnp.bfloat16)
    kwargs = dict(num_q_heads=4, num_kv_heads=1, head_dim=256, eps=_EPS)
    kwargs.update(k_scale=1.0, v_scale=1.0, interpret=True)
    lane = configs.KVLayout.SEQ_ALONG_LANE
    with pytest.raises(ValueError, match="page_size"):
        prologue.qkv_to_rpa_inputs(qkv, rows, rows, w, w, kv_layout=lane, **kwargs)


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,rotary_dim,block_tokens",
    [
        (64, 16, 8, 256, 256, 16),  # Gemma4-31B TP2 sliding: four K/V slots
        (64, 16, 2, 512, 128, 32),  # Gemma4-31B TP2 global: partial rotary
        (48, 4, 1, 128, 64, 16),  # one KV head: K/V rows padded in their slot
        (40, 4, 2, 256, 256, None),  # odd bucket, chosen block
        (1024, 8, 1, 512, 512, 512),  # two blocks (SEQ_ALONG_LANE: lane tiles)
    ],
)
@pytest.mark.parametrize("layout", ["nhd", "hnd"])
def test_dma_stores_write_the_same_bits(
    tokens, nq, nkv, d, rotary_dim, block_tokens, layout
):
    key = jax.random.key(tokens * nq + d)
    qkv = jax.random.normal(key, (tokens, (nq + 2 * nkv) * d), jnp.float32) * 2
    qkv = qkv.astype(jnp.bfloat16)
    angle = jax.random.uniform(jax.random.fold_in(key, 1), (tokens, rotary_dim // 2))
    cos, sin = jnp.cos(angle).astype(jnp.bfloat16), jnp.sin(angle).astype(jnp.bfloat16)
    w = (1 + 0.1 * jax.random.normal(jax.random.fold_in(key, 2), (d,))).astype(
        jnp.bfloat16
    )
    outs = [
        prologue.qkv_to_rpa_inputs(
            qkv,
            cos,
            sin,
            w,
            w,
            num_q_heads=nq,
            num_kv_heads=nkv,
            head_dim=d,
            eps=_EPS,
            k_scale=0.5,
            v_scale=1.0,
            block_tokens=block_tokens,
            dma_stores=dma,
            interpret=True,
            **_LAYOUTS[layout],
        )
        for dma in (False, True)
    ]
    (q, kv), (q_dma, kv_dma) = outs
    np.testing.assert_array_equal(
        np.asarray(q_dma.view(jnp.uint16)), np.asarray(q.view(jnp.uint16))
    )
    np.testing.assert_array_equal(
        np.asarray(kv_dma.view(jnp.uint8)), np.asarray(kv.view(jnp.uint8))
    )


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,rotary_dim,block_tokens",
    [
        (64, 16, 2, 512, 128, 32),  # Gemma4-31B TP2 global layer (fp8_compute)
        (48, 8, 1, 256, 256, 16),  # two FP8 query words per KV head
    ],
)
@pytest.mark.parametrize("layout", ["nhd", "hnd"])
def test_fp8_queries_are_quantize_queries(
    tokens, nq, nkv, d, rotary_dim, block_tokens, layout
):
    key = jax.random.key(tokens + nq * d)
    qkv = jax.random.normal(key, (tokens, (nq + 2 * nkv) * d), jnp.float32) * 3
    qkv = qkv.astype(jnp.bfloat16)
    angle = jax.random.uniform(jax.random.fold_in(key, 1), (tokens, rotary_dim // 2))
    cos, sin = jnp.cos(angle).astype(jnp.bfloat16), jnp.sin(angle).astype(jnp.bfloat16)
    w = jnp.ones((d,), jnp.bfloat16)
    kwargs = dict(
        num_q_heads=nq,
        num_kv_heads=nkv,
        head_dim=d,
        eps=_EPS,
        k_scale=1.0,
        v_scale=1.0,
        block_tokens=block_tokens,
        interpret=True,
        **_LAYOUTS[layout],
    )
    q, kv = prologue.qkv_to_rpa_inputs(qkv, cos, sin, w, w, **kwargs)
    q_dma, kv_dma, q_fp8 = prologue.qkv_to_rpa_inputs(
        qkv, cos, sin, w, w, dma_stores=True, fp8_queries=True, **kwargs
    )
    ref = KB.quantize_queries(q, jnp.float8_e4m3fn)
    assert q_fp8.shape == ref.shape and q_fp8.dtype == ref.dtype
    np.testing.assert_array_equal(
        np.asarray(q_fp8.view(jnp.uint8)), np.asarray(ref.view(jnp.uint8))
    )
    np.testing.assert_array_equal(
        np.asarray(q_dma.view(jnp.uint16)), np.asarray(q.view(jnp.uint16))
    )
    np.testing.assert_array_equal(
        np.asarray(kv_dma.view(jnp.uint8)), np.asarray(kv.view(jnp.uint8))
    )


def test_fp8_queries_need_dma_stores_and_heads_in_fours():
    qkv = jnp.zeros((32, 4 * 256), jnp.bfloat16)
    cs = jnp.zeros((32, 128), jnp.bfloat16)
    w = jnp.ones((256,), jnp.bfloat16)
    kwargs = dict(num_q_heads=2, num_kv_heads=1, head_dim=256, eps=_EPS)
    kwargs.update(k_scale=1.0, v_scale=1.0, fp8_queries=True, interpret=True)
    with pytest.raises(ValueError, match="fp8_queries"):
        prologue.qkv_to_rpa_inputs(qkv, cs, cs, w, w, dma_stores=True, **kwargs)


def _assert_within_one_step(out, ref, bits):
    """Same values up to one representable step on at most 0.1% of elements."""
    a = np.asarray(out.view(bits)).astype(np.int32)
    b = np.asarray(ref.view(bits)).astype(np.int32)
    steps = np.abs(a - b)
    assert steps.max() <= 1, steps.max()
    assert np.count_nonzero(steps) <= max(1, out.size // 1000), np.count_nonzero(steps)


def test_rejects_odd_q_heads_per_kv_head():
    qkv = jnp.zeros((32, 5 * 256), jnp.bfloat16)
    cs = jnp.zeros((32, 128), jnp.bfloat16)
    w = jnp.ones((256,), jnp.bfloat16)
    with pytest.raises(ValueError, match="even number"):
        prologue.qkv_to_rpa_inputs(
            qkv,
            cs,
            cs,
            w,
            w,
            num_q_heads=3,
            num_kv_heads=1,
            head_dim=256,
            eps=_EPS,
            k_scale=1.0,
            v_scale=1.0,
            interpret=True,
        )


def test_block_tokens_fit_vmem_and_tile():
    # 31B TP1 widths do not fit 512 tokens; a 65 x 128 bucket takes 128.
    assert prologue.choose_block_tokens(10240, (32 + 32) * 256, 32, 8, 256) == 256
    assert prologue.choose_block_tokens(10240, (32 + 8) * 512, 32, 2, 512) == 256
    assert prologue.choose_block_tokens(16384, (16 + 16) * 256, 16, 4, 256) == 512
    assert prologue.choose_block_tokens(8320, (16 + 16) * 256, 16, 4, 256) == 128
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
    """The sharded entry runs the prologue kernel, then the batched RPA kernels."""
    from vllm_torchtpu.layers.core import attention_interface

    tokens = 16
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("model",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=device,
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
    assert new_cache.shape == cache
    assert "qkv_to_rpa_inputs" in text
    if rpa_kwargs.get("prologue_fp8_queries"):
        # The FP8 queries come from the prologue: no cast of the bf16 ones
        # (quantize_queries' convert bf16 -> f8E4M3FN).
        casts = [
            line
            for line in text.splitlines()
            if "stablehlo.convert" in line
            and "bf16>) -> tensor<" in line
            and "f8E4M3FN>" in line
        ]
        assert "RPAm" in text and not casts, casts
