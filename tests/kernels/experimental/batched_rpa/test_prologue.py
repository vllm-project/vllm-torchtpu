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

from vllm_torchtpu.kernels.experimental.batched_rpa import prologue
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB

_EPS = 1e-6


def _reference(qkv, cos, sin, wq, wk, *, nq, nkv, d, k_scale, v_scale):
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

    q = rotary(rms(q, wq, nq)).astype(jnp.bfloat16)
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
    tokens, nq, nkv, d, rotary_dim, block_tokens, k_scale, v_scale
):
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
        ref_q, ref_kv = _reference(qkv, cos, sin, wq, wk, **heads)
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
    )
    assert q.shape == ref_q.shape and kv.shape == ref_kv.shape
    _assert_within_one_step(q, ref_q, jnp.int16)
    _assert_within_one_step(kv, ref_kv, jnp.int8)


@pytest.mark.parametrize(
    "tokens,nq,nkv,d,rotary_dim,block_tokens",
    [
        (64, 16, 8, 256, 256, 16),  # Gemma4-31B TP2 sliding: four K/V slots
        (64, 16, 2, 512, 128, 32),  # Gemma4-31B TP2 global: partial rotary
        (48, 4, 1, 128, 64, 16),  # one KV head: K/V rows padded in their slot
        (40, 4, 2, 256, 256, None),  # odd bucket, chosen block
    ],
)
def test_dma_stores_write_the_same_bits(tokens, nq, nkv, d, rotary_dim, block_tokens):
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


def test_sharded_attention_from_qkv_lowers_for_tpu():
    """The sharded entry runs the prologue kernel, then the batched RPA kernels."""
    from vllm_torchtpu.layers.core import attention_interface

    tokens, nq, nkv, d = 16, 2, 1, 256
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
    with jax.sharding.use_abstract_mesh(mesh):
        cache = KB.get_kv_cache_shape(128, 16, nkv, d, jnp.float8_e4m3fn)
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
            )

        out, new_cache = jax.eval_shape(attend, *args)
        text = jax.jit(attend).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert out.shape == (tokens, nq, d) and new_cache.shape == cache
    assert "qkv_to_rpa_inputs" in text
