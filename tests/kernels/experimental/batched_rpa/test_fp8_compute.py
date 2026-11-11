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
"""CPU contract tests for `ragged_paged_attention(fp8_compute=True)`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    flash_attention,
    wrapper,
)

FP8 = jnp.float8_e4m3fn


def _cfgs(num_q_heads, num_kv_heads, dtype_q=FP8):
    model = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=512,
        mask_value=-1e30,
    )
    serve = configs.ServingConfigs(
        num_seqs=8,
        page_size=256,
        total_q_tokens=16384,
        num_page_indices=8 * 640,
        dtype_q=dtype_q,
        dtype_kv=FP8,
        dtype_out=jnp.bfloat16,
        fp8_compute=True,
    )
    return model, serve


@pytest.mark.parametrize(
    "heads,expected",
    [
        ((4, 1), configs.BlockSizes(2048, 64, 512, 1)),
        ((16, 2), configs.BlockSizes(512, 32, 512, 1)),
        ((32, 4), configs.BlockSizes(256, 32, 512, 1)),
        ((8, 1), configs.BlockSizes(1024, 32, 512, 1)),
    ],
)
def test_fp8_prefill_block_sizes(heads, expected):
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        blocks = wrapper.fp8_prefill_block_sizes(*_cfgs(*heads), 64 * 1024 * 1024)
    assert blocks == expected


def test_fp8_prefill_block_sizes_preserves_hd128_default():
    model = configs.ModelConfigs(
        num_q_heads=32, num_kv_heads=8, head_dim=128, mask_value=-1e30
    )
    _, serve = _cfgs(32, 8)
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        got = wrapper.fp8_prefill_block_sizes(model, serve, 64 * 1024 * 1024)
        want = wrapper.calculate_block_sizes(model, serve, 64 * 1024 * 1024)[1]
    assert got == want
    assert got.batch_size == 2


def test_fp8_softmax_shift_preserves_small_tail_finite_output():
    logits = jnp.array([-40.0, -8.0, -5.0, -3.5, 0.0], dtype=jnp.float32)
    m = jnp.max(logits).astype(jnp.bfloat16)
    m_sub = (m - flash_attention.P_FP8_SHIFT).astype(jnp.float32)
    p_fp8 = jnp.exp(logits - m_sub).astype(FP8)
    rowsum = jnp.sum(p_fp8.astype(jnp.bfloat16), dtype=jnp.bfloat16)
    assert jnp.all(jnp.isfinite(p_fp8.astype(jnp.float32)))
    assert float(jnp.max(p_fp8.astype(jnp.float32))) <= float(jnp.finfo(FP8).max)
    assert float(rowsum) > 0.0
    assert float(p_fp8[1].astype(jnp.float32)) > 0.0


def test_fp8_prefill_block_sizes_shrink_to_fit_vmem():
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        blocks = wrapper.fp8_prefill_block_sizes(*_cfgs(4, 1), 32 * 1024 * 1024)
        cfgs = configs.RpaConfigs(
            block=blocks,
            model=_cfgs(4, 1)[0],
            serve=_cfgs(4, 1)[1],
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=32 * 1024 * 1024,
        )
        used = wrapper.fp8_prefill_vmem_bytes(cfgs)
    assert blocks.bq_sz < 2048
    assert used <= wrapper.FP8_PREFILL_BUFFER_VMEM_FRACTION * 32 * 1024 * 1024


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(kv_dtype=jnp.bfloat16), "fp8 KV cache"),
        (dict(num_q_heads=4, num_kv_heads=2), "multiple of 4"),
        (dict(q_scale=0.5), "q_scale"),
        (dict(return_lse=True), "return_lse"),
        (dict(use_per_token_scale=True), "per-token"),
    ],
)
def test_validate_fp8_compute_rejects(kwargs, match):
    args = dict(
        kv_dtype=FP8, num_q_heads=32, num_kv_heads=4, q_scale=None, return_lse=False
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        wrapper.validate_fp8_compute(**args)


def test_quantize_queries_keeps_head_order_and_saturates():
    kv_heads, tokens, g, d = 2, 3, 8, 128
    q = jnp.arange(kv_heads * tokens * g * d, dtype=jnp.float32)
    q = (q.reshape(kv_heads, tokens, g, d) % 97 - 48) / 8.0
    q = q.at[0, 0, 0, 0].set(1000.0).at[1, 2, 7, 5].set(-1000.0)
    q_hbm = q.astype(jnp.bfloat16).reshape(kv_heads, tokens, g // 2, 2, d)

    q8 = wrapper.quantize_queries(q_hbm, FP8)

    assert q8.shape == (kv_heads, tokens, g // 4, 4, d)
    assert q8.dtype == FP8
    got = np.asarray(q8.astype(jnp.float32)).reshape(kv_heads, tokens, g, d)
    want = np.array(q.astype(jnp.bfloat16).astype(FP8).astype(jnp.float32))
    want[0, 0, 0, 0] = 448.0
    want[1, 2, 7, 5] = -448.0
    np.testing.assert_array_equal(got, want)


def test_fp8_mixed_kernel_lowers_for_tpu7x():
    tokens, num_q_heads, num_kv_heads, head_dim, page_size = 1024, 4, 1, 512, 256
    max_seqs, pages_per_seq = 8, 32
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        cache_shape = wrapper.get_kv_cache_shape(
            max_seqs * pages_per_seq, page_size, num_kv_heads, head_dim, FP8
        )
        args = (
            jax.ShapeDtypeStruct((tokens, num_q_heads, head_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, num_kv_heads, head_dim), FP8),
            jax.ShapeDtypeStruct((tokens, num_kv_heads, head_dim), FP8),
            jax.ShapeDtypeStruct(cache_shape, FP8),
            jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
            jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
            jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        )

        def call(*xs):
            return wrapper.ragged_paged_attention(
                *xs, sm_scale=1.0, k_scale=1.0, v_scale=1.0, fp8_compute=True
            )

        lowered = jax.jit(call).trace(*args).lower(lowering_platforms=("tpu",))
    text = lowered.as_text()
    assert "RPAm-snh-p256-b1-q2048-k512" in text
    assert "RPAd-snh-p256" in text


def _mixed_args(tokens=1024, num_q_heads=4, num_kv_heads=1, head_dim=512):
    max_seqs, pages_per_seq, page_size = 8, 32, 256
    cache_shape = wrapper.get_kv_cache_shape(
        max_seqs * pages_per_seq, page_size, num_kv_heads, head_dim, FP8
    )
    return (
        jax.ShapeDtypeStruct((tokens, num_q_heads, head_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, num_kv_heads, head_dim), FP8),
        jax.ShapeDtypeStruct((tokens, num_kv_heads, head_dim), FP8),
        jax.ShapeDtypeStruct(cache_shape, FP8),
        jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
        jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((3,), jnp.int32),
    )


def test_prepared_fp8_queries_replace_quantize_queries():
    args = _mixed_args()
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))

    def call(*xs, fp8_compute=True, sm_scale=1.0, **kwargs):
        prepare = jax.tree.map(
            jnp.zeros_like,
            jax.eval_shape(
                lambda q, k, v: wrapper.prepare_inputs(q, k, v, jnp.bfloat16, FP8),
                *xs[:3],
            ),
        )
        q_fetch = wrapper.quantize_queries(prepare[0], FP8)
        return wrapper.ragged_paged_attention(
            *xs,
            sm_scale=sm_scale,
            k_scale=1.0,
            v_scale=1.0,
            fp8_compute=fp8_compute,
            prepared_inputs=(*prepare, q_fetch),
            **kwargs,
        )

    with use_abstract_mesh(mesh):
        text = jax.jit(call).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
        assert "RPAm-snh-p256-b1-q2048-k512" in text
        with pytest.raises(ValueError, match="fp8_compute"):
            jax.jit(lambda *xs: call(*xs, fp8_compute=False)).trace(*args)
