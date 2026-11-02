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
"""CPU contract tests for `ragged_paged_attention(fp8_compute=True)`.

The kernel's values need a device (`correctness_test.py`); these cover the host
side: the block geometry, the call validation, the query cast, and that the
mixed kernel still lowers for tpu7x with fp8 queries.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

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
        # Gemma4-31B global layers per TensorCore at TP8 / TP2 / TP1.
        ((4, 1), configs.BlockSizes(2048, 64, 512, 1)),
        ((16, 2), configs.BlockSizes(512, 32, 512, 1)),
        ((32, 4), configs.BlockSizes(256, 32, 512, 1)),
        # Gemma4-26B-A4B global layers at TP2.
        ((8, 1), configs.BlockSizes(1024, 32, 512, 1)),
    ],
)
def test_fp8_prefill_block_sizes(heads, expected):
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        blocks = wrapper.fp8_prefill_block_sizes(*_cfgs(*heads), 64 * 1024 * 1024)
    assert blocks == expected


@pytest.mark.parametrize("sliding_window", [1024, None])
def test_fp8_prefill_block_sizes_take_the_sliding_cap(sliding_window):
    model, serve = _cfgs(4, 1)
    model = dataclasses.replace(model, sliding_window=sliding_window)
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        blocks = wrapper.fp8_prefill_block_sizes(
            model, serve, 64 * 1024 * 1024, sliding_prefill_max_bkv=300
        )
    if sliding_window:
        # 300 rounds down to the 256-token MXU column stride and the 64-token chunk.
        assert blocks == configs.BlockSizes(256, 64, 256, 1)
    else:
        assert blocks == configs.BlockSizes(2048, 64, 512, 1)


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
    """The fp8 path builds and lowers both kernels without a device."""
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
