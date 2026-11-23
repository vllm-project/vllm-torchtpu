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
"""`TPU_KV_PACK_FUSED_PAD` on CPU: the padded pack equals pack-then-pad.

The SparseCore KV writer reads its input padded to a whole number of
descriptor blocks (for example 16 decode tokens become 32 rows). With the
flag off, `scatter_utils.scatter_rows` pads the packed uint32 rows. With the
flag on, `kv_cache_utils.pack_tokens(num_rows=...)` produces the padded rows
itself. These tests check, without a TPU, that the rows and destinations
handed to the SparseCore kernel are bit-identical either way. The kernel
itself needs a TPU; `test_mla_kv_cache_scatter.py` covers it there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter, scatter_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    SKIP_ROW,
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256


def _spec(cache_type, head_dim):
    return SparseMLAKVCacheSpec.create(
        cache_type, KVCacheLayout.SPARSECORE, 64, PAGE_SIZE, head_dim, 4
    )


def _fp8(num_tokens, dim, seed):
    """Random raw bytes as fp8, so every bit pattern, NaNs included, is used."""
    u8 = jax.random.randint(
        jax.random.PRNGKey(seed), (num_tokens, dim), 0, 256, jnp.uint8
    )
    return jax.lax.bitcast_convert_type(u8, jnp.float8_e4m3fn)


@pytest.mark.parametrize(
    "cache_type,head_dim", [(KVCacheType.NOPE, LKV_DIM), (KVCacheType.ROPE, ROPE_DIM)]
)
@pytest.mark.parametrize("num_tokens,num_rows", [(16, 32), (5, 16), (32, 32)])
def test_padded_pack_equals_pack_then_pad(cache_type, head_dim, num_tokens, num_rows):
    spec = _spec(cache_type, head_dim)
    values = _fp8(num_tokens, head_dim, num_tokens)

    unpadded = kv_cache_utils.pack_tokens(values, spec.token_bytes)
    ref = jnp.pad(unpadded, ((0, num_rows - num_tokens), (0, 0)))
    got = jax.jit(
        lambda v: kv_cache_utils.pack_tokens(v, spec.token_bytes, num_rows=num_rows)
    )(values)

    assert got.dtype == ref.dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_pad_to_plan_accepts_padded_or_unpadded_src():
    src = jnp.arange(16 * 4, dtype=jnp.uint32).reshape(16, 4)
    dst = jnp.arange(16, dtype=jnp.int32)
    a_src, a_dst = scatter_utils.pad_to_plan(src, dst, 32)
    b_src, b_dst = scatter_utils.pad_to_plan(jnp.pad(src, ((0, 16), (0, 0))), dst, 32)
    np.testing.assert_array_equal(np.asarray(a_src), np.asarray(b_src))
    np.testing.assert_array_equal(np.asarray(a_dst), np.asarray(b_dst))
    assert a_src.shape == (32, 4) and a_dst.shape == (32,)
    assert (np.asarray(a_dst)[16:] == SKIP_ROW).all()
    with pytest.raises(AssertionError, match="src has 20 rows"):
        scatter_utils.pad_to_plan(jnp.zeros((20, 4), jnp.uint32), dst, 32)


@pytest.mark.parametrize(
    "cache_type,head_dim", [(KVCacheType.NOPE, LKV_DIM), (KVCacheType.ROPE, ROPE_DIM)]
)
def test_scatter_hands_the_kernel_identical_inputs(monkeypatch, cache_type, head_dim):
    """`scatter.scatter` with the flag on and off gives the SparseCore kernel
    the same padded rows and destinations. The plan size is fixed at 32, the
    value for 16 tokens on tpu7x, because CPU has no SparseCore info."""
    spec = _spec(cache_type, head_dim)
    values = _fp8(16, head_dim, 1)
    dst_rows = jnp.arange(16, dtype=jnp.int32) * 3
    cache = jnp.zeros(spec.shape, spec.jax_dtype)
    seen = []

    def fake_scatter_rows(cache, src, dst_rows, name=""):
        seen.append(scatter_utils.pad_to_plan(src, dst_rows, 32))
        return cache

    monkeypatch.setattr(scatter_utils, "scatter_rows", fake_scatter_rows)
    monkeypatch.setattr(scatter_utils, "padded_num_tokens", lambda n: 32)

    monkeypatch.setenv("TPU_KV_PACK_FUSED_PAD", "0")
    scatter.scatter(cache, values, spec, dst_rows=dst_rows)
    monkeypatch.setenv("TPU_KV_PACK_FUSED_PAD", "1")
    scatter.scatter(cache, values, spec, dst_rows=dst_rows)

    (off_src, off_dst), (on_src, on_dst) = seen
    np.testing.assert_array_equal(np.asarray(on_src), np.asarray(off_src))
    np.testing.assert_array_equal(np.asarray(on_dst), np.asarray(off_dst))


def test_fused_pad_env_defaults_to_false(monkeypatch):
    import vllm_torchtpu.envs as envs

    monkeypatch.delenv("TPU_KV_PACK_FUSED_PAD", raising=False)
    assert envs.environment_variables["TPU_KV_PACK_FUSED_PAD"]() is False
