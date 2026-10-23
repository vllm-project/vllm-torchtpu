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

"""Production Qwen state precision across a full prefill and subsequent decode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [4096, 8192])
def test_qwen_prefill_then_decode_preserves_bf16_ssm_and_fp32_conv(tokens, monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels import pool_adapters
    from vllm_torchtpu.kernels.gdn.v3 import wrapper
    from vllm_torchtpu.layers.core import gdn_attention

    monkeypatch.setattr(envs, "TPU_GDN_CONV_QK_PAIR_LAYOUT", False)
    n_kq, n_v, dim, width, page = 16, 48, 128, 10240, 1664
    rng = np.random.default_rng(20261008)

    def normal(shape, dtype=jnp.bfloat16):
        return jnp.asarray(rng.normal(size=shape) * 0.02, dtype)

    indices = jnp.array([1], jnp.int32)
    conv = normal((2, 3, width), jnp.float32)
    ssm = normal((2, n_v, dim, dim))
    pool = jnp.zeros((2, page, 1, 4, 256), jnp.float8_e4m3fn)
    # Seed the native 256-lane carrier without a narrowing VMEM ref reshape.
    # The GDN state plan performs its own typed view inside the actual kernel.
    pool = pool_adapters.scatter_region(
        pool,
        ssm[1:].reshape(1, -1, 256),
        indices,
        tok0=0,
        ntok=1536,
        split=1,
    )
    pool = pool_adapters.scatter_region(
        pool,
        jnp.pad(conv[1:].reshape(1, 120, 256), ((0, 0), (0, 8), (0, 0))),
        indices,
        tok0=1536,
        ntok=128,
        split=1,
    )
    weight = normal((width, 1, 4))
    bias = normal((width,))
    a_log = normal((n_v,), jnp.float32)
    dt_bias = normal((n_v,), jnp.float32) - 2

    def dense(conv, ssm, qkv, b, a, context):
        return wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            conv,
            ssm,
            weight,
            bias,
            a_log,
            dt_bias,
            jnp.array([0, len(qkv)], jnp.int32),
            indices,
            jnp.array([0, 1, 1] if len(qkv) > 1 else [1, 1, 1], jnp.int32),
            jnp.array([context + len(qkv)], jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=dim,
            d_v=dim,
            kernel_size=4,
        )

    def pooled(pool, qkv, b, a, context):
        return gdn_attention.run_jax_gdn_attention_pooled_local(
            qkv,
            b,
            a,
            pool,
            weight,
            bias,
            a_log,
            dt_bias,
            jnp.array([0, len(qkv)], jnp.int32),
            indices,
            jnp.array([0, 1, 1] if len(qkv) > 1 else [1, 1, 1], jnp.int32),
            jnp.array([context + len(qkv)], jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=dim,
            d_v=dim,
            kernel_size=4,
            pool_block_tokens=page,
            recurrent_state_dtype=jnp.bfloat16,
            conv_state_dtype=jnp.float32,
        )

    run_dense, run_pooled = jax.jit(dense), jax.jit(pooled)
    context = 256  # Continue from nonzero, already-cached recurrent state.
    for count in (tokens, 1, 1, 1):
        qkv, b, a = normal((count, width)), normal((count, n_v)), normal((count, n_v))
        (conv, ssm), expected = run_dense(conv, ssm, qkv, b, a, context)
        pool, actual = run_pooled(pool, qkv, b, a, context)
        actual_ssm = pool_adapters.gather_region(
            pool,
            indices,
            tok0=0,
            ntok=1536,
            split=1,
            out_dtype=jnp.bfloat16,
            out_lanes=256,
        ).reshape(1, n_v, dim, dim)
        actual_conv = pool_adapters.gather_region(
            pool,
            indices,
            tok0=1536,
            ntok=128,
            split=1,
            out_dtype=jnp.float32,
        )[:, :120].reshape(1, 3, width)
        for got, wanted in (
            (actual, expected),
            (actual_ssm, ssm[1:]),
            (actual_conv, conv[1:]),
        ):
            np.testing.assert_allclose(
                np.asarray(got, np.float32),
                np.asarray(wanted, np.float32),
                rtol=0.02,
                atol=0.02,
            )
        context += count
