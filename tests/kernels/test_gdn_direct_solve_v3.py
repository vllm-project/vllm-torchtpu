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
"""CPU parity tests for the default-off VLLM_TPU_GDN_DIRECT_SOLVE path (Caslyn
Tonelli's GDN v3 direct-solve stack)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config


def _make_cfg(
    num_kq_heads: int = 2,
    num_v_heads: int = 16,
    kq_head_dim: int = 64,
    v_head_dim: int = 128,
    chunk_size: int = 32,
    compute_dtype: jnp.dtype = jnp.float32,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
) -> config.GDNConfig:
    dim_size = 2 * num_kq_heads * kq_head_dim + num_v_heads * v_head_dim
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=compute_dtype,
            act_out=compute_dtype,
            compute=compute_dtype,
            recurrent_state=jnp.float32,
            conv_state=jnp.float32,
        ),
        batch_size=chunk_size,
        dim_size=dim_size,
        kernel_size=4,
        tile_size=chunk_size,
        num_kq_heads=num_kq_heads,
        num_v_heads=num_v_heads,
        kq_head_dim=kq_head_dim,
        v_head_dim=v_head_dim,
        attention_mode=attention_mode,
    )


@pytest.mark.parametrize(
    "chunk,block_size",
    [(8, 8), (16, 8), (32, 8), (32, 16), (64, 16), (23, 8), (23, 16)],
)
def test_solve_triangular_rhs_matches_linalg_solve(chunk: int, block_size: int):
    rng = np.random.default_rng(100 + chunk * 7 + block_size)
    num_heads, rhs_width = 4, 16
    raw = rng.standard_normal((num_heads, chunk, chunk)).astype(np.float32) * 0.1
    strict_lower = np.tril(raw, k=-1)
    rhs = rng.standard_normal((num_heads, chunk, rhs_width)).astype(np.float32)

    got = compute_gdn.solve_triangular_rhs(
        jnp.asarray(strict_lower), jnp.asarray(rhs), block_size=block_size
    )
    eye = np.broadcast_to(np.eye(chunk, dtype=np.float32), strict_lower.shape)
    ref = jnp.linalg.solve(jnp.asarray(eye + strict_lower), jnp.asarray(rhs))
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_gdn_triangular_block_size_floor(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_GDN_DIRECT_SOLVE", raising=False)
    for n_v, expected_off, expected_on in [
        (4, 16, 16),
        (8, 16, 16),
        (16, 8, 8),
        (24, 4, 8),
        (32, 4, 8),
        (48, 2, 8),
        (64, 2, 8),
    ]:
        cfg = _make_cfg(num_kq_heads=2, num_v_heads=n_v)
        assert cfg.triangular_block_size == expected_off
        with monkeypatch.context() as mp:
            mp.setenv("VLLM_TPU_GDN_DIRECT_SOLVE", "1")
            assert cfg.triangular_block_size == expected_on
    kda_cfg = _make_cfg(
        num_kq_heads=2, num_v_heads=32, attention_mode=config.AttentionMode.KDA
    )
    base_kda = kda_cfg.triangular_block_size
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_TPU_GDN_DIRECT_SOLVE", "1")
        assert kda_cfg.triangular_block_size == base_kda


@pytest.mark.parametrize("v_per_kq", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("v_head_dim", [64, 128])
def test_dot_kq_with_v_heads_matches_naive(v_per_kq: int, v_head_dim: int):
    rng = np.random.default_rng(7 + v_per_kq * 13 + v_head_dim)
    num_kq = 2
    num_v = num_kq * v_per_kq
    m, k = 16, 32
    a = jnp.asarray(rng.standard_normal((num_kq, m, k)).astype(np.float32))
    b = jnp.asarray(rng.standard_normal((num_v, k, v_head_dim)).astype(np.float32))
    got = compute_gdn._dot_kq_with_v_heads(a, b, (((2,), (1,)), ((0,), (0,))))
    a_rep = jnp.repeat(a, v_per_kq, axis=0)
    ref = jax.lax.dot(
        a_rep,
        b,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "num_kq,num_v,kq_dim,v_dim,chunk,real_size",
    [
        (2, 16, 32, 128, 32, 32),  # Qwen3.8-like 8:1 group, paired V heads
        (1, 8, 32, 128, 32, 20),  # Partial masked tail
        (1, 8, 32, 128, 32, 12),  # Short real tail
        (2, 6, 32, 64, 16, 16),  # Odd v_per_kq=3, non-128 v_head_dim guard
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_chunked_gdn_direct_solve_matches_stock(
    monkeypatch, num_kq, num_v, kq_dim, v_dim, chunk, real_size, dtype
):
    aligned_v = 128
    seed = num_v * 31 + real_size * 7 + (1 if dtype == jnp.bfloat16 else 0)
    rng = np.random.default_rng(seed)
    cfg = _make_cfg(
        num_kq_heads=num_kq,
        num_v_heads=num_v,
        kq_head_dim=kq_dim,
        v_head_dim=v_dim,
        chunk_size=chunk,
        compute_dtype=dtype,
    )
    q = jnp.asarray(
        rng.standard_normal((1, num_kq, chunk, kq_dim)).astype(np.float32) * 0.5
    )
    k = jnp.asarray(
        rng.standard_normal((1, num_kq, chunk, kq_dim)).astype(np.float32) * 0.5
    )
    v = jnp.asarray(
        rng.standard_normal((1, num_v, chunk, v_dim)).astype(np.float32) * 0.5
    )
    b = jnp.asarray(
        rng.standard_normal((1, 1, chunk, aligned_v)).astype(np.float32) * 0.5
    )
    a = jnp.asarray(
        rng.standard_normal((1, 1, chunk, aligned_v)).astype(np.float32) * 0.3
    )
    s0 = jnp.asarray(
        rng.standard_normal((1, num_v, kq_dim, v_dim)).astype(np.float32) * 0.2
    )
    a_log = jnp.asarray(rng.uniform(-2.0, -0.5, size=(aligned_v,)).astype(np.float32))
    dt_bias = jnp.asarray(rng.uniform(-0.5, 0.5, size=(aligned_v,)).astype(np.float32))
    real_sizes = jnp.asarray([real_size], dtype=jnp.int32)

    monkeypatch.delenv("VLLM_TPU_GDN_DIRECT_SOLVE", raising=False)
    assert not envs.VLLM_TPU_GDN_DIRECT_SOLVE
    out_off, st_off = compute_gdn.chunked_gdn(
        real_sizes, q, k, v, b, a, s0, a_log, dt_bias, cfg
    )

    monkeypatch.setenv("VLLM_TPU_GDN_DIRECT_SOLVE", "1")
    assert envs.VLLM_TPU_GDN_DIRECT_SOLVE
    out_on, st_on = compute_gdn.chunked_gdn(
        real_sizes, q, k, v, b, a, s0, a_log, dt_bias, cfg
    )

    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(
        np.asarray(out_on), np.asarray(out_off), rtol=tol, atol=tol
    )
    np.testing.assert_allclose(
        np.asarray(st_on), np.asarray(st_off), rtol=tol, atol=tol
    )
    assert np.isfinite(np.asarray(out_on)).all()
    assert np.isfinite(np.asarray(st_on)).all()
    if real_size < chunk:
        np.testing.assert_array_equal(np.asarray(out_on)[:, real_size:], 0.0)


def test_default_flag_off_keeps_kernel_name(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_GDN_DIRECT_SOLVE", raising=False)
    cfg = _make_cfg(chunk_size=128)
    name_off = cfg.get_kernel_name()
    assert "_ds" not in name_off
    monkeypatch.setenv("VLLM_TPU_GDN_DIRECT_SOLVE", "1")
    assert cfg.get_kernel_name() == f"{name_off}_ds"
