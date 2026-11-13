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
"""Chunked GDN v3 compute at any chunk size and value-head count.

`fused_conv1d_gdn` clamps `mixed_tile_size` to the token count, so a prefill
of e.g. 24 tokens runs the chunked path with `chunk_size == 24`. A blocked
forward substitution in `invert_triangular_matrix` only covers whole blocks,
which breaks the chunked solve whenever the block size does not divide the
chunk. These tests run the pure-jnp compute on CPU (exact fp32 matmuls)
against a float64 reference, so they need no TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config

# Powers of two are controls; 128 is the largest dynamic-tiling mixed tile.
_CHUNK_SIZES = [8, 16, 20, 24, 31, 32, 40, 42, 48, 64, 128]
# (num_kq_heads, num_v_heads): a GQA slice, and 24 value heads (48 split over
# TP=2), for which a sub-block size tuned to the head count would be 5.
_HEADS = [(2, 4), (8, 24)]
_NUM_LANES = 128


def _cpu():
    return jax.default_device(jax.devices("cpu")[0])


def _cfg(chunk: int, n_kq: int, n_v: int, d_k: int,
         d_v: int) -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.float32.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=chunk,
        dim_size=2 * n_kq * d_k + n_v * d_v,
        kernel_size=4,
        tile_size=chunk,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
    )


@pytest.mark.parametrize("chunk", _CHUNK_SIZES)
def test_invert_triangular_matrix_any_chunk(chunk):
    rng = np.random.default_rng(chunk)
    num_heads = 4
    strict = np.tril(rng.uniform(-0.5, 0.5, (num_heads, chunk, chunk)), -1)
    # The solve treats the diagonal as 1 and must ignore what is stored there.
    t = (strict + 3.0 * np.eye(chunk)[None]).astype(np.float32)

    with _cpu():
        inv = np.asarray(
            jax.jit(compute_gdn.invert_triangular_matrix)(jnp.asarray(t)))

    expected = np.linalg.inv(
        strict.astype(np.float32).astype(np.float64) + np.eye(chunk)[None])
    assert inv.shape == t.shape
    np.testing.assert_allclose(inv, expected, rtol=1e-4, atol=1e-5)


def _gdn_reference(q, k, v, b, a, state, a_log, dt_bias, real_size,
                   v_per_kq_head):
    """Token-by-token float64 GDN recurrence for one sequence."""
    q, k, v, b, a, state = (np.asarray(x, np.float64)
                            for x in (q, k, v, b, a, state))
    a_log = np.asarray(a_log, np.float64)
    dt_bias = np.asarray(dt_bias, np.float64)
    num_v_heads, chunk, d_v = v.shape
    d_k = q.shape[-1]

    def l2_norm(x):
        return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + 1e-6)

    out = np.zeros((chunk, num_v_heads, d_v))
    state = state.copy()
    for h in range(num_v_heads):
        kq_head = h // v_per_kq_head
        for t in range(real_size):
            q_t = l2_norm(q[kq_head, t]) * d_k**-0.5
            k_t = l2_norm(k[kq_head, t])
            beta = 1.0 / (1.0 + np.exp(-b[t, h]))
            g = -np.exp(a_log[h]) * np.logaddexp(0.0, a[t, h] + dt_bias[h])
            state[h] *= np.exp(g)
            v_new = beta * (v[h, t] - k_t @ state[h])
            state[h] += np.outer(k_t, v_new)
            out[t, h] = q_t @ state[h]
    return out, state


def _run_chunked_gdn(cfg, real_size, q, k, v, b, a, state, a_log, dt_bias):
    """`chunked_gdn` on one sequence, on CPU."""

    def run(*args):
        return compute_gdn.chunked_gdn(*args, cfg=cfg)

    with _cpu():
        out, new_state = jax.jit(run)(
            jnp.asarray([real_size], jnp.int32),
            jnp.asarray(q[None]),
            jnp.asarray(k[None]),
            jnp.asarray(v[None]),
            jnp.asarray(b[None, None]),
            jnp.asarray(a[None, None]),
            jnp.asarray(state[None]),
            jnp.asarray(a_log),
            jnp.asarray(dt_bias),
        )
    return np.asarray(out), np.asarray(new_state)


@pytest.mark.parametrize("chunk", _CHUNK_SIZES)
@pytest.mark.parametrize("real_size_offset", [0, 3])
@pytest.mark.parametrize("n_kq, n_v", _HEADS)
def test_chunked_gdn_matches_recurrence(chunk, real_size_offset, n_kq, n_v):
    d_k, d_v = 128, 128
    cfg = _cfg(chunk, n_kq, n_v, d_k, d_v)
    real_size = chunk - real_size_offset

    rng = np.random.default_rng(1000 * chunk + 10 * real_size + n_v)
    q = rng.standard_normal((n_kq, chunk, d_k)).astype(np.float32)
    k = rng.standard_normal((n_kq, chunk, d_k)).astype(np.float32)
    v = rng.standard_normal((n_v, chunk, d_v)).astype(np.float32)
    # Gates are lane-padded to the aligned number of value heads.
    b = rng.standard_normal((chunk, _NUM_LANES)).astype(np.float32)
    a = rng.standard_normal((chunk, _NUM_LANES)).astype(np.float32)
    a_log = rng.standard_normal((_NUM_LANES, )).astype(np.float32)
    dt_bias = rng.standard_normal((_NUM_LANES, )).astype(np.float32)
    state = (0.1 * rng.standard_normal((n_v, d_k, d_v))).astype(np.float32)

    out, new_state = _run_chunked_gdn(cfg, real_size, q, k, v, b, a, state,
                                      a_log, dt_bias)

    expected_out, expected_state = _gdn_reference(q, k, v, b, a, state, a_log,
                                                  dt_bias, real_size,
                                                  cfg.v_per_kq_head)
    assert out.shape == (1, chunk, n_v, d_v)
    assert new_state.shape == (1, 1, n_v, d_k, d_v)
    np.testing.assert_allclose(out[0, :real_size],
                               expected_out[:real_size],
                               rtol=1e-4,
                               atol=1e-4)
    np.testing.assert_allclose(new_state[0, 0],
                               expected_state,
                               rtol=1e-4,
                               atol=1e-4)


@pytest.mark.parametrize("chunk, real_size", [(64, 19), (64, 44), (64, 63),
                                              (128, 100)])
def test_chunked_gdn_padding_leaves_newest_write_undecayed(chunk, real_size):
    """Rows past `real_size` add no decay, so the newest write stays undecayed.

    Only the last real token has a non-zero key, so the final state is that
    token's write, which no decay applies to. The cumulative log-decay must
    therefore end, at the chunk's last (padded) row, exactly where the last
    real row does. A sum that regroups its terms per row ends a few ulps off
    and scales the write by that; with a strong decay those ulps are large.
    """
    # Qwen3.5-35B-A3B's heads: 32 value heads give 32 decay sequences, so a
    # last row that is off for some of them is caught at any seed.
    n_kq, n_v, d_k, d_v = 16, 32, 128, 128
    cfg = _cfg(chunk, n_kq, n_v, d_k, d_v)

    rng = np.random.default_rng(chunk + real_size)
    q = rng.standard_normal((n_kq, chunk, d_k)).astype(np.float32)
    k = rng.standard_normal((n_kq, chunk, d_k)).astype(np.float32)
    k[:, :real_size - 1] = 0.0
    v = rng.standard_normal((n_v, chunk, d_v)).astype(np.float32)
    b = rng.standard_normal((chunk, _NUM_LANES)).astype(np.float32)
    # A log-decay of about -2 per token.
    a = (2.0 + rng.standard_normal((chunk, _NUM_LANES))).astype(np.float32)
    a_log = (0.5 * rng.standard_normal((_NUM_LANES, ))).astype(np.float32)
    dt_bias = np.zeros((_NUM_LANES, ), np.float32)
    state = np.zeros((n_v, d_k, d_v), np.float32)

    _, new_state = _run_chunked_gdn(cfg, real_size, q, k, v, b, a, state,
                                    a_log, dt_bias)
    _, expected_state = _gdn_reference(q, k, v, b, a, state, a_log, dt_bias,
                                       real_size, cfg.v_per_kq_head)
    # fp32 rounding of the write itself is about 2e-7 of its peak.
    np.testing.assert_allclose(
        new_state[0, 0],
        expected_state,
        rtol=0,
        atol=5e-7 * np.abs(expected_state).max(),
    )
