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
"""Pooled GDN prefill must match the dense path in results and in cost.

The pooled state seam decodes the pool's byte layout into the kernel's state
layout. Doing that once per kernel tile instead of once per sequence is
invisible in correctness and expensive in prefill, where one sequence spans
many tiles. This runs both paths over the same 4096-token prefill at
production geometry and compares outputs, final states, and time.

TPU only. Timings print (use `-s`); the ratio is the assertion.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput
from vllm_torchtpu.kernels.qwen4_exp.gdn_input import compact_qkv_input
from vllm_torchtpu.layers.core import gdn_attention

pytestmark = pytest.mark.tpu_test

# Measured 1.043-1.049x over five runs; a per-tile decode lands near 2.2x.
# Arms are timed best-of-N because each is only ~7.5 ms.
MAX_POOLED_OVER_DENSE = 1.1
BENCH_REPEATS = 3

BENCH_N_KQ, BENCH_N_V = 16, 64
BENCH_D_K, BENCH_D_V = 128, 128
BENCH_KERNEL_SIZE = 4
BENCH_DIM = BENCH_N_KQ * BENCH_D_K * 2 + BENCH_N_V * BENCH_D_V
BENCH_NUM_TOKENS = 4096
BENCH_ITERATIONS = 5
BENCH_DECODE_TILE_SIZE = 4
BENCH_KBS, BENCH_LANES = 256, 256
BENCH_PAYLOAD = (1, 4)
BENCH_TOK_BYTES = int(np.prod(BENCH_PAYLOAD) * BENCH_LANES * np.dtype(np.int8).itemsize)
BENCH_SSM_NTOK = BENCH_N_V * BENCH_D_K * BENCH_D_V * 4 // BENCH_TOK_BYTES
# fp32 conv: 3 * 12288 * 4 = 144 KiB, rounded up to one 256-token kernel block.
BENCH_CONV_NTOK = 256
# Granule coordinates of the two regions -- all the kernel is told.
BENCH_GRANULE = 512
BENCH_WORDS_PER_GRANULE = BENCH_GRANULE // 4
BENCH_SSM_GRANULES = BENCH_N_V * BENCH_D_K * BENCH_D_V * 4 // BENCH_GRANULE
BENCH_MANAGER_TOKENS = BENCH_SSM_NTOK + BENCH_CONV_NTOK  # 4352
BENCH_SPLIT = BENCH_MANAGER_TOKENS // BENCH_KBS
BENCH_NUM_MGR = 2  # null slot and one active request
BENCH_POOL_SHAPE = (
    (
        BENCH_NUM_MGR * BENCH_SPLIT,
        BENCH_KBS,
    )
    + BENCH_PAYLOAD
    + (BENCH_LANES,)
)
BENCH_STATE_IDX = np.array([1], dtype=np.int32)
BENCH_QUERY_START_LOC = np.array([0, BENCH_NUM_TOKENS], dtype=np.int32)
BENCH_DISTRIBUTION = np.array([0, 1, 1], dtype=np.int32)
# Greater than the query length so every iteration consumes its input state.
BENCH_SEQ_LENS = np.array([BENCH_NUM_TOKENS + 1], dtype=np.int32)


def _benchmark_mgr_words(pool):
    """The pool the way the kernel addresses it: one uint32 row per state.

    A ref's `bitcast(uint32)` packs along the *second-minor* (sublane) axis,
    so for a pool row `(..., packing, lanes)` a word spans one lane's `packing`
    sublanes, not `packing` adjacent elements. Moving that axis minormost
    first is what makes the host view agree with the kernel's granule stream.
    """
    words = jnp.moveaxis(pool, -2, -1).view(jnp.uint32)
    return words.reshape(pool.shape[0] // BENCH_SPLIT, -1)


def _benchmark_unwords(words, pool):
    """Inverse of `_benchmark_mgr_words`."""
    moved = words.reshape(*pool.shape[:-2], pool.shape[-1], 1).view(pool.dtype)
    return jnp.moveaxis(moved, -2, -1).reshape(pool.shape)


@pytest.mark.parametrize("ssm_dtype", [jnp.bfloat16, jnp.float32])
def test_prepared_pooled_prefill_then_decode(ssm_dtype, monkeypatch):
    """Compare prepared and ordinary input through the real pooled bridge."""
    from vllm import envs as vllm_envs

    monkeypatch.setattr(vllm_envs, "VLLM_KV_CACHE_LAYOUT", "NHD")
    n_kq, n_v, dim, width = 16, 48, 128, 10240
    block_size, token_bytes, slots = 128, 1024, 3
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=n_v * dim * dim * jnp.dtype(ssm_dtype).itemsize,
        conv_bytes=3 * width * jnp.dtype(jnp.float32).itemsize,
        token_bytes=token_bytes,
    )
    manager_tokens = (
        (layout.required_tokens + block_size - 1) // block_size * block_size
    )
    split = manager_tokens // block_size
    raw = np.empty((slots, manager_tokens * token_bytes), np.uint8)
    # Finite, nonzero initial states with distinct active/inactive owners.
    for slot in range(slots):
        raw[slot].fill((0x34, 0x3C, 0x35)[slot])
    pool_shape = (slots * split, block_size, 1, 4, 256)

    def make_pool():
        return jax.lax.bitcast_convert_type(
            jnp.array(raw.reshape(pool_shape), copy=True), jnp.float8_e4m3fn
        )

    rng = np.random.default_rng(54)

    def normal(shape, dtype=jnp.bfloat16):
        return jnp.asarray(rng.normal(0, 0.02, shape), dtype)

    weight, bias = normal((width, 1, 4)), normal((width,))
    a_log = normal((n_v,), jnp.float32)
    dt_bias = normal((n_v,), jnp.float32) - 2
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))

    @functools.partial(jax.jit, static_argnames=("prepared",), donate_argnums=(0,))
    def run(pool, qkv, b, a, context, *, prepared):
        count = qkv.shape[0]
        inputs = Conv1DInput(compact_qkv_input(qkv), qkv.dtype) if prepared else qkv
        return gdn_attention.run_jax_gdn_attention_pooled(
            inputs,
            b,
            a,
            pool,
            weight,
            bias,
            a_log,
            dt_bias,
            jnp.array([1, 2], jnp.int32),
            jnp.array([0, count, count], jnp.int32),
            jnp.array([0, 1, 1] if count > 1 else [1, 1, 1], jnp.int32),
            jnp.array([context + count, 0], jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=dim,
            d_v=dim,
            kernel_size=4,
            pool_block_tokens=manager_tokens,
            mesh=mesh,
            recurrent_state_dtype=ssm_dtype,
        )

    def typed_rows(region, dtype):
        # Inverse Mosaic's second-minor byte planes, independent of pool readers.
        size = np.dtype(dtype).itemsize
        packed = region.reshape(-1, size, 256).transpose(0, 2, 1).copy()
        return packed.view(dtype).reshape(-1, 256)

    def read_states(bits):
        ssm_bytes = bits[1, : layout.ssm_tokens * token_bytes]
        if ssm_dtype == jnp.bfloat16:
            words = typed_rows(ssm_bytes, np.uint32).reshape(n_v, dim // 2, dim)
            values = np.empty((n_v, dim, dim), np.uint16)
            values[:, 0::2] = (words & 65535).astype(np.uint16)
            values[:, 1::2] = (words >> 16).astype(np.uint16)
            recurrent = values.view(jnp.bfloat16)
        else:
            recurrent = typed_rows(ssm_bytes, np.float32).reshape(n_v, dim, dim)
        begin = layout.ssm_tokens * token_bytes
        conv_bytes = bits[1, begin : begin + layout.conv_tokens * token_bytes]
        conv = typed_rows(conv_bytes, np.float32)[: 3 * width // 256].reshape(
            3, width
        )
        return conv, recurrent

    ordinary, prepared = make_pool(), make_pool()
    context = 256
    for count in (128, 1, 1, 1):
        qkv, b, a = normal((count, width)), normal((count, n_v)), normal((count, n_v))
        ordinary, expected = run(ordinary, qkv, b, a, context, prepared=False)
        prepared, actual = run(prepared, qkv, b, a, context, prepared=True)
        assert actual.dtype == expected.dtype == qkv.dtype
        np.testing.assert_array_equal(actual, expected)
        bits = [
            np.asarray(jax.lax.bitcast_convert_type(p, jnp.uint8)).reshape(slots, -1)
            for p in (ordinary, prepared)
        ]
        np.testing.assert_array_equal(bits[0], bits[1])
        for value in bits:
            np.testing.assert_array_equal(value[[0, 2]], raw[[0, 2]])
        for got, want in zip(read_states(bits[1]), read_states(bits[0])):
            assert got.dtype == want.dtype
            assert np.isfinite(got.astype(np.float32)).all()
            np.testing.assert_array_equal(got, want)
        assert not np.array_equal(bits[0][1], raw[1]), "active state was not updated"
        print(
            f"pooled prepared parity: ssm={jnp.dtype(ssm_dtype)}, tokens={count}, exact"
        )
        context += count


def _benchmark_write_states(pool, recurrent, conv):
    words = _benchmark_mgr_words(pool)
    for granule0, payload in (
        (0, jnp.asarray(recurrent, jnp.float32)),
        (BENCH_SSM_GRANULES, jnp.asarray(conv, jnp.float32)),
    ):
        body = payload.reshape(1, -1).view(jnp.uint32)
        start = granule0 * BENCH_WORDS_PER_GRANULE
        words = words.at[BENCH_STATE_IDX, start : start + body.shape[1]].set(body)
    return _benchmark_unwords(words, pool)


def _benchmark_read_states(pool):
    words = _benchmark_mgr_words(pool)

    def take(granule0, shape):
        start = granule0 * BENCH_WORDS_PER_GRANULE
        nwords = int(np.prod(shape))
        return (
            words[BENCH_STATE_IDX, start : start + nwords]
            .view(jnp.float32)
            .reshape(1, *shape)
        )

    recurrent = take(0, (BENCH_N_V, BENCH_D_K, BENCH_D_V))
    conv = take(BENCH_SSM_GRANULES, (BENCH_KERNEL_SIZE - 1, BENCH_DIM))
    return conv, recurrent


def _benchmark_common_args(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias):
    return dict(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=BENCH_QUERY_START_LOC,
        state_indices=BENCH_STATE_IDX,
        distribution=BENCH_DISTRIBUTION,
        seq_lens=BENCH_SEQ_LENS,
        n_kq=BENCH_N_KQ,
        n_v=BENCH_N_V,
        d_k=BENCH_D_K,
        d_v=BENCH_D_V,
        kernel_size=BENCH_KERNEL_SIZE,
        decode_tile_size=BENCH_DECODE_TILE_SIZE,
    )


@functools.partial(jax.jit, donate_argnums=(0, 1))
def _benchmark_dense_five(
    conv_state, recurrent_state, qkv, b, a, conv_weight, conv_bias, a_log, dt_bias
):
    def step(states, _):
        new_states, output = wrapper.fused_conv1d_gdn(
            states=states,
            **_benchmark_common_args(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias),
        )
        return new_states, output

    return jax.lax.scan(
        step,
        (conv_state, recurrent_state),
        xs=None,
        length=BENCH_ITERATIONS,
    )


@functools.partial(jax.jit, donate_argnums=(0,))
def _benchmark_pooled_five(pool, qkv, b, a, conv_weight, conv_bias, a_log, dt_bias):
    def step(state_pool, _):
        state_pool, output = gdn_attention.run_jax_gdn_attention_pooled_local(
            mixed_qkv=qkv,
            b=b,
            a=a,
            recurrent_state=state_pool,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            A_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=BENCH_QUERY_START_LOC,
            state_indices=BENCH_STATE_IDX,
            distribution=BENCH_DISTRIBUTION,
            seq_lens=BENCH_SEQ_LENS,
            n_kq=BENCH_N_KQ,
            n_v=BENCH_N_V,
            d_k=BENCH_D_K,
            d_v=BENCH_D_V,
            kernel_size=BENCH_KERNEL_SIZE,
            pool_block_tokens=BENCH_MANAGER_TOKENS,
        )
        return state_pool, output

    return jax.lax.scan(
        step,
        pool,
        xs=None,
        length=BENCH_ITERATIONS,
    )


def _block_until_ready(tree):
    return jax.tree.map(lambda x: x.block_until_ready(), tree)


def _time_five_steps(run, make_args):
    """Best of BENCH_REPEATS. `make_args` rebuilds the arguments each time
    because the step functions donate their state buffers."""
    best = None
    result = None
    for _ in range(BENCH_REPEATS):
        args = make_args()
        start = time.perf_counter()
        result = _block_until_ready(run(*args))
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    return result, best


class TestProductionShapeDenseVsUnifiedPool:
    """Production-shape correctness and performance comparison.

    This reproduces the Qwen3.5-397B GDN geometry used by the end-to-end DP
    prefill benchmark: 16 KQ heads, 64 V heads, 128-d head dimensions, 4096
    tokens, a 4352-token unified-pool manager block, and decode_tile_size=4.
    Dense and pooled modes receive identical activations, weights, and logical
    initial states. Each mode runs one five-iteration single-JIT warmup,
    followed by a fresh five-iteration single-JIT timed run.

    Reference TPU observation on 2026-07-28:
      dense:  1.500 ms/iteration
      pooled: 1.492 ms/iteration
      change: -0.008 ms/iteration (0.995x)

    Timing remains diagnostic rather than an assertion because shared TPU load
    can vary. Outputs from all five iterations and both final states are strict
    correctness checks; the reference run had zero maximum absolute error.
    """

    def test_prefill_five_step_correctness_and_timing(self):
        keys = iter(jax.random.split(jax.random.key(20260728), 10))

        def normal(shape, scale):
            value = jax.random.normal(next(keys), shape, dtype=jnp.bfloat16)
            return value * jnp.asarray(scale, dtype=jnp.bfloat16)

        qkv = normal((BENCH_NUM_TOKENS, BENCH_DIM), 0.05)
        b = normal((BENCH_NUM_TOKENS, BENCH_N_V), 0.05)
        a = normal((BENCH_NUM_TOKENS, BENCH_N_V), 0.05)
        conv_weight = normal((BENCH_DIM, 1, BENCH_KERNEL_SIZE), 0.02)
        conv_bias = normal((BENCH_DIM,), 0.01)
        a_log = normal((BENCH_N_V,), 0.02).astype(jnp.float32)
        dt_bias = normal((BENCH_N_V,), 0.02).astype(jnp.float32) - 2.0
        initial_conv = normal((1, BENCH_KERNEL_SIZE - 1, BENCH_DIM), 0.01)
        initial_recurrent = normal(
            (1, BENCH_N_V, BENCH_D_K, BENCH_D_V),
            0.01,
        ).astype(jnp.float32)

        dense_conv = (
            jnp.zeros(
                (BENCH_NUM_MGR, BENCH_KERNEL_SIZE - 1, BENCH_DIM),
                dtype=jnp.float32,
            )
            .at[1]
            .set(initial_conv[0])
        )
        dense_recurrent = (
            jnp.zeros(
                (BENCH_NUM_MGR, BENCH_N_V, BENCH_D_K, BENCH_D_V),
                dtype=jnp.float32,
            )
            .at[1]
            .set(initial_recurrent[0])
        )
        pool = _benchmark_write_states(
            jnp.zeros(BENCH_POOL_SHAPE, dtype=jnp.int8),
            initial_recurrent,
            initial_conv,
        )

        shared_args = (qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)
        _block_until_ready(
            _benchmark_dense_five(
                jnp.copy(dense_conv), jnp.copy(dense_recurrent), *shared_args
            )
        )
        _block_until_ready(_benchmark_pooled_five(jnp.copy(pool), *shared_args))

        # Fresh, synchronized state keeps copies and initialization out of the
        # timed region. Each timed call executes five dependent iterations in
        # one compiled JIT.
        dense_result, dense_elapsed = _time_five_steps(
            _benchmark_dense_five,
            lambda: (
                jnp.copy(dense_conv).block_until_ready(),
                jnp.copy(dense_recurrent).block_until_ready(),
                *shared_args,
            ),
        )
        pooled_result, pooled_elapsed = _time_five_steps(
            _benchmark_pooled_five,
            lambda: (jnp.copy(pool).block_until_ready(), *shared_args),
        )

        (dense_final_conv, dense_final_recurrent), dense_outputs = dense_result
        pooled_final, pooled_outputs = pooled_result
        pooled_final_conv, pooled_final_recurrent = _benchmark_read_states(pooled_final)
        _block_until_ready((pooled_final_conv, pooled_final_recurrent))

        output_tol = dict(rtol=2e-2, atol=2e-2)
        state_tol = dict(rtol=2e-2, atol=2e-2)
        assert bool(jnp.allclose(pooled_outputs, dense_outputs, **output_tol))
        assert bool(
            jnp.allclose(
                pooled_final_conv.astype(jnp.float32),
                dense_final_conv[1:2].astype(jnp.float32),
                **state_tol,
            )
        )
        assert bool(
            jnp.allclose(
                pooled_final_recurrent,
                dense_final_recurrent[1:2],
                **state_tol,
            )
        )

        dense_ms = dense_elapsed * 1e3
        pooled_ms = pooled_elapsed * 1e3
        ratio = pooled_elapsed / dense_elapsed
        print(
            f"\n{BENCH_NUM_TOKENS} tokens x {BENCH_ITERATIONS} iters: "
            f"dense {dense_ms / BENCH_ITERATIONS:.3f} ms/iter, "
            f"pooled {pooled_ms / BENCH_ITERATIONS:.3f} ms/iter, "
            f"{ratio:.3f}x"
        )

        assert ratio < MAX_POOLED_OVER_DENSE, (
            f"pooled prefill is {ratio:.3f}x dense (limit "
            f"{MAX_POOLED_OVER_DENSE}); the pooled state seam may be decoding "
            "the source layout once per tile again"
        )
