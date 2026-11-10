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

Kernel comparisons require TPU. The environment-isolation regression is CPU-only.
Timings print (use `-s`); the ratio is the assertion.
"""

import functools
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.layers.core import gdn_attention

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


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    (
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "state_dtype",
        "tile_rows",
        "dynamic_tiling",
        "expect_packed",
        "input_dtype",
    ),
    [
        (16, 96, 128, 128, 4, jnp.bfloat16, 32, True, True, jnp.bfloat16),
        (16, 32, 128, 128, 4, jnp.float32, 128, True, True, jnp.bfloat16),
        (8, 8, 128, 128, 4, jnp.bfloat16, 128, True, True, jnp.bfloat16),
        (4, 8, 384, 128, 9, jnp.bfloat16, 128, True, True, jnp.bfloat16),
        (4, 8, 384, 128, 9, jnp.bfloat16, 64, False, True, jnp.bfloat16),
        (4, 8, 128, 128, 66, jnp.bfloat16, 64, False, False, jnp.bfloat16),
        (16, 48, 128, 128, 4, jnp.float32, 64, False, False, jnp.float32),
    ],
)
def test_pooled_packed_prefill_fuses_with_tiling(
    monkeypatch,
    n_kq,
    n_v,
    d_k,
    d_v,
    kernel_size,
    state_dtype,
    tile_rows,
    dynamic_tiling,
    expect_packed,
    input_dtype,
):
    if jax.devices()[0].device_kind != "TPU7x":
        pytest.skip("These dynamic tile sizes are validated on TPU v7x.")
    monkeypatch.setattr(wrapper.envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", dynamic_tiling)
    monkeypatch.setattr(gdn_attention.envs, "VLLM_KV_CACHE_LAYOUT", "NHD")
    # The tiling env is captured by the wrapper when JAX traces it.
    jax.clear_caches()
    tokens = 4096
    qkv_dim = 2 * n_kq * d_k + n_v * d_v
    keys = iter(jax.random.split(jax.random.key(38), 4))
    qkvz = jax.random.normal(
        next(keys), (tokens, qkv_dim + n_v * d_v), dtype=input_dtype
    ) * input_dtype(0.1)
    b, a = [
        jax.random.normal(next(keys), (tokens, n_v), dtype=jnp.bfloat16)
        for _ in range(2)
    ]
    weights = jax.random.normal(next(keys), (qkv_dim, 1, kernel_size)) * 0.1
    state_bytes = n_v * d_k * d_v * jnp.dtype(state_dtype).itemsize
    conv_bytes = (kernel_size - 1) * qkv_dim * 4
    pool_rows = (state_bytes + conv_bytes + 511) // 512

    @jax.jit(donate_argnums=(1,))
    def run(activation, pool, b, a, weights, query_start_loc, distribution, seq_lens):
        return gdn_attention.run_jax_gdn_attention_pooled_local(
            activation,
            b,
            a,
            pool,
            weights,
            None,
            jnp.zeros(n_v),
            jnp.zeros(n_v),
            query_start_loc,
            jnp.array([1, 2], dtype=jnp.int32),
            distribution,
            seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            pool_block_tokens=pool_rows,
            recurrent_state_dtype=state_dtype,
        )

    def args(activation, starts, distribution, has_initial_state, pool_dtype):
        words = np.zeros((3, pool_rows * 128), dtype=np.uint32)
        if has_initial_state:
            rng = np.random.default_rng(327)
            recurrent = rng.uniform(-0.1, 0.1, (2, n_v, d_k, d_v)).astype(state_dtype)
            if state_dtype == jnp.bfloat16:
                bits = recurrent.view(np.uint16).astype(np.uint32)
                recurrent_words = bits[..., ::2, :] | (
                    bits[..., 1::2, :] << np.uint32(16)
                )
            else:
                recurrent_words = recurrent.view(np.uint32)
            conv = rng.uniform(-0.1, 0.1, (2, kernel_size - 1, qkv_dim)).astype(
                np.float32
            )
            rec_end = state_bytes // 4
            words[1:, :rec_end] = recurrent_words.reshape(2, -1)
            words[1:, rec_end : rec_end + conv_bytes // 4] = conv.view(
                np.uint32
            ).reshape(2, -1)
        # Ref bitcasts pack the pool's second-minor axis into uint32 words.
        pool = np.ascontiguousarray(
            np.moveaxis(words.reshape(3, pool_rows, 1, 128, 1).view(np.int8), -1, -2)
        )
        pool = jax.device_put(pool.view(np.dtype(pool_dtype)))
        seq_lens = np.diff(starts) + int(has_initial_state)
        return (
            activation,
            pool,
            b,
            a,
            weights,
            jnp.array(starts, dtype=jnp.int32),
            jnp.array(distribution, dtype=jnp.int32),
            jnp.array(seq_lens, dtype=jnp.int32),
        )

    for pool_dtype in (jnp.int8, jnp.float8_e4m3fn):
        packed_args = args(qkvz, [0, tokens, tokens], [0, 0, 1], False, pool_dtype)
        packed = run.lower(*packed_args).compile()
        kernel_names = set(
            re.findall(
                r"fused_conv1d_gdn_per_seq_b1_c\d+_packed_tile_cast_row_tiled",
                packed.as_text(),
            )
        )
        expected_kernel = (
            f"fused_conv1d_gdn_per_seq_b1_c{tile_rows}_packed_tile_cast_row_tiled"
        )
        assert kernel_names == ({expected_kernel} if expect_packed else set())
        for starts, distribution, has_initial_state in (
            ([0, tokens, tokens], [0, 0, 1], False),
            ([0, 2 * tile_rows, tokens], [0, 0, 2], True),
            ([0, tile_rows + 1, tokens], [0, 0, 2], True),
            ([0, 1, tokens], [1, 1, 2], True),
        ):
            expected_pool, expected = run(
                *args(
                    qkvz[:, :qkv_dim],
                    starts,
                    distribution,
                    has_initial_state,
                    pool_dtype,
                )
            )
            actual_pool, actual = packed(
                *args(qkvz, starts, distribution, has_initial_state, pool_dtype)
            )
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            np.testing.assert_array_equal(
                np.asarray(actual_pool).view(np.uint8),
                np.asarray(expected_pool).view(np.uint8),
            )


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


@pytest.mark.tpu_test
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


# Internal prefill checkpoint at the same geometry: a fresh 4096-token prefill
# that checkpoints after 3072 tokens (the logical Mamba block under PP4xPCP8)
# into a second manager block.
CKPT_OFFSET = 3072
CKPT_STATE_BLOCK = 1
CKPT_BLOCK = 2
CKPT_NUM_MGR = 3  # null slot, the request's state block, its checkpoint block
CKPT_POOL_SHAPE = (
    (CKPT_NUM_MGR * BENCH_SPLIT, BENCH_KBS) + BENCH_PAYLOAD + (BENCH_LANES,)
)


def _fresh_prefill(pool, qkv, b, a, weights, *, state_block, ckpt):
    conv_weight, conv_bias, a_log, dt_bias = weights
    num_tokens = qkv.shape[0]
    extra = {}
    if ckpt:
        extra = dict(
            prefill_ckpt_offsets=jnp.array([CKPT_OFFSET], dtype=jnp.int32),
            prefill_ckpt_indices=jnp.array([CKPT_BLOCK], dtype=jnp.int32),
        )
    return gdn_attention.run_jax_gdn_attention_pooled_local(
        mixed_qkv=qkv,
        b=b,
        a=a,
        recurrent_state=pool,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        A_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=jnp.array([0, num_tokens], dtype=jnp.int32),
        state_indices=jnp.array([state_block], dtype=jnp.int32),
        distribution=BENCH_DISTRIBUTION,
        seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        n_kq=BENCH_N_KQ,
        n_v=BENCH_N_V,
        d_k=BENCH_D_K,
        d_v=BENCH_D_V,
        kernel_size=BENCH_KERNEL_SIZE,
        pool_block_tokens=BENCH_MANAGER_TOKENS,
        **extra,
    )


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("ckpt",))
def _checkpoint_prefill_five(pool, qkv, b, a, *weights, ckpt):
    def step(state_pool, _):
        return _fresh_prefill(
            state_pool, qkv, b, a, weights, state_block=CKPT_STATE_BLOCK, ckpt=ckpt
        )

    return jax.lax.scan(step, pool, xs=None, length=BENCH_ITERATIONS)


@functools.partial(jax.jit, donate_argnums=(0,))
def _first_chunk(pool, qkv, b, a, *weights):
    return _fresh_prefill(
        pool,
        qkv[:CKPT_OFFSET],
        b[:CKPT_OFFSET],
        a[:CKPT_OFFSET],
        weights,
        state_block=CKPT_BLOCK,
        ckpt=False,
    )


def _manager_block(pool, mgr):
    return np.asarray(pool[mgr * BENCH_SPLIT : (mgr + 1) * BENCH_SPLIT])


class TestProductionShapePrefillCheckpoint:
    """Checkpoint contents at production geometry.

    The checkpoint block must hold the bytes a real 3072-token first chunk
    writes as its final state, and the outputs and every other block must
    match the same prefill without a checkpoint.
    """

    def test_checkpoint_matches_first_chunk(self):
        keys = iter(jax.random.split(jax.random.key(20261015), 7))

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
        shared_args = (qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)
        pool = jnp.zeros(CKPT_POOL_SHAPE, dtype=jnp.int8)

        def make_args():
            return (jnp.copy(pool).block_until_ready(), *shared_args)

        def plain(*args):
            return _checkpoint_prefill_five(*args, ckpt=False)

        def with_ckpt(*args):
            return _checkpoint_prefill_five(*args, ckpt=True)

        pool_plain, out_plain = _block_until_ready(plain(*make_args()))
        pool_ckpt, out_ckpt = _block_until_ready(with_ckpt(*make_args()))
        pool_split, _ = _block_until_ready(_first_chunk(*make_args()))

        np.testing.assert_array_equal(np.asarray(out_ckpt), np.asarray(out_plain))
        for mgr in range(CKPT_NUM_MGR):
            if mgr != CKPT_BLOCK:
                np.testing.assert_array_equal(
                    _manager_block(pool_ckpt, mgr),
                    _manager_block(pool_plain, mgr),
                    f"block {mgr}",
                )
        np.testing.assert_array_equal(
            _manager_block(pool_ckpt, CKPT_BLOCK),
            _manager_block(pool_split, CKPT_BLOCK),
        )
