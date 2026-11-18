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
"""The Qwen3.5 GDN copy's state DMA waits on a TPU.

A decode tile of T sequences with a verify window of W tokens writes W state
checkpoints per sequence out of a double-buffered window, and the next tile
reuses that buffer once its writes have been waited for. StateInBufferedRef
and StateOutBufferedRef wait for each copy they issued; a single aggregate
wait over T * W checkpoints sliced past the window axis for W > 1 at T >= 2
and could leave checkpoints unlanded when the buffer was reused, so the
written states differed from run to run.

CPU interpret mode does not show this: with the aggregate waits put back, it
gives the same outputs and states as with the per-copy waits under every
DMA execution mode and with race detection on. So this runs on a TPU: W = 4
windows over many decode tiles (T = 2 and T = 4), repeated, must give
bit-identical outputs and states every time.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen3_5.kernels.gdn import wrapper

N_KQ, N_V, D, KERNEL_SIZE = 2, 8, 128, 4
DIM = 2 * N_KQ * D + N_V * D
WINDOW, NUM_SEQS, RUNS = 4, 64, 5


def _verify_batch(seed=7):
    rngs = iter(jax.random.split(jax.random.key(seed), 9))
    num_tokens = NUM_SEQS * WINDOW
    num_blocks = (NUM_SEQS + 1) * WINDOW
    return dict(
        qkv=jax.random.normal(next(rngs), (num_tokens, DIM), jnp.bfloat16),
        b=jax.random.normal(next(rngs), (num_tokens, N_V), jnp.bfloat16),
        a=jax.random.normal(next(rngs), (num_tokens, N_V), jnp.bfloat16),
        states=(
            jax.random.normal(next(rngs), (num_blocks, KERNEL_SIZE - 1, DIM)),
            jax.random.normal(next(rngs), (num_blocks, N_V, D, D)),
        ),
        conv_weight=jax.random.normal(next(rngs), (DIM, 1, KERNEL_SIZE), jnp.bfloat16),
        conv_bias=jax.random.normal(next(rngs), (DIM,), jnp.bfloat16),
        a_log=jax.random.normal(next(rngs), (N_V,), jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (N_V,), jnp.float32),
        query_start_loc=jnp.arange(NUM_SEQS + 1, dtype=jnp.int32) * WINDOW,
        state_indices=jnp.arange(1, NUM_SEQS + 1, dtype=jnp.int32) * WINDOW,
        distribution=jnp.array([NUM_SEQS] * 3, jnp.int32),
        seq_lens=jnp.full((NUM_SEQS,), 20, jnp.int32),
        read_offsets=jnp.zeros((NUM_SEQS,), jnp.int32),
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("decode_tile_size", [2, 4])
@pytest.mark.parametrize("bf16_qkv_mode", [0, 2])
def test_verify_window_checkpoints_are_identical_across_runs_on_tpu(
    decode_tile_size, bf16_qkv_mode
):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the state DMA waits only matter on a TPU")
    inputs = _verify_batch()
    results = []
    for _ in range(RUNS):
        # fused_conv1d_gdn donates `states`, so each run gets its own copy.
        call_inputs = dict(inputs, states=jax.tree.map(jnp.copy, inputs["states"]))
        results.append(
            jax.block_until_ready(
                wrapper.fused_conv1d_gdn(
                    **call_inputs,
                    n_kq=N_KQ,
                    n_v=N_V,
                    d_k=D,
                    d_v=D,
                    kernel_size=KERNEL_SIZE,
                    num_spec_tokens=WINDOW - 1,
                    decode_tile_size=decode_tile_size,
                    batched_only=True,
                    bf16_qkv_mode=bf16_qkv_mode,
                )
            )
        )
    first = [np.asarray(x) for x in jax.tree.leaves(results[0])]
    for run in results[1:]:
        for got, want in zip(jax.tree.leaves(run), first, strict=True):
            np.testing.assert_array_equal(np.asarray(got), want)
