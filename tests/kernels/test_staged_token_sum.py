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
"""The owner tail's per-token sum (``staged_token_sum``): bitwise
``owner_token_sum`` on routings the end-to-end tests rarely draw."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import owner_token_sum, staged_token_sum
from vllm_torchtpu.layers.core.ragged_a2a_moe import _owner_routes

TOPK = 10
NUM_TOKENS = 8
ROW_SHAPE = (16, 512)


def _reference(tp_ref, dp_ref, sent_ref, weight_ref, t, rows, i, *, topk, tp_size):
    return owner_token_sum(
        tp_ref,
        dp_ref,
        sent_ref,
        weight_ref,
        t,
        lambda j: rows[j, i],
        lambda r: rows[topk + r, i],
        topk=topk,
        tp_size=tp_size,
        dtype=rows.dtype,
        row_shape=rows.shape[2:],
    )


def _sums(fn, tp_size):
    def kernel(tp_ref, dp_ref, sent_ref, weight_ref, rows_ref, out_ref, stage_ref):
        stage_ref[...] = rows_ref[...]
        for i in range(NUM_TOKENS):
            out = fn(
                tp_ref,
                dp_ref,
                sent_ref,
                weight_ref,
                i,
                stage_ref,
                i,
                topk=TOPK,
                tp_size=tp_size,
            )
            out_ref[i] = out.astype(out_ref.dtype)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((NUM_TOKENS, *ROW_SHAPE), jnp.bfloat16),
            in_specs=[smem] * 4 + [vmem],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((TOPK + tp_size + 1, NUM_TOKENS, *ROW_SHAPE), jnp.bfloat16)
            ],
        )
    )


@pytest.mark.parametrize(
    "dp_size,tp_size,skip_padded",
    [(2, 4, False), (16, 4, True), (1, 4, False), (4, 2, True), (8, 1, False)],
)
def test_staged_token_sum_keeps_every_bit(dp_size, tp_size, skip_padded):
    """A padded token, one with all routes on one TP rank, one with a single
    route, padded routes; zero-weight routes; unsent routes' rows NaN; route
    and shared rows with +0 and -0 elements."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    rng = np.random.default_rng(dp_size * 10 + tp_size)
    num_local = 12
    num_experts = dp_size * tp_size * num_local
    ids = np.stack(
        [rng.choice(num_experts, TOPK, replace=False) for _ in range(NUM_TOKENS)]
    )
    rank0 = [e for e in range(num_experts) if (e // num_local) % tp_size == 0]
    ids[0] = -1
    ids[1] = rng.choice(rank0, TOPK, replace=False)
    ids[2, 1:] = -1
    ids[3, -3:] = -1
    weights = rng.random((NUM_TOKENS, TOPK)).astype(np.float32)
    weights[4, :4] = 0
    tp_s, dp_s, sent_s, weight_s = _owner_routes(
        jnp.asarray(ids, jnp.int32),
        jnp.asarray(weights, jnp.bfloat16),
        skip_padded_tokens=skip_padded,
        dtype=jnp.bfloat16,
        dp_size=dp_size,
        tp_size=tp_size,
        local_num_experts=num_local,
    )
    sent_s = sent_s.astype(jnp.int32)
    rows = rng.standard_normal((TOPK + tp_size + 1, NUM_TOKENS, *ROW_SHAPE)) * 0.3
    rows = rows.astype(jnp.bfloat16)
    rows[-1] = np.nan  # the free row
    rows[:, :, 0, :8] = 0.0
    rows[:, :, 1, :8] = -0.0
    rows[:TOPK][np.asarray(sent_s) == 0] = np.nan
    tables = (tp_s, dp_s, sent_s, weight_s, jnp.asarray(rows))
    want = np.asarray(_sums(_reference, tp_size)(*tables))
    got = np.asarray(_sums(staged_token_sum, tp_size)(*tables))
    assert not np.isnan(want.astype(np.float32)).any()
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))
