# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The routed experts' GEMM kernel (``megakernel.qwen38.experts``) on the host."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.megakernel.qwen38.experts import (  # noqa: E402
    BLOCK,
    ROWS,
    SCALE_ALIGN,
    expert_products,
    per_row_fp8,
    reference_products,
    requantization_scales,
)

SLAB, COLUMNS, INTER = 32, 64, 512  # FP8 rows as [32, 64] slabs
HIDDEN = SLAB * COLUMNS


def _weights(rng, groups, k, n):
    w = rng.standard_normal((groups, k, n)).astype(np.float32)
    block = rng.uniform(0.5, 2.0, (groups, k // BLOCK, n)).astype(np.float32)
    channel, ratio = requantization_scales(jnp.asarray(block))
    return jnp.asarray(w, jnp.float4_e2m1fn), channel, ratio


def test_expert_products_match_the_unfused_math_per_group():
    rng = np.random.default_rng(0)
    sizes = np.array([5, 0, 1030], np.int32)  # the last group spans two tiles
    groups, total = sizes.size, int(sizes.sum())
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    tokens = 64
    x_all = jnp.asarray(rng.standard_normal((tokens, HIDDEN)), jnp.bfloat16)
    xq, x_scale = per_row_fp8(x_all.astype(jnp.float32))
    token = jnp.asarray(rng.integers(0, tokens, total), jnp.int32)
    w13, c13, r13 = _weights(rng, groups, HIDDEN, 2 * INTER)
    w2, c2, r2 = _weights(rng, groups, INTER, HIDDEN)
    span = total + 3

    padded = jnp.pad(token, (0, span + ROWS - total))
    out = expert_products(
        xq.reshape(tokens, SLAB, COLUMNS)[padded],
        jnp.pad(x_scale[token, 0], (0, span + ROWS + SCALE_ALIGN - total)),
        jnp.asarray(starts),
        jnp.asarray(sizes),
        (w13, c13, r13, w2, c2, r2),
        span=span,
        interpret=True,
    )

    out = np.asarray(out.reshape(span, HIDDEN), np.float32)
    rows = x_all
    for g in range(groups):
        if not sizes[g]:
            continue
        mine = token[starts[g] : starts[g] + sizes[g]]
        want = reference_products(
            rows[mine], w13[g], c13[g], r13[g], w2[g], c2[g], r2[g]
        )
        got = out[starts[g] : starts[g] + sizes[g]]
        np.testing.assert_allclose(
            got, np.asarray(want, np.float32), rtol=1e-2, atol=1e-2
        )
