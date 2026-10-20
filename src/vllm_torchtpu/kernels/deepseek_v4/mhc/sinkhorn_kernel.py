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

"""FP32 post gates and Sinkhorn embedded in four-stream mHC kernels."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def single_token_post_comb(
    scaled: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    eps: float,
    multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array]:
    """Register-only post gates and Sinkhorn epilogue for a four-stream token."""
    post = jax.nn.sigmoid(scaled[:, 4:8] * scale[0, 1] + base[:, 4:8]) * multiplier
    # Concatenating four rows avoids Mosaic's unsupported (1, 16) reshape.
    logits = jnp.concatenate(
        [
            scaled[:, 8 + i * 4 : 12 + i * 4] * scale[0, 2]
            + base[:, 8 + i * 4 : 12 + i * 4]
            for i in range(4)
        ],
        axis=0,
    )
    value = jax.nn.softmax(logits, axis=-1) + eps
    value = value / (jnp.sum(value, axis=0, keepdims=True) + eps)

    # Replicate across lanes so the four-value normalizations operate on
    # sublanes/registers. Preserve the original FP32 reduction pairings.
    lanes = pltpu.get_tpu_info().num_lanes
    state = jnp.stack(
        [jnp.broadcast_to(value[i : i + 1, :].T, (4, lanes)) for i in range(4)],
        axis=0,
    )

    def normalize(_, matrix):
        row_sum = (matrix[:, 0, :] + matrix[:, 1, :]) + (
            matrix[:, 2, :] + matrix[:, 3, :]
        )
        matrix = matrix / (row_sum[:, None, :] + eps)
        column_sum = (matrix[0, :, :] + matrix[2, :, :]) + (
            matrix[1, :, :] + matrix[3, :, :]
        )
        return matrix / (column_sum[None, :, :] + eps)

    result = jax.lax.fori_loop(0, repeats - 1, normalize, state)
    return post, jnp.concatenate([result[i, :, :1].T for i in range(4)], axis=0)
