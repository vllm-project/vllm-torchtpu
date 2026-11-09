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
"""Head reductions and block maxima shared by the BF16 index score kernels."""

import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def _reduce_heads(logits, weights, queries, stripe):
    query_group = math.gcd(queries, stripe)
    head_group = stripe // query_group
    grouped_rows = queries * head_group
    width = logits.shape[-1]
    accumulated = jnp.zeros((grouped_rows, width), jnp.float32)
    # Consume aligned head/query stripes without padding Q=6 to eight queries.
    # Reassociate the FP32 reduction; identical FP32 bits are not required.
    for chunk in range(weights.shape[0]):
        activated = jnp.maximum(
            logits[chunk * grouped_rows : (chunk + 1) * grouped_rows, :], 0
        )
        accumulated = accumulated + activated * weights[chunk, :, None]
    accumulated = accumulated.reshape(queries // query_group, stripe, width)
    outputs = []
    for group in range(queries // query_group):
        head_parts = [
            accumulated[group, h * query_group : (h + 1) * query_group, :]
            for h in range(head_group)
        ]
        # Preserve the existing cyclic reduction's FP32 addition order, while
        # leaving only the dependency tree for returned query rows live.
        for shift in reversed(range(head_group.bit_length() - 1)):
            head_parts = [
                head_parts[h] + head_parts[(h - (1 << shift)) % head_group]
                for h in range(head_group)
            ]
        outputs.append(head_parts[0])
    return jnp.concatenate(outputs, axis=0)


def _block_max(values, lanes, block_size):
    indices = jnp.broadcast_to(
        (jnp.arange(lanes // block_size) * block_size + block_size - 1)[None, :, None],
        (values.shape[0], lanes // block_size, 1),
    )
    parts = []
    for start in range(0, values.shape[1], lanes):
        window = values[:, start : start + lanes]
        for power in range(block_size.bit_length() - 1):
            shift = 1 << power
            window = jnp.maximum(window, pltpu.roll(window, shift, axis=1))
        # Preserve the row batching dimension even for one query. take_along_axis
        # squeezes it in that case, which Mosaic's gather lowering cannot handle.
        parts.append(
            jax.lax.gather(
                window,
                indices,
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(),
                    collapsed_slice_dims=(1,),
                    start_index_map=(1,),
                    operand_batching_dims=(0,),
                    start_indices_batching_dims=(0,),
                ),
                slice_sizes=(1, 1),
                mode="promise_in_bounds",
            )
        )
    return jnp.concatenate(parts, axis=1)
