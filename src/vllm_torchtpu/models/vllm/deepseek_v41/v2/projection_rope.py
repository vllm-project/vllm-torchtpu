# Copyright 2025 Google LLC
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

"""Torch bridge for fused FP8 query projection and RoPE."""

from typing import Any, cast

import jax
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.attention_preprocessing import qb_rope, rope_gather
from vllm_torchtpu.kernels.deepseek_v4.rope import rope
from vllm_torchtpu.kernels.quantized_matmul.projection_rope import chain, is_supported
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

_op: Any = None
_rows_op: Any = None


def projection_rope(
    x: jax.Array,
    w: jax.Array,
    scale: jax.Array,
    pos: jax.Array,
    cache: jax.Array,
    rows: jax.Array | None = None,
) -> jax.Array:
    """``rows``: the step's shared ``rope_gather.gather_rows(cache, pos)``."""
    # Token dimensions become concrete during JAX lowering. A Torch/Python
    # equality guard would violate the model's dynamic-token export contract.
    # With shared rows, qb_rope beats `chain` at T=288 (23.74 vs 26.26 us,
    # bitwise equal) and the gather is already paid once per step.
    if rows is None and is_supported(x, w, scale, pos, cache):
        return chain(x, w, scale, pos, cache)
    # Every other case: one kernel writes Q in the attention layout,
    # with no HBM round trip of the projection and no separate relayout.
    if rows is None:
        rows = rope_gather.gather_rows(cache, pos)
    if qb_rope.supported(x, w, scale, rows):
        return qb_rope.fused_qb_rope(x, w, scale, rows)
    projected = xla_quantized_matmul(x, w, scale).reshape(x.shape[0], 64, 512)
    return rope(projected, pos, cache)


def _fake(x, *_):
    return x.new_empty((x.shape[0], 64, 512))


def initialize_projection_rope(*, shared_rows: bool = False) -> Any:
    """Op ``(x, w, scale, pos, cache[, rows])``; ``shared_rows`` adds ``rows``."""
    global _op, _rows_op
    if shared_rows:
        if _rows_op is None:
            _rows_op = pallas.jax_op(
                "pallas::fp8_projection_rope_rows",
                projection_rope,
                mesh=get_pallas_op_mesh(),
                input_partition_specs=cast(Any, (P(),) * 6),
            )
            _rows_op.register_fake(_fake)
        return _rows_op
    if _op is None:
        _op = pallas.jax_op(
            "pallas::fp8_projection_rope",
            projection_rope,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=cast(Any, (P(),) * 5),
        )
        _op.register_fake(_fake)
    return _op
