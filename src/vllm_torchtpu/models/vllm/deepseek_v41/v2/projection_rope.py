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

from vllm_torchtpu.kernels.deepseek_v4.rope import rope
from vllm_torchtpu.kernels.quantized_matmul.projection_rope import chain, is_supported
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

_op: Any = None


def projection_rope(
    x: jax.Array,
    w: jax.Array,
    scale: jax.Array,
    pos: jax.Array,
    cache: jax.Array,
) -> jax.Array:
    # Token dimensions become concrete during JAX lowering. A Torch/Python
    # equality guard would violate the model's dynamic-token export contract.
    if is_supported(x, w, scale, pos, cache):
        return chain(x, w, scale, pos, cache)
    projected = xla_quantized_matmul(x, w, scale).reshape(x.shape[0], 64, 512)
    return rope(projected, pos, cache)


def initialize_projection_rope() -> Any:
    global _op
    if _op is None:
        _op = pallas.jax_op(
            "pallas::fp8_projection_rope",
            projection_rope,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=cast(Any, (P(),) * 5),
        )
        _op.register_fake(
            lambda x, w, scale, pos, cache: x.new_empty((x.shape[0], 64, 512))
        )
    return _op
