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

import jax
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput
from vllm_torchtpu.kernels.qwen4_exp.gdn_input import (
    compact_qkv_input,
    supports_compact_qkv,
)
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
)
from vllm_torchtpu.utils import get_mesh_shape_product

_MODEL_AXIS = "model"


def prepare_gdn_input(
    qkv: jax.Array, request_capacity: int, mesh: jax.sharding.Mesh
) -> jax.Array | Conv1DInput:
    """Prepare chunked Qwen inputs without branching on device metadata.

    Buckets with at most one token per request slot keep the original input.
    Larger eligible buckets cover prefill/mixed work. This is a shape policy,
    not a claim that every row is live: padded decode buckets can also qualify.
    """
    tokens, width = qkv.shape
    local_width = width // get_mesh_shape_product(mesh, _MODEL_AXIS)
    if tokens <= request_capacity or not supports_compact_qkv(
        (tokens, local_width), qkv.dtype
    ):
        return qkv
    values = jax.shard_map(
        compact_qkv_input,
        mesh=mesh,
        in_specs=P(None, _MODEL_AXIS),
        out_specs=P(None, None, _MODEL_AXIS),
        check_vma=False,
    )(qkv)
    return Conv1DInput(values, qkv.dtype)


class Qwen4ExpGatedDeltaNetAttention(VllmGatedDeltaNetAttention):
    """Qwen input preparation with the shared GDN operator and state handling."""

    @staticmethod
    def _prepare_pooled_gdn_input(
        mixed_qkv: jax.Array,
        seq_lens: jax.Array,
        mesh: jax.sharding.Mesh,
    ) -> jax.Array | Conv1DInput:
        return prepare_gdn_input(mixed_qkv, seq_lens.shape[0], mesh)
