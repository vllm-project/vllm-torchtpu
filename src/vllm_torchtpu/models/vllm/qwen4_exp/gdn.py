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

from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput
from vllm_torchtpu.kernels.qwen4_exp.gdn_input import (
    compact_qkv_input,
    supports_compact_qkv,
)
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
)


def prepare_gdn_input(qkv: jax.Array) -> jax.Array | Conv1DInput:
    """Prepare a shard-local projection when its conversion tiles fit."""
    tokens, width = qkv.shape
    if not supports_compact_qkv((tokens, width), qkv.dtype):
        return qkv
    return Conv1DInput(compact_qkv_input(qkv), qkv.dtype)


class Qwen4ExpGatedDeltaNetAttention(VllmGatedDeltaNetAttention):
    """Qwen input preparation with the shared GDN operator and state handling."""

    @staticmethod
    def _prepare_pooled_gdn_input(
        mixed_qkv: jax.Array,
    ) -> jax.Array | Conv1DInput:
        return prepare_gdn_input(mixed_qkv)
