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
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase


class TpuOwnedKVCache(AttentionLayerBase):
    """A layer that declares its own KV cache spec to the TPU runner.

    The runner collects specs by layer type. Layers outside vLLM's attention
    types subclass this so the runner takes the spec from `get_kv_cache_spec`,
    which lets a model add side caches without editing the runner. Rules:

    - Without `kv_cache_shape`, the spec is normalized like any attention
      spec: the runner may set its dtype to the runner KV cache dtype and pad
      its page, and the attention backend derives the tensor shape. An
      integer-typed `MLAAttentionSpec` needs `kv_cache_shape`.
    - With `kv_cache_shape`, the spec reaches vLLM as is (no normalization)
      and the cache tensor has that shape. Then the spec must be an
      `MLAAttentionSpec`, the unified KV block pool must be off, the leading
      dimension must be `num_blocks`, and the spec's page must hold exactly
      the bytes of `kv_cache_shape(1)`.
    - Layers with equal (normalized) specs share a KV cache group, so they
      must all return the same shape.
    - Models with Mamba layers are not supported.
    """

    def kv_cache_shape(self, num_blocks: int) -> tuple[int, ...] | None:
        """The exact shape of this layer's cache tensor for `num_blocks`
        blocks (leading dimension `num_blocks`), or ``None`` to let the
        attention backend derive it from the normalized spec."""
        del num_blocks
        return None
