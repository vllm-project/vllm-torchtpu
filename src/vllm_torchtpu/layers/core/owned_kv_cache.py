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
    types subclass this so the spec from `get_kv_cache_spec` is used as is,
    which lets a model add side caches without editing the runner.
    """

    def kv_cache_shape(self, num_blocks: int) -> tuple[int, ...] | None:
        """The exact shape of this layer's cache tensor, or ``None`` to let
        the attention backend derive it from the spec.

        A layer that returns a shape gets its spec to vLLM unnormalized, so
        the spec's page size must match the shape's bytes per block.
        """
        del num_blocks
        return None
