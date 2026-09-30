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
"""DeepSeek-V4 CSA compressed-KV cache layout: SparseCore-native NoPE + RoPE.

Compatibility wrapper re-exporting from llm_tpu_core.
"""

from vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.csa_cache_layout import *  # noqa: F403
from vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.csa_cache_layout import (
    as_u8_slabs,
    encode_rope,
    is_word_array,
    rope_words,
    slabs_to_words,
    u8_slab_shape,
    words_to_slabs,
)

__all__ = [
    "as_u8_slabs",
    "encode_rope",
    "is_word_array",
    "rope_words",
    "slabs_to_words",
    "u8_slab_shape",
    "words_to_slabs",
]
