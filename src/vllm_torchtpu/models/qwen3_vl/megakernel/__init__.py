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
"""Qwen3-VL text decoder megakernel (the `qwen3_vl` shorthand).

The megakernel serves the decoder's single-request decode steps and, under
speculative decoding, their verify steps of `1 + K` tokens; prefill, vision
and every step with prompt tokens left stay on the stock model. Serve
it with `VLLM_TPU_MEGAKERNEL=1` (by the checkpoint's arch) or `qwen3_vl`, and
the stock paths this kernel's KV pages match: `VLLM_KV_CACHE_LAYOUT=HND`,
`--attention-backend CUSTOM`, at any page size of 128-token multiples.
Importing this module loads no torch / jax / vLLM: `create` is imported lazily.
"""

from __future__ import annotations

from vllm_torchtpu.megakernel.core.runtime.contract import TpuMonoSpec

SPEC = TpuMonoSpec(
    name="qwen3_vl",
    version="1",
    create=".adapter:create",
    chips=("TPU7x",),
    tp_sizes=(8,),
    features=frozenset({"mrope", "multimodal", "prefix_caching", "spec_decode"}),
    requires={"kv_cache_dtype": "auto"},
    # Cosmos3 is a Qwen3-VL whose decoder is the Qwen3-VL text model.
    architectures=("Cosmos3ForConditionalGeneration",),
    model_types=("cosmos3_omni",),
    top_level_options=frozenset({"profile", "buckets"}),
    numerics="exact",
    decline_codes=(
        "invalid_geometry",
        "geometry_mismatch",
        "backbone_not_found",
        "layer_count_mismatch",
        "kv_cache_layout",
    ),
)
