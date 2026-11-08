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
"""LFM3-VL (``Lfm3VlForConditionalGeneration``, e.g. LiquidAI/LFM3-VL-2B) on TPU.

Text-only for now: the checkpoint's language model is LFM3 under
``model.language_model.`` and is served with ``Lfm3ForCausalLM``; the vision
tower and projector weights are not loaded, so image and video inputs are not
accepted (the class does not implement SupportsMultiModal).
"""

from __future__ import annotations

from vllm.model_executor.models.utils import WeightsMapper

from vllm_torchtpu.models.vllm.lfm3 import Lfm3ForCausalLM


class Lfm3VlForConditionalGeneration(Lfm3ForCausalLM):
    # Prefixes apply after LFM3's substring and stacked renames, so vision
    # names dropped here may already carry `.short_conv.` / `.qkv_proj.`.
    hf_to_vllm_mapper = Lfm3ForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
            # TODO: build the vision tower and projector to serve images.
            "model.vision_tower.": None,
            "model.multi_modal_projector.": None,
        },
    )
