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
"""Lifecycle patch registrations for structured speculative decoding."""

from vllm_torchtpu.patch_registry import Patch

STRUCTURED_SPEC_DECODE_PATCHES = (
    Patch(
        "vllm_torchtpu.structured_spec_decoding.patches.scheduler_patch:patch_vllm_grammar_jump_decoding",  # noqa: E501
        ("engine_core",),
    ),
)
