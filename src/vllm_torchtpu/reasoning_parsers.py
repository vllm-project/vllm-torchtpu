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
"""Reasoning parser registrations, imported only when a server asks for one.

``minimax_m3``: ``minimax_m3_reasoning_parser`` (the same parser, its marker
searches in C).
"""

from vllm.reasoning import ReasoningParserManager


def register_reasoning_parsers() -> None:
    ReasoningParserManager.register_lazy_module(
        "minimax_m3",
        "vllm_torchtpu.minimax_m3_reasoning_parser",
        "MiniMaxM3ReasoningParser",
    )
