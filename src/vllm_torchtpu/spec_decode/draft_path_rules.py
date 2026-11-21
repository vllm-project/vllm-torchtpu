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
"""Model-provided rules that turn the draft-path programs on by default.

Main keeps greedy top tokens, fast token substitution and the fused MTP draft
loop opt-in (TPU_GREEDY_TOP_TOKENS, VLLM_TPU_FAST_TOKEN_SUBSTITUTION,
VLLM_TPU_FUSED_DRAFT_LOOP). A model package registers a rule at model
registration (models/vllm/__init__.py) for the configurations it has
validated with them; the code that turns those paths on (spec_decode/utils.py)
asks only whether a registered rule holds, so it carries no model-specific
logic. This module imports nothing, so registering a rule at plugin load
costs nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_RULES: list[Callable[[Any], bool]] = []


def register(rule: Callable[[Any], bool]) -> None:
    """Add ``rule(vllm_config) -> bool``; registering the same rule twice is a no-op."""
    if rule not in _RULES:
        _RULES.append(rule)


def holds(vllm_config: Any) -> bool:
    """Whether any registered rule turns the draft-path programs on for this
    configuration."""
    return any(rule(vllm_config) for rule in _RULES)


def matching(vllm_config: Any) -> list[str]:
    """Qualified names of the registered rules that hold for this
    configuration (logged at runner start)."""
    return [f"{r.__module__}.{r.__qualname__}" for r in _RULES if r(vllm_config)]
