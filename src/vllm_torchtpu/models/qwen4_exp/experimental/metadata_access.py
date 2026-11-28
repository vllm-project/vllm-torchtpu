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
"""Accessors for a Mamba group's indices in the shared AttentionMetadata.

The Qwen3.8 GDN and PLE layers read their group's state and checkpoint slots
through these two functions, so a later change to where the indices live
touches one place.
"""

from typing import Any


def mamba_state_of(md: Any) -> Any:
    """The group's Mamba state slots, ``md.mamba_state_indices``.

    ``md`` is the group's `AttentionMetadata` (or an object with the same
    field); the value is None for groups without Mamba layers.
    """
    return md.mamba_state_indices


def mamba_ckpt_of(md: Any) -> Any:
    """The group's Mamba checkpoint slots, ``md.mamba_ckpt_indices``.

    None when the step writes no checkpoints (no speculative decoding).
    """
    return md.mamba_ckpt_indices
