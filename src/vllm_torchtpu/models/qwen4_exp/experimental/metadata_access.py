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
"""Accessors for per-group Mamba indices in the shared AttentionMetadata.

`mamba_state_of` / `mamba_ckpt_of` read the group's row of the staged walk's
stacked indices when the shared metadata builder provides them
(VLLM_TPU_STAGED_WALK_STACKED), and the plain per-group fields otherwise, so
they work with or without that shared feature.
"""

from typing import Any


def mamba_state_of(md: Any) -> Any:
    """`md.mamba_state_indices`, or its row of the staged walk's stack.

    Call it inside the traced forward: vLLM's compiled forward resolves
    metadata attributes as graph inputs, so the stack enters whole and the row
    slice is an in-graph op.
    """
    stack = getattr(md, "mamba_state_stack", None)
    if stack is None:
        return getattr(md, "mamba_state_indices", None)
    return stack[md.mamba_stack_row]


def mamba_ckpt_of(md: Any) -> Any:
    """`md.mamba_ckpt_indices`, or its row of the stack; see `mamba_state_of`."""
    stack = getattr(md, "mamba_ckpt_stack", None)
    if stack is None:
        return getattr(md, "mamba_ckpt_indices", None)
    return stack[md.mamba_stack_row]
