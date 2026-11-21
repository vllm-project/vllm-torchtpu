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
"""Row-threshold gating for MoE sequence parallelism under tensor parallelism.

When ``TPU_SP_MIN_ROWS > 0``, steps whose global token count reaches the
threshold run sequence-parallel across TP ranks (attn/GDN inputs gathered,
outputs reduce-scattered, and MoE dispatched via ``fused_moe_ep_v3``). Smaller
steps keep tokens replicated on every rank and reduce via ``all_reduce``
(``small_all_reduce_op`` for small batches), avoiding the extra collectives.
When ``TPU_SP_MIN_ROWS <= 0`` (the default), every step is sequence-parallel.
"""

from vllm_torchtpu import envs

_ACTIVE = False


def min_rows() -> int:
    return int(envs.TPU_SP_MIN_ROWS or 0)


def enabled() -> bool:
    return min_rows() > 0


def step_is_sequence_parallel(full_num_tokens: int) -> bool:
    min_r = min_rows()
    return min_r <= 0 or int(full_num_tokens) >= min_r


def set_active(active: bool) -> bool:
    """Record whether the step about to run uses sequence parallelism."""
    global _ACTIVE
    prev = _ACTIVE
    _ACTIVE = bool(active)
    return prev


def is_active() -> bool:
    return _ACTIVE
