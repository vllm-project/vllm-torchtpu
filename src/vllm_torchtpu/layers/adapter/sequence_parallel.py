# Copyright 2025 Google LLC
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
"""Sequence-parallel linear and normalization adapter helpers for TPU.

Provides fused all-gather, reduce-scatter, and row-gated linear wrappers for
TP-sharded token sequences.
"""

from collections.abc import Sequence

from vllm_torchtpu import envs

_ACTIVE = False


def min_rows() -> int:
    return int(envs.TPU_SP_MIN_ROWS or (257 if envs.TPU_MOE_SEQUENCE_PARALLEL else 0))


def enabled() -> bool:
    """Whether sequence-parallel MoE is on at all (main #1463's switch).

    TPU_SP_MIN_ROWS only sets the row threshold below which an SP step runs
    replicated (see min_rows); it never turns SP on by itself.
    """
    return bool(envs.TPU_MOE_SEQUENCE_PARALLEL)


def fp8_gather() -> bool:
    """Whether the attention-side all-gather carries fp8 rows and scales."""
    return int(envs.TPU_SP_FP8_GATHER or 0) > 0


def step_is_sequence_parallel(full_num_tokens: int) -> bool:
    return enabled() and int(full_num_tokens) >= min_rows()


def set_active(active: bool) -> None:
    """Record the form of the step about to run.

    The runner calls it before the compiled model, never a traced forward:
    a global written inside the trace becomes a graph output.
    """
    global _ACTIVE
    _ACTIVE = bool(active)


def is_active() -> bool:
    return _ACTIVE


def rows_are_sliced(is_sequence_parallel: bool) -> bool:
    """Whether a MoE layer's rows are this rank's slice of the step."""
    return is_sequence_parallel and _ACTIVE


def rows_are_replicated(is_sequence_parallel: bool) -> bool:
    """Whether a sequence-parallel MoE layer sees the step's rows replicated
    on every rank (a step below min_rows)."""
    return is_sequence_parallel and not _ACTIVE


def check_buckets(compile_sizes: Sequence[int], tp_size: int) -> None:
    """Raise unless every sequence-parallel bucket splits evenly over TP."""
    threshold = min_rows()
    uneven = [n for n in compile_sizes if n >= threshold and n % tp_size]
    if uneven:
        raise NotImplementedError(
            f"TPU_MOE_SEQUENCE_PARALLEL splits token buckets >= {threshold} "
            f"over tensor_parallel_size={tp_size}; buckets {uneven} do not divide."
        )
