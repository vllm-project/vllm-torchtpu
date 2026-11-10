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
from typing import Any

from vllm_torchtpu import envs

_ACTIVE = False

# Smallest step, in rows, that runs sequence-parallel. Below it the step keeps
# its rows replicated over TP: slicing a few rows saves less in the MoE than the
# extra all-gather and reduce-scatter cost. A layout measured to cross over at a
# different row count gets an entry in _MIN_ROWS_BY_LAYOUT, keyed on
# (data_parallel_size, tensor_parallel_size).
#
# 0 until a model reads each step's form (rows_are_sliced): with it, every
# sequence-parallel step stays sliced and every compile size follows main's
# alignment rule, as on main. The change that makes the model follow the
# step's form raises the threshold.
DEFAULT_MIN_ROWS = 0
_MIN_ROWS_BY_LAYOUT: dict[tuple[int, int], int] = {}


def parallel_layout(parallel_config: Any = None) -> tuple[int, int] | None:
    """(data_parallel_size, tensor_parallel_size) of ``parallel_config``, or of
    this process's initialized model-parallel groups; None when neither is
    available (for example, before the workers set up their groups)."""
    if parallel_config is not None:
        return (
            int(getattr(parallel_config, "data_parallel_size", 1)),
            int(getattr(parallel_config, "tensor_parallel_size", 1)),
        )
    from vllm.distributed import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        return None
    return (
        int(parallel_state.get_dp_group().world_size),
        int(parallel_state.get_tp_group().world_size),
    )


def min_rows(parallel_config: Any = None) -> int:
    """The row threshold for the layout of ``parallel_config``, or of this
    process's model-parallel groups when it is None."""
    layout = parallel_layout(parallel_config)
    if layout is None:
        return DEFAULT_MIN_ROWS
    return _MIN_ROWS_BY_LAYOUT.get(layout, DEFAULT_MIN_ROWS)


def enabled() -> bool:
    """Whether sequence-parallel MoE is on at all (main #1463's switch).

    The row threshold (see min_rows) only decides which steps of an enabled
    model run sliced; it never turns SP on by itself.
    """
    return bool(envs.TPU_MOE_SEQUENCE_PARALLEL)


def fp8_gather() -> bool:
    """Whether the attention-side all-gather carries fp8 rows and scales.

    Always true. It acts only on a sequence-parallel step, whose gathered rows
    feed FP8 projections that quantize their input per row: quantizing each row
    before the gather instead of after it gives the same inputs and moves about
    half the bytes of a bf16 gather.
    """
    return True


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
