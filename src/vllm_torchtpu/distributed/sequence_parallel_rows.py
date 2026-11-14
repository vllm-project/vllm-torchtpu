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
"""Row gate for sequence-parallel MoE (TPU_MOE_SEQUENCE_PARALLEL=1).

With ``TPU_SP_MIN_ROWS=N``, a backbone step of fewer than N tokens keeps every
token on every TP rank: the attention output is all-reduced and the MoE combine
all-reduces the routed experts' partial sums. Two collectives per layer instead
of four suit the small, latency-bound decode steps. Steps of N tokens or more
keep the sequence-parallel form.

The model forward decides from ``positions.shape[-1]``, which leaves a shape
guard, so every token bucket on one side of N shares a trace and the other side
gets its own (``compilation.shape_variants``). The MoE communicator cannot see
the positions, so the runner records the step's token count before calling the
backbone and the communicator applies the same rule to it at trace time.
"""

import contextlib
from collections.abc import Iterator

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.compilation import shape_variants

_backbone_step_num_tokens: int | None = None


def replicated_rows_step(num_tokens: int | torch.SymInt | None) -> bool:
    """Whether a backbone step of ``num_tokens`` keeps every token on every rank.

    The token-count comparison comes last, so with TPU_SP_MIN_ROWS unset no
    shape guard is added and the trace is the plain sequence-parallel one. A
    trace outside the backbone warmup keeps the sequence-parallel form: only
    the warmup gives each side of the threshold its own trace.
    """
    min_rows = envs.TPU_SP_MIN_ROWS
    if min_rows <= 0 or num_tokens is None:
        return False
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return num_tokens < min_rows


@contextlib.contextmanager
def backbone_step(num_tokens: int) -> Iterator[None]:
    """Scope of one backbone forward, outside the compiled region.

    The token count is cleared afterwards so other forwards, such as the MTP
    drafter, which always runs sequence parallel, see no replicated step.
    """
    global _backbone_step_num_tokens
    _backbone_step_num_tokens = int(num_tokens)
    try:
        yield
    finally:
        _backbone_step_num_tokens = None


def replicated_rows_moe_step() -> bool:
    """:func:`replicated_rows_step` for the MoE of the running backbone step."""
    return replicated_rows_step(_backbone_step_num_tokens)
