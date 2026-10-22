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
"""Static selection backends for StreamIndex's score-buffer contract."""

from dataclasses import dataclass
from typing import ClassVar, Protocol

import jax

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk


class TopKBackend(Protocol):
    """Select exact winners from i32 buffers containing IEEE float32 score bits.

    Only columns below each row length are eligible; -inf marks invalid scores.
    Return i32 column IDs, with valid winners packed before a -1 suffix. Winner
    order is unspecified; threshold ties may choose any eligible tied IDs. Optional
    scores use the same i32 float32-bit representation, with -inf padding.
    Backends own any input reinterpretation, hardware eligibility and tiling.
    Implementations must be hashable because they are static JIT arguments.
    """

    overlaps_scoring: ClassVar[bool]

    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]: ...


@dataclass(frozen=True)
class SparseCoreTopK:
    """The original selector, including its score/next-chunk overlap schedule."""

    overlaps_scoring: ClassVar[bool] = True

    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        return sparsecore_topk(
            score_bits,
            k,
            row_lengths=row_lengths,
            write_empty_rows=True,
            scheduling_group_id=scheduling_group_id,
            return_scores=return_scores,
        )
