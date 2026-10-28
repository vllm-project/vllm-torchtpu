# SPDX-License-Identifier: Apache-2.0
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
    """The original selector, including its score/next-chunk overlap schedule.

    ``bounded_copy`` copies only each row's live prefix of the score buffer
    (see ``sparsecore_topk``); same winners.
    """

    overlaps_scoring: ClassVar[bool] = True
    bounded_copy: bool = False

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
            bounded_copy=self.bounded_copy,
        )
