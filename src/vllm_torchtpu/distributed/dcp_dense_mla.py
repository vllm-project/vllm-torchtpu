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
"""What the dense MLA DCP path needs the runner to check.

Prefill tokens attend a context gathered from the DCP group into a static pool
that holds one sequence's whole context (see
`PallasMLAttentionBackendImpl._forward_dense_mla_dcp`), so a step's prefill
contexts must fit it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class DenseMlaDcp:
    dcp_size: int
    page_size: int
    max_model_len: int

    @property
    def pool_pages(self) -> int:
        """Pages per rank of one sequence's whole context (no scratch)."""
        return -(-self.max_model_len // (self.page_size * self.dcp_size))


_ACTIVE: DenseMlaDcp | None = None


def enable(dcp_size: int, page_size: int, max_model_len: int) -> None:
    """Called by every dense MLA layer that runs under DCP; all must agree."""
    global _ACTIVE
    config = DenseMlaDcp(dcp_size, page_size, max_model_len)
    if _ACTIVE is not None and config != _ACTIVE:
        raise ValueError(f"Dense MLA DCP layers disagree: {_ACTIVE} vs {config}.")
    _ACTIVE = config


def active() -> DenseMlaDcp | None:
    return _ACTIVE


def check_prefill_pool(
    num_computed_tokens: Sequence[int], num_scheduled_tokens: Sequence[int]
) -> None:
    """Raise if the step's prefill contexts overflow the gathered pool."""
    if _ACTIVE is None:
        return
    cycle = _ACTIVE.page_size * _ACTIVE.dcp_size
    needed = sum(
        -(-int(ctx) // cycle)
        for ctx, n in zip(num_computed_tokens, num_scheduled_tokens)
        if n > 1
    )
    if needed > _ACTIVE.pool_pages:
        raise RuntimeError(
            f"This step's prefill contexts need {needed} pages per rank, the "
            f"dense MLA DCP pool holds {_ACTIVE.pool_pages}: schedule at most "
            "one partial prefill with context per step."
        )
