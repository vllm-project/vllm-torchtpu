# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Copied from vLLM v0.33.0 vllm/v1/sample/logits_processor/interface.py,
# unchanged apart from imports; see vllm_torchtpu/runner/vllm_v1/__init__.py.
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = auto()
    # Two-way i1<->i2 req swap within batch
    SWAP = auto()


# Batch indices of any removed requests.
RemovedRequest = int

# (index, params, prompt_tok_ids, output_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]


@dataclass(frozen=True)
class BatchUpdate:
    """Persistent batch state change info for logitsprocs."""

    batch_size: int  # Current num reqs in batch

    # Metadata for requests added to, removed from, and moved
    # within the persistent batch.
    #
    # Key assumption: the `output_tok_ids` list (which is an element of each
    # tuple in `added`) is a reference to the request's running output tokens
    # list; via this reference, the logits processors always see the latest
    # list of generated output tokens.
    #
    # NOTE:
    # * Added or moved requests may replace existing requests with the same
    #   index.
    # * Operations should be processed in the following order:
    #   - removed, added, moved
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


class LogitsProcessor(ABC):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate sampling params for this logits processor.

        Raise ``VLLMValidationError`` (preferred) / ``ValueError`` (backward compatible)
        for invalid params. Bare ``ValueError`` is converted to ``VLLMValidationError``
        at the engine boundary so online serving returns HTTP 400.
        """
        return None

    @abstractmethod
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be
        modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """True if logits processor has no impact on the
        argmax computation in greedy sampling.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        batch_update: "BatchUpdate | None",
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.

        Args:
            batch_update: Non-None iff there have been changes
                to the batch makeup.

        """
        raise NotImplementedError
