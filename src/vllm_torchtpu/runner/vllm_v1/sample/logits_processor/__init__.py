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

# Copied from vLLM v0.33.0 vllm/v1/sample/logits_processor/__init__.py. Apart
# from imports, _load_custom_logitsprocs loads no custom logits processors and
# the plugin and FQCN loaders it called are removed; see
# vllm_torchtpu/runner/vllm_v1/__init__.py.
import inspect
import itertools
from abc import abstractmethod
from collections.abc import Sequence
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import torch
from vllm.exceptions import VLLMValidationError
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.sampling_params import SamplingParams

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.vllm_v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
    process_dict_updates,
)
from vllm_torchtpu.runner.vllm_v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from vllm_torchtpu.runner.vllm_v1.sample.logits_processor.state import (
    BatchUpdateBuilder,
    LogitsProcessors,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Error message when the user tries to initialize vLLM with a pooling model
# and custom logitsproces
STR_POOLING_REJECTS_LOGITSPROCS = (
    "Pooling models do not support custom logits processors."
)

# Error message when the user tries to initialize vLLM with a speculative
# decoding enabled and custom logitsproces
STR_SPEC_DEC_REJECTS_LOGITSPROCS = (
    "Custom logits processors are not supported when speculative decoding is enabled."
)

BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
]


def _load_custom_logitsprocs(
    logits_processors: Sequence[str | type[LogitsProcessor]] | None,
) -> list[type[LogitsProcessor]]:
    """Load all custom logits processors.

    TPU edit: loads none, neither installed vllm.logits_processors plugins nor
    the ones passed in, because TPUModelRunner applies no custom logits
    processors; platforms/tpu_platform.py _validate_tpu_runner_features rejects
    configured ones at startup. Upstream skips loading only when
    current_platform.is_tpu(), which is False for this out-of-tree platform.

    Args:
      logits_processors: potentially mixed list of logitproc types and
                         logitproc type fully-qualified names (FQCNs)

    Returns:
      An empty list

    """
    return []


def build_logitsprocs(
    vllm_config: "VllmConfig",
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (),
) -> LogitsProcessors:
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug(
            "Skipping logits processor loading because pooling models"
            " do not support logits processors."
        )
        return LogitsProcessors()

    # Check if speculative decoding is enabled.
    if vllm_config.speculative_config:
        if custom_logitsprocs:
            raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
        logger.warning(
            "min_p and logit_bias parameters won't work with speculative decoding."
        )
        return LogitsProcessors(
            [MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]
        )

    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory)
        for ctor in itertools.chain(
            BUILTIN_LOGITS_PROCESSORS, custom_logitsprocs_classes
        )
    )


cached_load_custom_logitsprocs = lru_cache(_load_custom_logitsprocs)


def validate_logits_processors_parameters(
    logits_processors: Sequence[str | type[LogitsProcessor]] | None,
    sampling_params: SamplingParams,
):
    logits_processors = (
        tuple(logits_processors) if logits_processors is not None else None
    )
    for logits_procs in cached_load_custom_logitsprocs(logits_processors):
        try:
            logits_procs.validate_params(sampling_params)
        except ValueError as e:
            # Legacy custom logitsprocs may still raise ValueError from
            # validate_params; convert for backward compatibility.
            raise VLLMValidationError(str(e)) from e


class AdapterLogitsProcessor(LogitsProcessor):
    """Wrapper for per-request logits processors.

    To wrap a specific per-request logits processor,
    * Subclass `AdapterLogitsProcessor`
    * Implement `self.is_argmax_invariant()` base-class method
    * Implement `self.new_req_logits_processor(params)`

    `self.__init__(vllm_config, device, is_pin_memory)` does not need to be
    overridden in general. However, to implement custom constructor behavior -
    especially any logic which operates on or stores `vllm_config`, `device`,
    or `is_pin_memory` - `self.__init__(vllm_config, device, is_pin_memory)`
    must be overridden and the override must call
    `super().__init__(vllm_config, device, is_pin_memory)`
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        """Subclass must invoke
        `super().__init__(vllm_config, device, is_pin_memory)`.

        Subclass constructor may find it useful to utilize the `vllm_config`,
        `device` and `is_pin_memory` argument. However regardless of whether
        these arguments are used, the vLLM logits processor interface requires
        all three arguments to be present.
        """
        # Map req index -> logits processor state
        #
        # State representation is a partial[Tensor] comprising a request-level
        # logits processor with the output token ids argument and (if required)
        # the prompt token ids argument pre-populated
        #
        # Note that the partial carries a *reference* to output token ids, and
        # will thus always operate on the list as it is currently, not as it
        # was when the partial was created.
        self.req_info: dict[int, partial[torch.Tensor]] = {}

    @abstractmethod
    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        """Consume request info; return a per-request logits processor.

        Return None if logits processor does not need to be applied to request

        Args:
          params: request sampling params

        Returns:
          None if logits processor should not be applied to request; otherwise
          returns a `RequestLogitsProcessor` instance

        """
        raise NotImplementedError

    def _new_state(
        self,
        params: SamplingParams,
        prompt_ids: list[int] | None,
        output_ids: list[int],
    ) -> partial[torch.Tensor] | None:
        """Return state representation for new request.

        Returns None if logits processor is not applicable to request

        Args:
          params: request sampling params
          prompt_ids: request prompt token ids
          output_ids: decoded tokens so far for this request

        Returns:
          logits processor partial[Tensor] or None

        """
        if req_lp := self.new_req_logits_processor(params):
            if len(inspect.signature(req_lp).parameters) == 3:
                if prompt_ids is None:
                    raise ValueError(
                        "Prompt token ids are required for this "
                        "logits processor but were not provided."
                    )
                args = [prompt_ids, output_ids]
            else:
                args = [output_ids]
            return partial(req_lp, *args)
        return None

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(
            self.req_info,
            batch_update,
            self._new_state,
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.req_info:
            # Apply per-request logits processors to corresponding rows of
            # logits tensor
            for req_idx, req_lp in self.req_info.items():
                req_logits = logits[req_idx]
                new_logits = req_lp(req_logits)
                if new_logits is not req_logits:
                    # Modify logits tensor row in-place if necessary
                    logits[req_idx] = new_logits
        return logits


__all__ = [
    "LogitsProcessor",
    "LogitBiasLogitsProcessor",
    "MinPLogitsProcessor",
    "MinTokensLogitsProcessor",
    "BatchUpdate",
    "BatchUpdateBuilder",
    "MoveDirectionality",
    "LogitsProcessors",
    "build_logitsprocs",
    "STR_POOLING_REJECTS_LOGITSPROCS",
    "AdapterLogitsProcessor",
]
