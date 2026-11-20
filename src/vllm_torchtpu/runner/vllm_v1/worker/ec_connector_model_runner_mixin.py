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

# Copied from vLLM v0.33.0 vllm/v1/worker/ec_connector_model_runner_mixin.py,
# unchanged apart from imports; see vllm_torchtpu/runner/vllm_v1/__init__.py.
"""Define EC connector functionality mixin for model runners."""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.v1.outputs import ECConnectorOutput

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a EC connector functionality mixin for ModelRunner (GPU, TPU)
class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ):
        if not has_ec_transfer():
            logger.debug("Not have ec transfer please check")
            return
        connector = get_ec_transfer()
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)

    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:
        return (
            ECConnectorModelRunnerMixin._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()
            else nullcontext()
        )

    # This context manager must be used within an active forward context.
    # It encapsulates the entire EC connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        output = ECConnectorOutput()

        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        if ec_connector.is_producer:
            # Pass the cache explicitly: a producer needs it to publish an item
            # that is already cached from an earlier step, which this step will
            # not re-encode -- `save_caches` never fires for those.
            ec_connector.start_save_caches(encoder_cache=encoder_cache, **kwargs)

        # Load caches for consumer or both roles
        if ec_connector.is_consumer:
            ec_connector.start_load_caches(encoder_cache, **kwargs)

        try:
            yield output
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.ec_connector_worker_meta = ec_connector.build_connector_worker_meta()

            ec_connector.clear_connector_metadata()
