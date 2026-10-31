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
"""Opt-in Raiden send results carried by vLLM's worker metadata channel."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def raiden_pd_completion_modes(vllm_config: "VllmConfig") -> bool:
    """Validate and return whether opt-in Raiden P send completion is enabled."""
    additional = vllm_config.additional_config
    send = (
        additional.get("raiden_pd_send_completion", False)
        if isinstance(additional, dict)
        else False
    )
    if not isinstance(send, bool):
        raise ValueError("additional_config.raiden_pd_send_completion must be a bool")
    if not send:
        return False

    kv = vllm_config.kv_transfer_config
    stage3 = str(envs.TPU_KV_RESHARD_TRANSPORT).strip().lower() == "raiden"
    parallel = vllm_config.parallel_config
    if (
        not stage3
        or kv is None
        or kv.kv_role != "kv_producer"
        or parallel.tensor_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
    ):
        raise ValueError(
            "raiden_pd_send_completion requires a Raiden Stage-3 "
            "kv_producer with TP1/DP1/PP1"
        )

    connectors = (
        kv.kv_connector_extra_config["connectors"]
        if kv.kv_connector == "TPUMultiConnector"
        else [{"kv_connector": kv.kv_connector}]
    )
    raiden_count = sum(c["kv_connector"] == "TPURaidenConnector" for c in connectors)
    if raiden_count != 1:
        raise ValueError(
            "raiden_pd_send_completion requires exactly one TPURaidenConnector"
        )
    return True


class RaidenSendOutcome(StrEnum):
    SUCCESS = "success"
    SEND_FAILURE = "send_failure"
    UNCLAIMED_TIMEOUT = "unclaimed_timeout"


_OUTCOME_PRIORITY = {
    RaidenSendOutcome.SUCCESS: 0,
    RaidenSendOutcome.UNCLAIMED_TIMEOUT: 1,
    RaidenSendOutcome.SEND_FAILURE: 2,
}


def merge_raiden_send_outcomes(
    current: RaidenSendOutcome | None,
    candidate: RaidenSendOutcome,
) -> RaidenSendOutcome:
    if current is None or _OUTCOME_PRIORITY[candidate] > _OUTCOME_PRIORITY[current]:
        return candidate
    return current


@dataclass
class RaidenSendResults(KVConnectorWorkerMetadata):
    # Preserve the terminal reason across worker/rank aggregation.
    # finished_sending remains the independent cleanup barrier.
    outcomes: dict[str, RaidenSendOutcome] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "RaidenSendResults":
        assert isinstance(other, RaidenSendResults), (
            "Raiden send results must aggregate with the same connector type"
        )
        for req_id, outcome in other.outcomes.items():
            self.outcomes[req_id] = merge_raiden_send_outcomes(
                self.outcomes.get(req_id), outcome
            )
        return self
