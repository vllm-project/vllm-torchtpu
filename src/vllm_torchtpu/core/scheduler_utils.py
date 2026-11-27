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
"""Helpers shared by the TPU schedulers and hybrid-state retention."""

from typing import NamedTuple

from vllm.config import KVTransferConfig
from vllm.v1.request import Request, RequestStatus

_WAITING_STATUSES = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


def is_waiting_status(request: Request) -> bool:
    """Whether ``request`` sits in the waiting queue (WAITING or PREEMPTED)."""
    return request.status in _WAITING_STATUSES


def format_counts(counts: dict[str, int]) -> str:
    """Formats a reason -> count map as ``a:1,b:2`` for a telemetry field."""
    return ",".join(f"{k}:{v}" for k, v in counts.items()) or "-"


class KvRoles(NamedTuple):
    """An engine's prefill/decode disaggregation roles."""

    producer: bool
    consumer: bool


def kv_roles(kv_transfer_config: KVTransferConfig | None) -> KvRoles:
    """The engine's roles, from its ``kv_transfer_config`` (None without KV
    transfer).

    ``producer``: ``kv_role == "kv_producer"``, a prefill leg. ``consumer``:
    vLLM's ``is_kv_consumer and not is_kv_producer``, a decode leg. vLLM's
    role properties count ``kv_both`` (an offloading connector on an
    aggregated engine) as both a producer and a consumer; here it has
    neither role.
    """
    if kv_transfer_config is None:
        return KvRoles(producer=False, consumer=False)
    return KvRoles(
        producer=kv_transfer_config.kv_role == "kv_producer",
        consumer=bool(
            kv_transfer_config.is_kv_consumer and not kv_transfer_config.is_kv_producer
        ),
    )
