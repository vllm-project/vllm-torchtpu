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
"""Data-parallel request routing by exact API-server in-flight counts.

vLLM's ``DPLBAsyncMPClient`` scores each engine by
``max(client_count * in_flight, waiting + running)``, where ``waiting`` and
``running`` come from the DP coordinator's periodic snapshot. Because a
snapshot continues to count requests that completed after it was published, an
engine that just freed up can tie with busy engines on the stale snapshot and
receive stacked requests via the rotating tie-breaker. When a single API server
is used, its local ``engine_inflight`` counts are exact; enabling
``VLLM_TPU_DP_LB_EXACT_INFLIGHT`` ranks engines by those exact counts while
retaining upstream's rotating tie-break scan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

logger = init_logger(__name__)

_MARKER = "_tpu_exact_inflight_patch"


def check_api_server_count(parallel_config: ParallelConfig) -> None:
    """Reject ``VLLM_TPU_DP_LB_EXACT_INFLIGHT`` when multiple API servers run."""
    if not envs.VLLM_TPU_DP_LB_EXACT_INFLIGHT:
        return
    num_api_servers = getattr(parallel_config, "_api_process_count", 1)
    if isinstance(num_api_servers, int) and num_api_servers > 1:
        raise ValueError(
            "VLLM_TPU_DP_LB_EXACT_INFLIGHT requires --api-server-count=1: each of "
            f"the {num_api_servers} API servers would rank engines by the "
            "requests it routed alone."
        )


def patch_dp_lb_client() -> None:
    """Rank DP engines by exact in-flight counts when enabled."""
    if not envs.VLLM_TPU_DP_LB_EXACT_INFLIGHT:
        return
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    original = DPLBAsyncMPClient.get_core_engine_for_request
    if getattr(original, _MARKER, False):
        return

    def get_core_engine_for_request(self, request):
        assert self.client_count == 1, (
            "VLLM_TPU_DP_LB_EXACT_INFLIGHT requires a single API server"
        )
        snapshot = self.lb_engines
        # Score max(in_flight, waiting + running) reduces to exact in_flight
        # while preserving upstream's rotating tie-breaker scan.
        self.lb_engines = [
            [0, self.engine_inflight[engine], *counts[2:]]
            for engine, counts in zip(self.core_engines, snapshot)
        ]
        try:
            return original(self, request)
        finally:
            self.lb_engines = snapshot

    setattr(get_core_engine_for_request, _MARKER, True)
    DPLBAsyncMPClient.get_core_engine_for_request = get_core_engine_for_request
    logger.info("Applied TPU patch: DP load balancing by exact in-flight counts.")
