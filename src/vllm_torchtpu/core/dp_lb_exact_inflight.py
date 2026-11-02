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
"""Data-parallel routing by the API server's exact in-flight counts.

vLLM's ``DPLBAsyncMPClient`` scores each engine by ``max(client_count *
in_flight, waiting + running)``, the latter from the DP coordinator's latest
snapshot. The coordinator republishes at most every 100 ms (later under MoE
DP), each snapshot replaces the client's local counts, and a snapshot still
counts requests that finished after it was taken. When a closed-loop wave of
requests finishes within a few tens of milliseconds, an engine that just freed
up then ties with busy ones and the rotating tie-break can stack the next
requests on one engine. With one API server the client's own in-flight counts
are exact, so ``VLLM_TPU_DP_LB_EXACT_INFLIGHT`` has the upstream scan rank
engines by those alone; upstream's penalty on an engine's waiting queue under
KV-cache pressure goes with the snapshot. vLLM exposes no hook for this score.
"""

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MARKER = "_tpu_exact_inflight_patch"


def check_api_server_count(parallel_config) -> None:
    """Refuse the flag with several API servers: each counts only the requests
    it routed itself."""
    num_api_servers = parallel_config._api_process_count
    if envs.VLLM_TPU_DP_LB_EXACT_INFLIGHT and num_api_servers > 1:
        raise ValueError(
            "VLLM_TPU_DP_LB_EXACT_INFLIGHT needs --api-server-count=1: each of "
            f"the {num_api_servers} API servers would rank engines by the "
            "requests it routed alone."
        )


def patch_dp_lb_client() -> None:
    """With the flag set, rank engines by in-flight counts in
    ``DPLBAsyncMPClient.get_core_engine_for_request``."""
    if not envs.VLLM_TPU_DP_LB_EXACT_INFLIGHT:
        return
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    original = DPLBAsyncMPClient.get_core_engine_for_request
    if getattr(original, _MARKER, False):
        return

    def get_core_engine_for_request(self, request):
        # check_api_server_count refuses this at startup.
        assert self.client_count == 1, "VLLM_TPU_DP_LB_EXACT_INFLIGHT: one API server"
        snapshot = self.lb_engines
        # Upstream's score max(in_flight, waiting + running) becomes the exact
        # in-flight count; ties keep upstream's rotating scan.
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
