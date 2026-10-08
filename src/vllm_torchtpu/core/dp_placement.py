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
"""Release router for data-parallel prompt admission (DP_PLACEMENT=release).

vLLM's DP load balancer places each request on arrival, so the engines'
prefill steps drift apart. The release router holds arrivals in the API
server and releases them as a group, one per engine where it can, so the
engines run their prefill steps together.
"""

from vllm_torchtpu import envs


def install_if_enabled() -> None:
    """Select the release router as vLLM's DP load-balancing client."""
    if envs.DP_PLACEMENT is None:
        return
    from vllm.v1.engine import core_client

    from vllm_torchtpu.core.dp_release_client import ReleaseDPLBAsyncMPClient

    # EngineCoreClient.make_async_mp_client resolves this name per call.
    core_client.DPLBAsyncMPClient = ReleaseDPLBAsyncMPClient
