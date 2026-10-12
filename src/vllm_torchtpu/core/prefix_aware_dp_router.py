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
"""Prefix-affine data-parallel request routing (companion to PrefixAwareScheduler).

vLLM's ``DPLBAsyncMPClient`` load-balances each new request onto the
least-loaded DP engine, which scatters the turns of one multi-turn agentic
session across engines and defeats per-engine HBM prefix caching. This module
routes a request to the engine selected by a hash of its first
``PREFIX_ROUTING_TOKENS`` prompt tokens, so requests sharing a prompt prefix
land on the same engine (where ``PrefixAwareScheduler`` then admits the warmest
ones first). The affine engine is only used while it is not overloaded:

  * this client's in-flight count on it < ``PREFIX_ROUTING_MAX_QUEUE``,
  * the coordinator-reported waiting count < ``PREFIX_ROUTING_MAX_QUEUE``,
  * its KV-cache usage < ``PREFIX_ROUTING_MAX_KV``.

Otherwise, and for requests with an explicit ``data_parallel_rank`` or no
prompt tokens, the stock vLLM policy is used unchanged.

This is the in-tree form of the "Prefix-Aware DP Routing Hook" previously
appended (base64) to ``vllm/v1/engine/core_client.py`` by the DeepSeek-V4
AgentX CDK recipe; routing decisions are identical.

Enable in the API-server process with::

    from vllm_torchtpu.core.prefix_aware_dp_router import (
        install_prefix_aware_dp_routing,
    )

    install_prefix_aware_dp_routing()
"""

from __future__ import annotations

import array
import zlib
from collections.abc import Sequence
from typing import Any

import vllm_torchtpu.envs as envs

_PATCH_MARKER = "_tpu_prefix_aware_dp_routing_patch"


def prefix_route_index(
    prompt_token_ids: Sequence[int], num_engines: int, prefix_len: int
) -> int:
    """Returns the prefix-affine engine index for ``prompt_token_ids``.

    The hash is CRC32 over the first ``min(len, prefix_len)`` token ids encoded
    as native unsigned 32-bit ints, modulo ``num_engines``.
    """
    if num_engines <= 0:
        raise ValueError(f"num_engines must be positive, got {num_engines}")
    if prefix_len <= 0:
        raise ValueError(f"prefix_len must be positive, got {prefix_len}")
    k = min(len(prompt_token_ids), prefix_len)
    prefix_hash = zlib.crc32(array.array("I", prompt_token_ids[:k]).tobytes())
    return prefix_hash % num_engines


def _env_enabled() -> bool:
    return bool(envs.ENABLE_PREFIX_ROUTING)


def install_prefix_aware_dp_routing(client_cls: Any = None) -> bool:
    """Patches ``client_cls.get_core_engine_for_request`` in place.

    Args:
      client_cls: class to patch; defaults to vLLM's ``DPLBAsyncMPClient``.

    Returns:
      True if the routing is active on ``client_cls`` after the call (newly
      installed or already installed), False if disabled via
      ``ENABLE_PREFIX_ROUTING``.
    """
    if not _env_enabled():
        return False
    if client_cls is None:
        from vllm.v1.engine.core_client import DPLBAsyncMPClient

        client_cls = DPLBAsyncMPClient
    if client_cls.__dict__.get(_PATCH_MARKER, False):
        return True

    prefix_len = int(envs.PREFIX_ROUTING_TOKENS)
    max_queue = int(envs.PREFIX_ROUTING_MAX_QUEUE)
    max_kv = float(envs.PREFIX_ROUTING_MAX_KV)
    if prefix_len <= 0:
        raise ValueError(f"PREFIX_ROUTING_TOKENS must be positive, got {prefix_len}")
    orig_get = client_cls.get_core_engine_for_request

    def prefix_aware_get_core_engine(self, request):
        if getattr(request, "data_parallel_rank", None) is None:
            tokens = getattr(request, "prompt_token_ids", None)
            if tokens:
                candidate_idx = prefix_route_index(
                    tokens, len(self.core_engines), prefix_len
                )
                waiting, _running, kv_usage = self.lb_engines[candidate_idx]
                chosen = self.core_engines[candidate_idx]
                inflight = self.engine_inflight[chosen]
                if inflight < max_queue and waiting < max_queue and kv_usage < max_kv:
                    # Same bookkeeping as the stock policy so aborts route to the
                    # right engine and the local load estimate stays consistent.
                    self.lb_engines[candidate_idx][0] += self.client_count
                    self.reqs_in_flight[request.request_id] = chosen
                    self.engine_inflight[chosen] += 1
                    return chosen
        return orig_get(self, request)

    client_cls.get_core_engine_for_request = prefix_aware_get_core_engine
    setattr(client_cls, _PATCH_MARKER, True)
    return True
