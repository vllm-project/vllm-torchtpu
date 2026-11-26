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
"""Named MiniMax M3 recipes: profile -> {slot: variants in order of preference}.

Every profile pins every slot of `variants.VARIANTS`. `DEFAULT_PROFILE` is
the proven-best served recipe (the AgentX curve: TP8, EP, DCP 2 / 4 / 8,
DSpark); `vllm serve` uses it with no extra flags. Pick another one with
``--additional-config '{"model_plan": {"profile": "<name>"}}'``; the AgentX
c32 / c48 / c64 points (DCP 4) serve `agentx-int4` with --kv-cache-dtype fp8.
"""

DEFAULT_PROFILE = "agentx"

PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    # Served best (v2d / v3 FULLs): Pallas TP all-reduce up to 512 KiB views,
    # fused spec verify.
    "agentx": {
        "model": ("msa_sparse",),
        "draft": ("dspark",),
        "tp_all_reduce": ("pallas_512k", "xla"),
        "spec_verify": ("fused", "unfused"),
        "msa_cache": ("kv_cache_dtype",),
    },
    # agentx with 4-bit MSA caches (needs --kv-cache-dtype fp8: the DSpark
    # draft's KV stays fp8). Served at c32 / c48 / c64 with DCP 4 and KV
    # offload; c4 / c16 (DCP 2) keep fp8 MSA caches (agentx).
    "agentx-int4": {
        "model": ("msa_sparse",),
        "draft": ("dspark",),
        "tp_all_reduce": ("pallas_512k", "xla"),
        "spec_verify": ("fused", "unfused"),
        "msa_cache": ("int4",),
    },
}
