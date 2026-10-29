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
"""Qwen3.5 megakernel template (the `qwen35` shorthand).

Serve it with `VLLM_TPU_MEGAKERNEL=1` (by the checkpoint's arch), `qwen35`, or
`--additional-config '{"megakernel": {"plugin": "<this package>"}}'`;
template settings (`profile`, `mode`, `buckets`, ...) go in `options`.
Importing this module loads no torch / jax / vLLM: `create` is imported lazily.
"""

from __future__ import annotations

from vllm_torchtpu.megakernel_runtime.plugin import MegakernelPlugin

PLUGIN = MegakernelPlugin(
    name="qwen35",
    version="1",
    create=".adapter:create",
    capabilities=frozenset({"spec_decode", "mrope", "prefix_caching", "kv_offload"}),
    # Every decline `create` / `bind` can return beyond the runtime's own codes,
    # so peers log `peer_declined:<code>` rather than `other`.
    decline_codes=(
        "non_moe_checkpoint",
        "invalid_geometry",
        "geometry_mismatch",
        "backbone_not_found",
        "layer_count_mismatch",
        "unaligned_cache_layout",
        "unsupported_tp_size",
        "unsupported_tpu_chip",
        "empty_buckets",
    ),
)
