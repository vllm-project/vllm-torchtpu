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
"""The megakernel's compile-cache salt (stdlib only).

The salt is the digest of the plugin's cache-layout request, or `None` when
the plugin makes none. It carries no plugin fingerprint or source hash, so
stock compile keys stay byte-identical with a megakernel on, and editing a
plugin never re-keys stock graphs.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import vllm_torchtpu.envs as envs

from .config import import_plugin, missing_template, resolve_callable, select_plugin
from .fingerprint import canonical_json


def compile_cache_salt(vllm_config: Any) -> str | None:
    """`sha256(canonical(layout_request(vllm_config, options)))[:16]`, or `None`."""
    model_cfg = getattr(vllm_config, "model_config", None)
    cfg, resolved = select_plugin(
        envs.VLLM_TPU_MEGAKERNEL,
        getattr(vllm_config, "additional_config", None),
        getattr(model_cfg, "hf_config", None),
    )
    if not cfg.enabled or resolved is None:
        return None
    ref, options = resolved
    if missing_template(ref) is not None:
        return None
    plugin, module = import_plugin(ref)
    if plugin.layout_request is None:
        return None
    fn = resolve_callable(plugin.layout_request, module, what="layout_request")
    req = fn(vllm_config, options)
    if req is None:
        return None
    if dataclasses.is_dataclass(req) and not isinstance(req, type):
        req = dataclasses.asdict(req)
    return hashlib.sha256(canonical_json(req).encode("utf-8")).hexdigest()[:16]


__all__ = ["compile_cache_salt"]
