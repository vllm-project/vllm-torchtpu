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
from collections.abc import Mapping
from typing import Any

import vllm_torchtpu.envs as envs
from vllm_torchtpu.megakernel.errors import MegakernelConfigError

from .config import import_plugin, missing_template, resolve_callable, select_plugin
from .fingerprint import canonical_json


def resolve_layout_request(
    vllm_config: Any,
    *,
    select_fn: Any = None,
    import_fn: Any = None,
) -> dict[str, Any] | None:
    """Call `plugin.layout_request(vllm_config, options)` if defined."""
    sel = select_plugin if select_fn is None else select_fn
    imp = import_plugin if import_fn is None else import_fn
    model_cfg = getattr(vllm_config, "model_config", None)
    hf_cfg = getattr(model_cfg, "hf_config", None)
    cfg, resolved = sel(
        envs.VLLM_TPU_MEGAKERNEL,
        getattr(vllm_config, "additional_config", None),
        hf_cfg,
    )
    if not cfg.enabled or resolved is None:
        return None
    ref, options = resolved
    if missing_template(ref, hf_cfg) is not None:
        return None
    plugin, module = imp(ref)
    layout_req = getattr(plugin, "layout_request", None)
    if layout_req is None:
        return None
    fn = resolve_callable(layout_req, module, what="layout_request")
    req = fn(vllm_config, dict(options))
    if req is None:
        return None
    if dataclasses.is_dataclass(req) and not isinstance(req, type):
        req = dataclasses.asdict(req)
    if not isinstance(req, Mapping):
        raise MegakernelConfigError(
            "megakernel layout_request must return a Mapping or dataclass "
            f"instance, got {type(req).__name__}"
        )
    return dict(req)


def compile_cache_salt(vllm_config: Any) -> str | None:
    """`sha256(canonical(layout_request(vllm_config, options)))[:16]`, or `None`."""
    req = resolve_layout_request(vllm_config)
    if req is None:
        return None
    return hashlib.sha256(canonical_json(req).encode("utf-8")).hexdigest()[:16]


__all__ = ["compile_cache_salt", "resolve_layout_request"]
