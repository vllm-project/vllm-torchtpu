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
default compile keys stay byte-identical with a megakernel on, and editing a
plugin never re-keys baseline graphs.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

import vllm_torchtpu.envs as envs
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime.contract import LayoutRequest, TpuMonoSpec

from .config import (
    MegakernelConfig,
    PluginRef,
    import_plugin,
    missing_template,
    resolve_callable,
    select_plugin,
)
from .fingerprint import canonical_json

if TYPE_CHECKING:
    from types import ModuleType


class _ModelConfigLike(Protocol):
    """Minimal structural view of `vllm_config.model_config` for salt resolution."""

    @property
    def hf_config(self) -> object:
        """Return the HuggingFace model config object."""


class SaltConfigLike(Protocol):
    """Minimal structural view of `VllmConfig` read by `compile_cache_salt`."""

    @property
    def model_config(self) -> _ModelConfigLike | None:
        """Return `vllm_config.model_config` when present."""

    @property
    def additional_config(self) -> object:
        """Return `vllm_config.additional_config` mapping when present."""


SelectFn = Callable[
    [object, object, object],
    tuple[MegakernelConfig, tuple[PluginRef, dict[str, object]] | None],
]
ImportFn = Callable[[PluginRef], tuple[TpuMonoSpec, "ModuleType"]]


def resolve_plugin_hook(
    vllm_config: SaltConfigLike,
    hook_name: str,
    *,
    select_fn: SelectFn | None = None,
    import_fn: ImportFn | None = None,
) -> tuple[Callable[..., object], dict[str, object]] | None:
    """Resolve `(callable, options)` for `TpuMonoSpec.<hook_name>`, or `None`."""
    sel = select_plugin if select_fn is None else select_fn
    imp = import_plugin if import_fn is None else import_fn
    model_cfg = vllm_config.model_config
    hf_cfg = model_cfg.hf_config if model_cfg is not None else None
    cfg, resolved = sel(
        envs.VLLM_TPU_MEGAKERNEL,
        vllm_config.additional_config,
        hf_cfg,
    )
    if not cfg.enabled or resolved is None:
        return None
    ref, options = resolved
    if missing_template(ref, hf_cfg) is not None:
        return None
    plugin, module = imp(ref)
    target = getattr(plugin, hook_name, None)
    if target is None:
        return None
    fn = resolve_callable(target, module, what=hook_name)
    return fn, dict(options)


def resolve_layout_request(
    vllm_config: SaltConfigLike,
    *,
    select_fn: SelectFn | None = None,
    import_fn: ImportFn | None = None,
) -> dict[str, object] | None:
    """Call `plugin.layout_request(vllm_config, options)` if defined."""
    resolved = resolve_plugin_hook(
        vllm_config,
        "layout_request",
        select_fn=select_fn,
        import_fn=import_fn,
    )
    if resolved is None:
        return None
    fn, options = resolved
    req = fn(vllm_config, options)
    if req is None:
        return None
    if isinstance(req, LayoutRequest):
        return dict(req.to_dict())
    if dataclasses.is_dataclass(req) and not isinstance(req, type):
        req = dataclasses.asdict(req)
    if not isinstance(req, Mapping):
        msg = (
            "megakernel layout_request must return a LayoutRequest, Mapping, or "
            f"dataclass instance, got {type(req).__name__}"
        )
        raise MegakernelConfigError(msg)
    return {str(k): v for k, v in req.items()}


def compile_cache_salt(vllm_config: SaltConfigLike) -> str | None:
    """Return `sha256(canonical(layout_request(vllm_config, options)))[:16]`."""
    req = resolve_layout_request(vllm_config)
    if req is None:
        return None
    return hashlib.sha256(canonical_json(req).encode("utf-8")).hexdigest()[:16]


_LAYOUT_ALIASES: Mapping[str, str] = {
    "page_size": "block_size",
    "dcp_interleave_size": "cp_kv_cache_interleave_size",
    "dcp_kv_cache_interleave_size": "cp_kv_cache_interleave_size",
}
_LAYOUT_FIELDS = ("block_size", "cp_kv_cache_interleave_size", "mamba_slot_tokens")


def parse_layout_request(raw: object) -> LayoutRequest:
    """Normalize a `LayoutRequest`, dataclass instance, or alias mapping."""
    if isinstance(raw, LayoutRequest):
        return raw
    if dataclasses.is_dataclass(raw) and not isinstance(raw, type):
        raw = dataclasses.asdict(raw)
    if not isinstance(raw, Mapping):
        msg = "layout_request must be a LayoutRequest, Mapping, or dataclass"
        raise MegakernelConfigError(msg)
    parsed: dict[str, int] = {}
    for raw_key, value in raw.items():
        key_name = str(raw_key)
        canonical = _LAYOUT_ALIASES.get(key_name, key_name)
        if canonical not in _LAYOUT_FIELDS:
            msg = f"layout_request has unknown key {key_name!r}"
            raise MegakernelConfigError(msg)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                msg = f"layout_request.{key_name} must be a positive int, got {value!r}"
                raise MegakernelConfigError(msg)
            if parsed.setdefault(canonical, value) != value:
                msg = f"layout_request has conflicting values for {canonical}"
                raise MegakernelConfigError(msg)
    return LayoutRequest(**parsed)


__all__ = [
    "SaltConfigLike",
    "compile_cache_salt",
    "parse_layout_request",
    "resolve_layout_request",
    "resolve_plugin_hook",
]
