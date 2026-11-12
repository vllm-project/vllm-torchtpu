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
"""Compile-cache salt and lazy plugin hook resolution for `megakernel.adapter`.

The salt is the SHA-256 prefix of the plugin's canonicalized cache-layout
request, or `None` when the plugin makes none. It carries no plugin source
hash, so editing a plugin never re-keys stock graphs.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import TYPE_CHECKING

import vllm_torchtpu.envs as envs

from .config import import_plugin, missing_template, resolve_callable, select_plugin
from .constants import HOOK_LAYOUT_REQUEST, PLUGIN_HOOK_NAMES
from .fingerprint import canonical_json

if TYPE_CHECKING:
    from collections.abc import Callable

_SALT_HEX_CHARS: int = 16


def resolve_plugin_hook(
    vllm_config: object,
    hook: str,
) -> tuple[Callable[..., object], dict[str, object]] | None:
    """Resolve an enabled megakernel plugin's optional hook callable and options.

    Args:
        vllm_config: Active vLLM configuration holding `additional_config` and
            `model_config.hf_config`.
        hook: Hook attribute name on `TpuMonoSpec` (`'layout_request'` or
            `'load_tensor'`).

    Returns:
        Tuple `(resolved_callable, options_dict)` when megakernel is enabled and
        the selected plugin defines `hook`, or `None` when disabled, missing a
        template, or `getattr(spec, hook)` is `None`.

    Raises:
        ValueError: If `hook` is not in `PLUGIN_HOOK_NAMES`.
    """
    if hook not in PLUGIN_HOOK_NAMES:
        msg = f"Unknown plugin hook {hook!r}"
        raise ValueError(msg)
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
    target = (
        plugin.layout_request if hook == HOOK_LAYOUT_REQUEST else plugin.load_tensor
    )
    if target is None:
        return None
    return resolve_callable(target, module, what=hook), dict(options)


def compile_cache_salt(vllm_config: object) -> str | None:
    """Compute the 16-hex-char compile-cache salt from `layout_request`, or `None`.

    Args:
        vllm_config: Active vLLM configuration passed to `layout_request`.

    Returns:
        16-character SHA-256 hex digest of the canonical JSON layout request, or
        `None` when megakernel is disabled or `layout_request` returns `None`.
    """
    resolved_hook = resolve_plugin_hook(vllm_config, HOOK_LAYOUT_REQUEST)
    if resolved_hook is None:
        return None
    fn, options = resolved_hook
    req = fn(vllm_config, options)
    if req is None:
        return None
    if dataclasses.is_dataclass(req) and not isinstance(req, type):
        req = dataclasses.asdict(req)
    digest = hashlib.sha256(canonical_json(req).encode("utf-8")).hexdigest()
    return digest[:_SALT_HEX_CHARS]


__all__ = ["compile_cache_salt", "resolve_plugin_hook"]
