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
"""Runner-facing hook types for the megakernel adapter.

Kept free of `jax` / `torch` imports so the runner can import them at module
load without pulling in the megakernel library.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.errors import MegakernelConfigError

from .config import import_plugin, missing_template, resolve_callable, select_plugin
from .salt import resolve_layout_request as _resolve_layout_request

logger = init_logger(__name__)

_SUPPRESS_ATTR = "_megakernel_suppress_depth"
_LOAD_TENSOR_PARAMS_WITH_OPTIONS: int = 3
_LAYOUT_APPLIED_ATTR = "_megakernel_layout_applied"
LAYOUT_FLOOR_KEYS: tuple[str, ...] = (
    "block_size",
    "mamba_slot_tokens",
    "mamba_block_size",
    "slot_stride_tokens",
)


@dataclasses.dataclass(frozen=True)
class StepContext:
    """The inputs of one runner forward, exactly as the stock model gets them.

    `model_kwargs` are the extra keyword arguments the runner passes to the
    stock model's forward; a megakernel that can't consume them declines the
    step. A megakernel that handles the step returns what the stock model
    would: hidden states, or `(hidden_states, aux_hidden_states)`; the runner
    normalizes both paths the same way.
    """

    input_ids: Any
    positions: Any
    inputs_embeds: Any = None
    intermediate_tensors: Any = None
    model_kwargs: Mapping[str, Any] | None = None
    # The step's token ids when the runner fed the stock model
    # `inputs_embeds` built from them alone (a multimodal-capable model, but
    # no multimodal embeddings in this step); `None` otherwise.
    token_ids: Any = None


@contextlib.contextmanager
def suppressed(runner: Any) -> Iterator[None]:
    """Force the stock model forward for every step of `runner` in this scope.

    The runner wraps its dummy / precompile / profile forwards in this, so
    every stock graph those runs compile is compiled at startup even when the
    megakernel could have served the bucket. Nests; `runner` may be any object
    that accepts attributes.
    """
    depth = _suppress_depth(runner)
    setattr(runner, _SUPPRESS_ATTR, depth + 1)
    try:
        yield
    finally:
        setattr(runner, _SUPPRESS_ATTR, depth)


def _suppress_depth(runner: Any) -> int:
    # Read the instance dict: attribute-synthesizing objects (mocks) must not
    # count as suppressed.
    depth = getattr(runner, "__dict__", {}).get(_SUPPRESS_ATTR, 0)
    return depth if isinstance(depth, int) else 0


def is_suppressed(runner: Any) -> bool:
    """Whether `runner` is inside a `suppressed(runner)` scope."""
    return runner is not None and _suppress_depth(runner) > 0


def resolve_layout_request(vllm_config: object) -> dict[str, Any] | None:
    """Call `plugin.layout_request(vllm_config, options)` if defined."""
    return _resolve_layout_request(
        vllm_config, select_fn=select_plugin, import_fn=import_plugin
    )


def _apply_block_size_request(cache_cfg: Any, req: Mapping[str, Any]) -> int | None:
    """Apply requested `block_size` onto `cache_config`; return original block size."""
    orig_block_size = (
        getattr(cache_cfg, "block_size", None) if cache_cfg is not None else None
    )
    if "block_size" not in req or req["block_size"] is None:
        return orig_block_size
    if cache_cfg is None or not hasattr(cache_cfg, "block_size"):
        raise MegakernelConfigError(
            "megakernel layout_request sets block_size, but "
            "vllm_config.cache_config.block_size is unavailable"
        )
    new_val = int(req["block_size"])
    old_val = getattr(cache_cfg, "block_size", None)
    if getattr(cache_cfg, "user_specified_block_size", False):
        if old_val != new_val:
            logger.info(
                "Keeping user-specified cache_config.block_size=%s over "
                "megakernel layout_request=%s",
                old_val,
                new_val,
            )
        return orig_block_size
    if old_val != new_val:
        logger.info(
            "Megakernel layout_request set cache_config.block_size: %s -> %s",
            old_val,
            new_val,
        )
    cache_cfg.block_size = new_val
    return orig_block_size


def _apply_interleave_request(
    par_cfg: Any, req: Mapping[str, Any], orig_block_size: int | None
) -> None:
    """Apply requested `cp_kv_cache_interleave_size` onto `parallel_config`."""
    if (
        "cp_kv_cache_interleave_size" not in req
        or req["cp_kv_cache_interleave_size"] is None
    ):
        return
    if par_cfg is None or not hasattr(par_cfg, "cp_kv_cache_interleave_size"):
        raise MegakernelConfigError(
            "megakernel layout_request sets cp_kv_cache_interleave_size, but "
            "vllm_config.parallel_config.cp_kv_cache_interleave_size is unavailable"
        )
    new_val = int(req["cp_kv_cache_interleave_size"])
    if new_val <= 0:
        raise MegakernelConfigError(
            "megakernel layout_request cp_kv_cache_interleave_size must be > 0, "
            f"got {req['cp_kv_cache_interleave_size']!r}"
        )
    old_val = getattr(par_cfg, "cp_kv_cache_interleave_size", None)
    user_set = getattr(
        par_cfg, "user_specified_cp_kv_cache_interleave_size", False
    ) or (
        old_val is not None
        and int(old_val) not in (1, int(orig_block_size or 0), new_val)
    )
    if user_set and old_val != new_val:
        raise MegakernelConfigError(
            "megakernel layout_request "
            f"cp_kv_cache_interleave_size={new_val} conflicts with "
            f"user-specified cp_kv_cache_interleave_size={old_val}"
        )
    if old_val != new_val:
        logger.info(
            "Megakernel layout_request set "
            "parallel_config.cp_kv_cache_interleave_size: %s -> %s",
            old_val,
            new_val,
        )
    par_cfg.cp_kv_cache_interleave_size = new_val


def apply_layout_request(vllm_config: object) -> dict[str, Any] | None:
    """Apply requested pre-sizing layout fields onto `vllm_config`."""
    cached = getattr(vllm_config, _LAYOUT_APPLIED_ATTR, None)
    if isinstance(cached, dict):
        return cached
    req = resolve_layout_request(vllm_config)
    if req is None:
        return None
    for k in LAYOUT_FLOOR_KEYS:
        if k in req and req[k] is not None and int(req[k]) <= 0:
            raise MegakernelConfigError(
                f"megakernel layout_request {k} must be > 0, got {req[k]!r}"
            )
    cache_cfg = getattr(vllm_config, "cache_config", None)
    par_cfg = getattr(vllm_config, "parallel_config", None)
    orig_block_size = _apply_block_size_request(cache_cfg, req)
    _apply_interleave_request(par_cfg, req, orig_block_size)
    setattr(vllm_config, _LAYOUT_APPLIED_ATTR, dict(req))
    return req


def load_tensor_stream(
    vllm_config: object,
    weights: Iterable[tuple[str, object]],
) -> Iterator[tuple[str, Any]]:
    """Stream weights through `plugin.load_tensor` before loading.

    Yields `(name, tensor)` unchanged. If the megakernel is enabled and
    the plugin defines `load_tensor`, calls `load_tensor(name, tensor, options)`
    (or 2-arg `load_tensor(name, tensor)`) on each pair.
    """
    model_cfg = getattr(vllm_config, "model_config", None)
    cfg, resolved = select_plugin(
        envs.VLLM_TPU_MEGAKERNEL,
        getattr(vllm_config, "additional_config", None),
        getattr(model_cfg, "hf_config", None),
    )
    fn = None
    options: dict[str, Any] = {}
    if cfg.enabled and resolved is not None:
        ref, opts = resolved
        options = dict(opts)
        if missing_template(ref) is None:
            plugin, module = import_plugin(ref)
            hook = getattr(plugin, "load_tensor", None)
            if hook is not None:
                fn = resolve_callable(hook, module, what="load_tensor")
    if fn is None:
        yield from weights
        return

    try:
        sig = inspect.signature(fn)
        takes_options = len(sig.parameters) >= _LOAD_TENSOR_PARAMS_WITH_OPTIONS
    except (ValueError, TypeError):
        takes_options = True
    for name, tensor in weights:
        if takes_options:
            fn(name, tensor, options)
        else:
            fn(name, tensor)
        yield name, tensor


__all__ = [
    "LAYOUT_FLOOR_KEYS",
    "StepContext",
    "apply_layout_request",
    "is_suppressed",
    "load_tensor_stream",
    "resolve_layout_request",
    "suppressed",
]
