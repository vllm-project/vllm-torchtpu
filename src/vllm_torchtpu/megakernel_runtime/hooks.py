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
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .config import import_plugin, missing_template, resolve_callable, select_plugin

_SUPPRESS_ATTR = "_megakernel_suppress_depth"
_LAYOUT_APPLIED_ATTR = "_megakernel_layout_applied"
_LAYOUT_KEYS = frozenset(
    {"block_size", "page_size", "mamba_slot_tokens", "dcp_interleave_size"}
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

    Callers (such as a model runner's dummy / precompile / profile forwards)
    can wrap forwards in this scope so every stock graph those runs compile is
    compiled at startup even when the megakernel could have served the
    bucket. Nests; `runner` may be any object that accepts attributes.
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


def _plugin_hook(vllm_config: object, attr: str) -> tuple[Any, dict[str, Any]] | None:
    from vllm_torchtpu import envs

    model_cfg = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_cfg, "hf_config", None)
    cfg, resolved = select_plugin(
        envs.VLLM_TPU_MEGAKERNEL,
        getattr(vllm_config, "additional_config", None),
        hf_config,
    )
    if not cfg.enabled or resolved is None:
        return None
    ref, options = resolved
    if missing_template(ref, hf_config) is not None:
        return None
    plugin, module = import_plugin(ref)
    hook = getattr(plugin, attr, None)
    if hook is None:
        return None
    fn = resolve_callable(hook, module, what=attr)
    return fn, dict(options)


def resolve_layout_request(vllm_config: object) -> dict[str, Any] | None:
    """Call `plugin.layout_request` if defined and return requested fields."""
    res = _plugin_hook(vllm_config, "layout_request")
    if res is None:
        return None
    fn, options = res
    req = fn(vllm_config, options)
    if req is None:
        return None
    if dataclasses.is_dataclass(req) and not isinstance(req, type):
        req = dataclasses.asdict(req)
    return dict(req) if isinstance(req, Mapping) else None


def _validate_layout_request(req: Mapping[str, Any]) -> None:
    if unknown := sorted(set(req) - _LAYOUT_KEYS):
        raise ValueError(f"layout_request: unknown keys {unknown}")
    for key in _LAYOUT_KEYS & set(req):
        val = req[key]
        if type(val) is not int or val <= 0:
            raise ValueError(f"layout_request {key}={val!r} must be a positive int")
    if req.get("page_size", req.get("block_size")) != req.get(
        "block_size", req.get("page_size")
    ):
        raise ValueError("layout_request: page_size and block_size disagree")


def _apply_cache_layout_fields(cache_cfg: Any, req: Mapping[str, Any]) -> None:
    """Apply layout request fields onto cache configuration."""
    _validate_layout_request(req)
    block = req.get("block_size", req.get("page_size"))
    if block is not None and not getattr(cache_cfg, "user_specified_block_size", False):
        cache_cfg.block_size = block


def apply_layout_request(vllm_config: object) -> dict[str, Any] | None:
    """Apply requested pre-sizing layout fields onto `vllm_config`."""
    cached = getattr(vllm_config, _LAYOUT_APPLIED_ATTR, None)
    if cached is not None:
        return cached or None
    req = resolve_layout_request(vllm_config)
    if req is None:
        if vllm_config is not None:
            setattr(vllm_config, _LAYOUT_APPLIED_ATTR, {})
        return None
    _validate_layout_request(req)
    cache_cfg = getattr(vllm_config, "cache_config", None)
    if cache_cfg is not None:
        _apply_cache_layout_fields(cache_cfg, req)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if "dcp_interleave_size" in req:
        dcp_size = req["dcp_interleave_size"]
        if cache_cfg is not None:
            cur_block = getattr(cache_cfg, "block_size", None)
            if cur_block is not None and cur_block % dcp_size != 0:
                msg = (
                    f"layout_request: block_size ({cur_block}) must be divisible "
                    f"by dcp_interleave_size ({dcp_size})"
                )
                raise ValueError(msg)
        if parallel_config is not None:
            parallel_config.dcp_kv_cache_interleave_size = dcp_size
            if hasattr(parallel_config, "cp_kv_cache_interleave_size"):
                parallel_config.cp_kv_cache_interleave_size = dcp_size
    if vllm_config is not None:
        setattr(vllm_config, _LAYOUT_APPLIED_ATTR, dict(req))
    return req


WEIGHT_CONSUMED: object = object()


def load_tensor_stream(
    vllm_config: object,
    weights: Iterable[tuple[str, object]],
) -> Iterator[tuple[str, Any]]:
    """Stream weights through `plugin.load_tensor` before loading."""
    res = _plugin_hook(vllm_config, "load_tensor")
    if res is None:
        yield from weights
        return
    fn, options = res
    for name, tensor in weights:
        out = fn(name, tensor, options)
        if out is WEIGHT_CONSUMED:
            continue
        yield out if isinstance(out, tuple) and len(out) == 2 else (name, tensor)


__all__ = [
    "StepContext",
    "WEIGHT_CONSUMED",
    "load_tensor_stream",
    "apply_layout_request",
    "is_suppressed",
    "load_tensor_stream",
    "resolve_layout_request",
    "suppressed",
]
