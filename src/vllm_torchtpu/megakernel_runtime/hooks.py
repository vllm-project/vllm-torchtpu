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
from collections.abc import Mapping as RuntimeMapping
from typing import TYPE_CHECKING

from vllm_torchtpu import envs

from .config import import_plugin, missing_template, resolve_callable, select_plugin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

_SUPPRESS_ATTR = "_megakernel_suppress_depth"
_LAYOUT_APPLIED_ATTR = "_megakernel_layout_applied"
_BLOCK_SIZE_ATTR = "block_size"
_DCP_INTERLEAVE_ATTR = "dcp_kv_cache_interleave_size"
_CP_INTERLEAVE_ATTR = "cp_kv_cache_interleave_size"
_LAYOUT_KEYS = frozenset(
    {"block_size", "page_size", "mamba_slot_tokens", "dcp_interleave_size"}
)
_STREAM_PAIR_LEN = 2


@dataclasses.dataclass(frozen=True)
class StepContext:
    """The inputs of one runner forward, exactly as the stock model gets them.

    `model_kwargs` are the extra keyword arguments the runner passes to the
    stock model's forward; a megakernel that can't consume them declines the
    step. A megakernel that handles the step returns what the stock model
    would: hidden states, or `(hidden_states, aux_hidden_states)`; the runner
    normalizes both paths the same way.
    """

    input_ids: object
    positions: object
    inputs_embeds: object = None
    intermediate_tensors: object = None
    model_kwargs: Mapping[str, object] | None = None
    # The step's token ids when the runner fed the stock model
    # `inputs_embeds` built from them alone (a multimodal-capable model, but
    # no multimodal embeddings in this step); `None` otherwise.
    token_ids: object = None


@contextlib.contextmanager
def suppressed(runner: object) -> Iterator[None]:
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


def _suppress_depth(runner: object) -> int:
    """Return active suppression depth on `runner` without triggering mock synthesis."""
    depth = getattr(runner, "__dict__", {}).get(_SUPPRESS_ATTR, 0)
    return depth if isinstance(depth, int) else 0


def is_suppressed(runner: object) -> bool:
    """Whether `runner` is inside a `suppressed(runner)` scope."""
    return runner is not None and _suppress_depth(runner) > 0


def _plugin_hook(
    vllm_config: object, attr: str
) -> tuple[Callable[..., object], dict[str, object]] | None:
    """Resolve optional hook `attr` and options from the active megakernel plugin."""
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


def resolve_layout_request(vllm_config: object) -> dict[str, object] | None:
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
    if not isinstance(req, RuntimeMapping):
        msg = (
            "layout_request must return a Mapping or dataclass instance, "
            f"got {type(req).__name__}"
        )
        raise TypeError(msg)
    return dict(req)


def _validate_layout_request(req: Mapping[str, object]) -> None:
    """Validate layout request keys and positive integer values."""
    if unknown := sorted(set(req) - _LAYOUT_KEYS):
        msg = f"layout_request: unknown keys {unknown}"
        raise ValueError(msg)
    for key in _LAYOUT_KEYS & set(req):
        val = req[key]
        if type(val) is not int or val <= 0:
            msg = f"layout_request {key}={val!r} must be a positive int"
            raise ValueError(msg)
    if req.get("page_size", req.get("block_size")) != req.get(
        "block_size", req.get("page_size")
    ):
        msg = "layout_request: page_size and block_size disagree"
        raise ValueError(msg)


def _apply_cache_layout_fields(cache_cfg: object, req: Mapping[str, object]) -> None:
    """Apply layout request fields onto cache configuration."""
    _validate_layout_request(req)
    block = req.get("block_size", req.get("page_size"))
    if block is not None and not getattr(cache_cfg, "user_specified_block_size", False):
        setattr(cache_cfg, _BLOCK_SIZE_ATTR, block)


def _apply_dcp_interleave(
    req: Mapping[str, object],
    cache_cfg: object,
    parallel_config: object,
) -> None:
    """Apply `dcp_interleave_size` onto cache and parallel configurations."""
    if "dcp_interleave_size" not in req:
        return
    dcp_raw = req["dcp_interleave_size"]
    if not isinstance(dcp_raw, int):
        return
    dcp_size = dcp_raw
    if cache_cfg is not None:
        cur_block = getattr(cache_cfg, _BLOCK_SIZE_ATTR, None)
        if isinstance(cur_block, int) and cur_block % dcp_size != 0:
            msg = (
                f"layout_request: block_size ({cur_block}) must be divisible "
                f"by dcp_interleave_size ({dcp_size})"
            )
            raise ValueError(msg)
    if parallel_config is not None:
        setattr(parallel_config, _DCP_INTERLEAVE_ATTR, dcp_size)
        if hasattr(parallel_config, _CP_INTERLEAVE_ATTR):
            setattr(parallel_config, _CP_INTERLEAVE_ATTR, dcp_size)


def apply_layout_request(vllm_config: object) -> dict[str, object] | None:
    """Apply requested pre-sizing layout fields onto `vllm_config`."""
    cached = getattr(vllm_config, _LAYOUT_APPLIED_ATTR, None)
    if isinstance(cached, dict):
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
    _apply_dcp_interleave(req, cache_cfg, parallel_config)
    if vllm_config is not None:
        setattr(vllm_config, _LAYOUT_APPLIED_ATTR, dict(req))
    return req


WEIGHT_CONSUMED: object = object()


def load_tensor_stream(
    vllm_config: object,
    weights: Iterable[tuple[str, object]],
) -> Iterator[tuple[str, object]]:
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
        if (
            isinstance(out, tuple)
            and len(out) == _STREAM_PAIR_LEN
            and isinstance(out[0], str)
        ):
            yield (out[0], out[1])
        else:
            yield (name, tensor)


__all__ = [
    "WEIGHT_CONSUMED",
    "StepContext",
    "apply_layout_request",
    "is_suppressed",
    "load_tensor_stream",
    "resolve_layout_request",
    "suppressed",
]
