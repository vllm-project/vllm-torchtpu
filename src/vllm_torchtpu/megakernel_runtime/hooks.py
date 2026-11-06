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
"""Runner-facing hooks for megakernel layout pre-sizing and weight streaming.

Kept free of `jax` and `torch` imports so platform and runner modules can
import these hooks at module load without pulling in the megakernel library.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
from typing import TYPE_CHECKING, Any, Final, Protocol

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime.contract import LayoutRequest

from .config import import_plugin, select_plugin
from .salt import SaltConfigLike, parse_layout_request, resolve_plugin_hook
from .salt import resolve_layout_request as _resolve_raw_layout_request

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

logger = init_logger(__name__)

_SUPPRESS_ATTR: Final[str] = "_megakernel_suppress_depth"
WEIGHT_CONSUMED: Final[object] = object()


class CacheConfigLike(Protocol):
    """Structural contract for `vllm_config.cache_config` during pre-sizing."""

    block_size: Any
    user_specified_block_size: bool


class ParallelConfigLike(Protocol):
    """Structural contract for `vllm_config.parallel_config` during pre-sizing."""

    cp_kv_cache_interleave_size: int
    dcp_kv_cache_interleave_size: int


class VllmConfigLike(SaltConfigLike, Protocol):
    """Structural contract for `VllmConfig` passed into megakernel hooks."""

    @property
    def cache_config(self) -> CacheConfigLike | None:
        """Return `vllm_config.cache_config` when present."""

    @property
    def parallel_config(self) -> ParallelConfigLike | None:
        """Return `vllm_config.parallel_config` when present."""


@dataclasses.dataclass(frozen=True)
class StepContext:
    """Inputs of one runner forward step, matching the baseline model call.

    `model_kwargs` holds extra keyword arguments passed to the baseline model
    forward; a megakernel that cannot consume them declines the step. A
    megakernel that handles the step returns hidden states or
    `(hidden_states, aux_hidden_states)` for the runner to normalize.
    """

    input_ids: Any
    positions: Any
    inputs_embeds: Any = None
    intermediate_tensors: Any = None
    model_kwargs: Mapping[str, Any] | None = None
    token_ids: Any = None


@contextlib.contextmanager
def suppressed(runner: object) -> Iterator[None]:
    """Force the baseline model forward for every step of `runner` in scope."""
    depth = _suppress_depth(runner)
    vars(runner)[_SUPPRESS_ATTR] = depth + 1
    try:
        yield
    finally:
        vars(runner)[_SUPPRESS_ATTR] = depth


def _suppress_depth(runner: object) -> int:
    """Return active suppression nesting depth without triggering mock attrs."""
    inst_dict = vars(runner) if hasattr(runner, "__dict__") else {}
    depth = inst_dict.get(_SUPPRESS_ATTR, 0)
    return depth if isinstance(depth, int) else 0


def is_suppressed(runner: object) -> bool:
    """Return whether `runner` is inside a `suppressed(runner)` scope."""
    return runner is not None and _suppress_depth(runner) > 0


def resolve_layout_request(vllm_config: SaltConfigLike) -> LayoutRequest | None:
    """Resolve and validate `plugin.layout_request(vllm_config, options)`."""
    raw = _resolve_raw_layout_request(
        vllm_config, select_fn=select_plugin, import_fn=import_plugin
    )
    return None if raw is None else parse_layout_request(raw)


def _apply_block_size_request(
    cache_cfg: CacheConfigLike | None, requested_block_size: int | None
) -> int | None:
    """Apply `requested_block_size` onto `cache_cfg`; return prior block size."""
    orig_block_size = None if cache_cfg is None else cache_cfg.block_size
    if requested_block_size is None:
        return orig_block_size
    if cache_cfg is None:
        msg = (
            "megakernel layout_request sets block_size, but "
            "vllm_config.cache_config is None"
        )
        raise MegakernelConfigError(msg)
    if cache_cfg.user_specified_block_size:
        if orig_block_size != requested_block_size:
            logger.info(
                "Keeping user-specified cache_config.block_size=%s over "
                "megakernel layout_request=%s",
                orig_block_size,
                requested_block_size,
            )
        return orig_block_size
    if orig_block_size != requested_block_size:
        logger.info(
            "Megakernel layout_request set cache_config.block_size: %s -> %s",
            orig_block_size,
            requested_block_size,
        )
    cache_cfg.block_size = requested_block_size
    return orig_block_size


def _apply_interleave_request(
    par_cfg: ParallelConfigLike | None,
    cache_cfg: CacheConfigLike | None,
    requested_interleave: int | None,
    orig_block_size: int | None,
) -> None:
    """Apply `requested_interleave` onto `par_cfg` when compatible with block size."""
    if requested_interleave is None:
        return
    if par_cfg is None:
        msg = (
            "megakernel layout_request sets cp_kv_cache_interleave_size, but "
            "vllm_config.parallel_config is None"
        )
        raise MegakernelConfigError(msg)
    old_val = par_cfg.cp_kv_cache_interleave_size
    allowed_prior = (1, orig_block_size or 0, requested_interleave)
    if old_val not in allowed_prior:
        msg = (
            "megakernel layout_request "
            f"cp_kv_cache_interleave_size={requested_interleave} conflicts with "
            f"configured cp_kv_cache_interleave_size={old_val}"
        )
        raise MegakernelConfigError(msg)
    if (
        cache_cfg is not None
        and cache_cfg.user_specified_block_size
        and cache_cfg.block_size is not None
        and (
            cache_cfg.block_size < requested_interleave
            or cache_cfg.block_size % requested_interleave != 0
        )
    ):
        logger.info(
            "Skipping megakernel cp_kv_cache_interleave_size=%s because "
            "user-specified block_size=%s is not a multiple of it",
            requested_interleave,
            cache_cfg.block_size,
        )
        return
    if old_val != requested_interleave:
        logger.info(
            "Megakernel layout_request set "
            "parallel_config.cp_kv_cache_interleave_size: %s -> %s",
            old_val,
            requested_interleave,
        )
    par_cfg.cp_kv_cache_interleave_size = requested_interleave
    if par_cfg.dcp_kv_cache_interleave_size in allowed_prior:
        par_cfg.dcp_kv_cache_interleave_size = requested_interleave


def apply_layout_request(vllm_config: VllmConfigLike) -> LayoutRequest | None:
    """Apply validated plugin pre-sizing fields onto `vllm_config`."""
    req = resolve_layout_request(vllm_config)
    if req is None:
        return None
    cache_cfg = vllm_config.cache_config
    par_cfg = vllm_config.parallel_config
    orig_block_size = _apply_block_size_request(cache_cfg, req.block_size)
    _apply_interleave_request(
        par_cfg, cache_cfg, req.cp_kv_cache_interleave_size, orig_block_size
    )
    return req


def _resolve_load_tensor_hook(
    vllm_config: SaltConfigLike,
) -> tuple[Callable[..., object], dict[str, object]] | None:
    """Resolve `(load_tensor_fn, options)` when enabled and declared, else `None`."""
    return resolve_plugin_hook(
        vllm_config,
        "load_tensor",
        select_fn=select_plugin,
        import_fn=import_plugin,
    )


def _accepts_options_arg(fn: Callable[..., object]) -> bool:
    """Return whether `fn` accepts 3 positional arguments `(name, tensor, options)`."""
    try:
        inspect.signature(fn).bind_partial(None, None, None)
    except TypeError:
        return False
    except ValueError:
        return True
    return True


def _stream_with_hook[TensorT](
    weights: Iterable[tuple[str, TensorT]],
    fn: Callable[..., object],
    options: dict[str, object],
) -> Iterator[tuple[str, TensorT]]:
    """Yield `(name, tensor)` pairs not consumed by `fn`."""
    takes_options = _accepts_options_arg(fn)
    for name, tensor in weights:
        outcome = fn(name, tensor, options) if takes_options else fn(name, tensor)
        if outcome is WEIGHT_CONSUMED:
            continue
        yield name, tensor


def load_tensor_stream[TensorT](
    vllm_config: SaltConfigLike,
    weights: Iterable[tuple[str, TensorT]],
) -> Iterator[tuple[str, TensorT]]:
    """Stream weights through `plugin.load_tensor` before PyTorch loading.

    Yields `(name, tensor)` unchanged unless `plugin.load_tensor` returns
    `WEIGHT_CONSUMED`, which drops tensors already packed into the megakernel.
    """
    resolved = _resolve_load_tensor_hook(vllm_config)
    if resolved is None:
        yield from weights
        return
    fn, options = resolved
    yield from _stream_with_hook(weights, fn, options)


@contextlib.contextmanager
def _temporary_instance_attr(
    target: object, name: str, value: object
) -> Iterator[None]:
    """Temporarily assign `name` on `vars(target)` and restore it on exit."""
    target_vars = vars(target)
    had_own = name in target_vars
    previous = target_vars.get(name)
    target_vars[name] = value
    try:
        yield
    finally:
        if had_own:
            target_vars[name] = previous
        else:
            target_vars.pop(name, None)


def _call_observed_get_all(
    orig_get_all: Callable[..., Iterable[tuple[str, object]]],
    fn: Callable[..., object],
    options: dict[str, object],
    *args: object,
    **kwargs: object,
) -> Iterator[tuple[str, object]]:
    """Stream `model_loader.get_all_weights` through `_stream_with_hook`."""
    return _stream_with_hook(orig_get_all(*args, **kwargs), fn, options)


def _call_wrapped_model_lw(
    orig_model_lw: Callable[..., object],
    fn: Callable[..., object],
    options: dict[str, object],
    weights: Iterable[tuple[str, object]],
    *w_args: object,
    **w_kwargs: object,
) -> object:
    """Pass streamed weights into `model.load_weights`."""
    return orig_model_lw(_stream_with_hook(weights, fn, options), *w_args, **w_kwargs)


def _call_observed_load_weights(
    orig_load_weights: Callable[..., object],
    fn: Callable[..., object],
    options: dict[str, object],
    model: object,
    *args: object,
    **kwargs: object,
) -> object:
    """Stream weights into `model.load_weights` on fallback loaders."""
    orig_model_lw = getattr(model, "load_weights", None)
    if not callable(orig_model_lw):
        return orig_load_weights(model, *args, **kwargs)
    wrapped_model_lw = functools.partial(
        _call_wrapped_model_lw, orig_model_lw, fn, options
    )
    with _temporary_instance_attr(model, "load_weights", wrapped_model_lw):
        return orig_load_weights(model, *args, **kwargs)


@contextlib.contextmanager
def observe_model_loader(
    vllm_config: SaltConfigLike, model_loader: object
) -> Iterator[None]:
    """Intercept `model_loader` weight iteration when `spec.load_tensor` is active."""
    resolved = _resolve_load_tensor_hook(vllm_config)
    if resolved is None:
        yield
        return
    fn, options = resolved
    orig_get_all = getattr(model_loader, "get_all_weights", None)
    if callable(orig_get_all):
        wrapped_get_all = functools.partial(
            _call_observed_get_all, orig_get_all, fn, options
        )
        with _temporary_instance_attr(model_loader, "get_all_weights", wrapped_get_all):
            yield
        return

    orig_load_weights = getattr(model_loader, "load_weights", None)
    if callable(orig_load_weights):
        wrapped_lw = functools.partial(
            _call_observed_load_weights, orig_load_weights, fn, options
        )
        with _temporary_instance_attr(model_loader, "load_weights", wrapped_lw):
            yield
        return
    yield


__all__ = [
    "WEIGHT_CONSUMED",
    "CacheConfigLike",
    "LayoutRequest",
    "ParallelConfigLike",
    "StepContext",
    "VllmConfigLike",
    "apply_layout_request",
    "is_suppressed",
    "load_tensor_stream",
    "observe_model_loader",
    "resolve_layout_request",
    "suppressed",
]
