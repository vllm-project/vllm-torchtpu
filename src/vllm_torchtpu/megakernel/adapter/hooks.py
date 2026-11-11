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
import functools
import types
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from .salt import resolve_plugin_hook

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_SUPPRESS_ATTR = "_megakernel_suppress_depth"


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


_MISSING = object()


@contextlib.contextmanager
def _temporary_attr(target: object, name: str, value: object) -> Iterator[None]:
    """Temporarily set attribute `name` on `target` and restore prior state."""
    inst_dict = getattr(target, "__dict__", None)
    had_own = inst_dict is not None and name in inst_dict
    is_slot = isinstance(getattr(type(target), name, None), types.MemberDescriptorType)
    previous = (
        inst_dict[name]
        if had_own
        else (getattr(target, name, _MISSING) if is_slot else _MISSING)
    )
    setattr(target, name, value)
    try:
        yield
    finally:
        if had_own or (is_slot and previous is not _MISSING):
            setattr(target, name, previous)
        else:
            with contextlib.suppress(AttributeError):
                delattr(target, name)


def _observe_weights[TensorT](
    weights: Iterable[tuple[str, TensorT]],
    fn: Callable[..., object],
    options: dict[str, object],
) -> Iterator[tuple[str, TensorT]]:
    """Pass each `(name, tensor)` pair to `fn` before yielding it unchanged."""
    for name, tensor in weights:
        fn(name, tensor, options)
        yield name, tensor


def load_tensor_stream[TensorT](
    vllm_config: object,
    weights: Iterable[tuple[str, TensorT]],
) -> Iterator[tuple[str, TensorT]]:
    """Stream `weights` through `plugin.load_tensor` when enabled."""
    resolved = resolve_plugin_hook(vllm_config, "load_tensor")
    if resolved is None:
        return iter(weights)
    fn, options = resolved
    return _observe_weights(weights, fn, options)


class _OuterLoadWeightsProxy:
    """Forward `model.load_weights`, observing only the outer weight stream."""

    def __init__(
        self,
        orig_load: Callable[..., object],
        fn: Callable[..., object],
        options: dict[str, object],
    ) -> None:
        """Bind original `load_weights` and observation callback."""
        self._orig = orig_load
        self._fn = fn
        self._options = options
        self._active = False

    def __call__(
        self,
        weights: Iterable[tuple[str, object]],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Observe `weights` on entry; skip nested `self.load_weights` calls."""
        if self._active:
            return self._orig(weights, *args, **kwargs)
        self._active = True
        try:
            return self._orig(
                _observe_weights(weights, self._fn, self._options),
                *args,
                **kwargs,
            )
        finally:
            self._active = False


def _call_get_all_observed(
    orig_get_all: Callable[..., Iterable[tuple[str, object]]],
    fn: Callable[..., object],
    options: dict[str, object],
    *args: object,
    **kwargs: object,
) -> Iterator[tuple[str, object]]:
    """Return observed iterator from `model_loader.get_all_weights`."""
    return _observe_weights(orig_get_all(*args, **kwargs), fn, options)


def _call_loader_observed(
    orig_load_weights: Callable[..., object],
    fn: Callable[..., object],
    options: dict[str, object],
    model: object,
    *args: object,
    **kwargs: object,
) -> object:
    """Wrap `model.load_weights` with `_OuterLoadWeightsProxy` during load."""
    orig_model_lw = getattr(model, "load_weights", None)
    if not callable(orig_model_lw):
        return orig_load_weights(model, *args, **kwargs)
    proxy = _OuterLoadWeightsProxy(orig_model_lw, fn, options)
    with _temporary_attr(model, "load_weights", proxy):
        return orig_load_weights(model, *args, **kwargs)


@contextlib.contextmanager
def observe_model_loader(vllm_config: object, model_loader: object) -> Iterator[None]:
    """Show each streamed checkpoint tensor to `plugin.load_tensor` during load."""
    resolved = resolve_plugin_hook(vllm_config, "load_tensor")
    if resolved is None:
        yield
        return
    fn, options = resolved

    override: tuple[str, object] | None = None
    if callable(orig_get_all := getattr(model_loader, "get_all_weights", None)):
        override = (
            "get_all_weights",
            functools.partial(_call_get_all_observed, orig_get_all, fn, options),
        )
    elif callable(orig_lw := getattr(model_loader, "load_weights", None)):
        override = (
            "load_weights",
            functools.partial(_call_loader_observed, orig_lw, fn, options),
        )
    if override is None:
        yield
        return
    with _temporary_attr(model_loader, *override):
        yield


__all__ = [
    "StepContext",
    "is_suppressed",
    "load_tensor_stream",
    "observe_model_loader",
    "suppressed",
]
