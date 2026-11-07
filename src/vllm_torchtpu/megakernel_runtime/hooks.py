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


@contextlib.contextmanager
def _temporary_attr(target: object, name: str, value: object) -> Iterator[None]:
    """Temporarily set instance attribute `name` on `target` and restore on exit."""
    inst_vars = vars(target)
    had_own = name in inst_vars
    previous = inst_vars.get(name)
    inst_vars[name] = value
    try:
        yield
    finally:
        if had_own:
            inst_vars[name] = previous
        else:
            inst_vars.pop(name, None)


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
        yield from weights
        return
    fn, options = resolved
    yield from _observe_weights(weights, fn, options)


def _wrap_load_weights(
    orig_load_weights: Callable[..., object],
    fn: Callable[..., object],
    options: dict[str, object],
) -> Callable[..., object]:
    """Wrap `model_loader.load_weights` so `model.load_weights` sees each tensor."""

    def observed_loader_lw(model: object, *args: object, **kwargs: object) -> object:
        """Intercept outer `model.load_weights` stream once per call."""
        orig_model_lw = getattr(model, "load_weights", None)
        if not callable(orig_model_lw):
            return orig_load_weights(model, *args, **kwargs)
        active = False

        def observed_model_lw(
            weights: Iterable[tuple[str, object]],
            *w_args: object,
            **w_kwargs: object,
        ) -> object:
            """Stream outer `weights` through `_observe_weights`."""
            nonlocal active
            if active:
                return orig_model_lw(weights, *w_args, **w_kwargs)
            active = True
            try:
                return orig_model_lw(
                    _observe_weights(weights, fn, options), *w_args, **w_kwargs
                )
            finally:
                active = False

        with _temporary_attr(model, "load_weights", observed_model_lw):
            return orig_load_weights(model, *args, **kwargs)

    return observed_loader_lw


@contextlib.contextmanager
def observe_model_loader(vllm_config: object, model_loader: object) -> Iterator[None]:
    """Show each streamed checkpoint tensor to `plugin.load_tensor` during load."""
    resolved = resolve_plugin_hook(vllm_config, "load_tensor")
    if resolved is None:
        yield
        return
    fn, options = resolved
    orig_get_all = getattr(model_loader, "get_all_weights", None)
    if callable(orig_get_all):

        def observed_get_all(
            *args: object, **kwargs: object
        ) -> Iterator[tuple[str, object]]:
            """Stream `get_all_weights` pairs through `_observe_weights`."""
            return _observe_weights(orig_get_all(*args, **kwargs), fn, options)

        with _temporary_attr(model_loader, "get_all_weights", observed_get_all):
            yield
        return
    orig_load_weights = getattr(model_loader, "load_weights", None)
    if callable(orig_load_weights):
        wrapped_lw = _wrap_load_weights(orig_load_weights, fn, options)
        with _temporary_attr(model_loader, "load_weights", wrapped_lw):
            yield
        return
    yield


__all__ = [
    "StepContext",
    "is_suppressed",
    "load_tensor_stream",
    "observe_model_loader",
    "suppressed",
]
