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

Kept strictly stdlib-only at import (`R-2.1`) so the runner can import these
helpers at module load without pulling in `jax`, `torch`, `vllm_torchtpu.envs`,
or `megakernel.core`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
from typing import TYPE_CHECKING, Any

from .constants import HOOK_LOAD_TENSOR, SUPPRESS_ATTR

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Iterable, Iterator, Mapping


@dataclasses.dataclass(frozen=True, slots=True)
class StepContext:
    """Inputs of one runner forward step, matching the stock model signature.

    `model_kwargs` holds extra keyword arguments passed to the stock model's
    forward; a megakernel that cannot consume them declines the step. A
    megakernel that serves the step returns hidden states or
    `(hidden_states, aux_hidden_states)`, normalized identically by the runner.
    """

    input_ids: Any
    positions: Any
    inputs_embeds: Any = None
    intermediate_tensors: Any = None
    model_kwargs: Mapping[str, Any] | None = None


class _SuppressionScope:
    """Context manager tracking re-entrant runner megakernel suppression depth."""

    def __init__(self, runner: object) -> None:
        """Initializes the instance."""
        self._runner = runner
        self._prev: int = 0

    def __enter__(self) -> None:
        """Increment runner suppression depth."""
        runner_dict = getattr(self._runner, "__dict__", None)
        if isinstance(runner_dict, dict):
            prev = runner_dict.get(SUPPRESS_ATTR, 0)
            self._prev = prev if isinstance(prev, int) else 0
            runner_dict[SUPPRESS_ATTR] = self._prev + 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Restore runner suppression depth."""
        runner_dict = getattr(self._runner, "__dict__", None)
        if isinstance(runner_dict, dict):
            if self._prev > 0:
                runner_dict[SUPPRESS_ATTR] = self._prev
            else:
                runner_dict.pop(SUPPRESS_ATTR, None)


def suppressed(runner: object) -> _SuppressionScope:
    """Force the stock model forward for every step of `runner` in this scope.

    Callers (such as warmup, precompile, or profiling forwards) wrap stock
    passes in this scope so every stock graph is compiled at startup even when
    the megakernel could serve that bucket. Re-entrant; `runner` may be any
    object with an instance `__dict__`.

    Args:
        runner: Model runner instance to suppress megakernel dispatch on.

    Returns:
        Re-entrant context manager restoring the prior suppression depth on exit.
    """
    return _SuppressionScope(runner)


def _suppress_depth(runner: object) -> int:
    """Read suppression depth directly from `runner`'s instance dictionary."""
    runner_dict = getattr(runner, "__dict__", None)
    if isinstance(runner_dict, dict):
        depth = runner_dict.get(SUPPRESS_ATTR, 0)
        return depth if isinstance(depth, int) else 0
    return 0


def is_suppressed(runner: object) -> bool:
    """Return whether `runner` is currently inside a `suppressed(runner)` scope."""
    return runner is not None and _suppress_depth(runner) > 0


def _resolve_load_tensor_hook(
    vllm_config: object,
) -> tuple[Callable[..., object], dict[str, object]] | None:
    """Lazily resolve `TpuMonoSpec.load_tensor` while keeping module import pure."""
    # Deferred import preserves R-2.1 stdlib-only module load on hooks.py.
    salt_mod = importlib.import_module(".salt", __package__)
    return salt_mod.resolve_plugin_hook(vllm_config, HOOK_LOAD_TENSOR)


class _InstanceMethodPatch:
    """Reversible instance-attribute patch restoring class or instance methods."""

    def __init__(self, target: object, attr_name: str) -> None:
        """Initializes the instance."""
        self._target = target
        self._attr_name = attr_name
        inst_dict = getattr(target, "__dict__", None)
        self._had_own = isinstance(inst_dict, dict) and attr_name in inst_dict
        self._prev = getattr(target, attr_name, None)
        self._active = False

    def install(self, replacement: Callable[..., object]) -> None:
        """Install `replacement` on `target`."""
        setattr(self._target, self._attr_name, replacement)
        self._active = True

    def restore(self) -> None:
        """Idempotently restore `target` to its pre-patch attribute state."""
        if not self._active:
            return
        self._active = False
        if self._had_own:
            setattr(self._target, self._attr_name, self._prev)
            return
        inst_dict = getattr(self._target, "__dict__", None)
        if isinstance(inst_dict, dict) and self._attr_name in inst_dict:
            delattr(self._target, self._attr_name)


def _observe_pair[TensorT](
    pair: tuple[str, TensorT],
    *,
    fn: Callable[..., object],
    options: Mapping[str, object],
) -> tuple[str, TensorT]:
    """Invoke `fn(name, tensor, options)` and forward `(name, tensor)` unchanged."""
    name, tensor = pair
    fn(name, tensor, options)
    return name, tensor


def _observe_weights[TensorT](
    weights: Iterable[tuple[str, TensorT]],
    fn: Callable[..., object],
    options: Mapping[str, object],
) -> Iterator[tuple[str, TensorT]]:
    """Map `weights` lazily through `fn` while yielding unchanged pairs."""
    return (_observe_pair(pair, fn=fn, options=options) for pair in weights)


def load_tensor_stream[TensorT](
    vllm_config: object,
    weights: Iterable[tuple[str, TensorT]],
) -> Iterator[tuple[str, TensorT]]:
    """Stream `weights` through `TpuMonoSpec.load_tensor` when enabled.

    Args:
        vllm_config: Active vLLM configuration used to resolve the plugin hook.
        weights: Iterable of `(name, tensor)` checkpoint pairs.

    Returns:
        Iterator yielding the unchanged `(name, tensor)` pairs after observation,
        or `iter(weights)` directly when no `load_tensor` hook is active.
    """
    resolved = _resolve_load_tensor_hook(vllm_config)
    if resolved is None:
        return iter(weights)
    fn, options = resolved
    return _observe_weights(weights, fn, options)


class _ModelLoadWeightsInterceptor:
    """Single-entry `model.load_weights` wrapper preventing nested double-reads."""

    def __init__(
        self,
        patch: _InstanceMethodPatch,
        orig_load_weights: Callable[..., object],
        fn: Callable[..., object],
        options: Mapping[str, object],
    ) -> None:
        """Initializes the instance."""
        self._patch = patch
        self._orig_load_weights = orig_load_weights
        self._fn = fn
        self._options = options

    def __call__(
        self,
        weights: Iterable[tuple[str, object]],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Restore `model.load_weights` immediately and stream observed pairs."""
        self._patch.restore()
        observed = _observe_weights(weights, self._fn, self._options)
        return self._orig_load_weights(observed, *args, **kwargs)


class _ModelLoaderObserver:
    """Context manager intercepting `model_loader.load_weights` during load."""

    def __init__(
        self,
        loader: object,
        orig_loader_lw: Callable[..., object],
        fn: Callable[..., object],
        options: Mapping[str, object],
    ) -> None:
        """Initializes the instance."""
        self._patch = _InstanceMethodPatch(loader, "load_weights")
        self._orig_loader_lw = orig_loader_lw
        self._fn = fn
        self._options = options

    def _observed_loader_load_weights(
        self,
        model: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Temporarily wrap `model.load_weights` for one loader invocation."""
        orig_model_lw = getattr(model, "load_weights", None)
        if not callable(orig_model_lw):
            return self._orig_loader_lw(model, *args, **kwargs)
        model_patch = _InstanceMethodPatch(model, "load_weights")
        interceptor = _ModelLoadWeightsInterceptor(
            model_patch,
            orig_model_lw,
            self._fn,
            self._options,
        )
        model_patch.install(interceptor)
        try:
            return self._orig_loader_lw(model, *args, **kwargs)
        finally:
            model_patch.restore()

    def __enter__(self) -> None:
        """Install temporary `load_weights` observer on `loader`."""
        self._patch.install(self._observed_loader_load_weights)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Restore original `loader.load_weights` attribute state."""
        self._patch.restore()


def observe_model_loader(
    vllm_config: object,
    model_loader: object,
) -> contextlib.AbstractContextManager[None]:
    """Observe streamed checkpoint tensors via `TpuMonoSpec.load_tensor`.

    Wraps `model_loader.load_weights` so every `(name, tensor)` pair streamed
    into `model.load_weights` (covering `DefaultModelLoader` and
    `RunaiModelStreamerLoader`) is shown once to `load_tensor` before
    `process_weights_after_loading` repacks weights in place.

    Args:
        vllm_config: Active vLLM configuration used to resolve the plugin hook.
        model_loader: Model loader instance providing `load_weights`.

    Returns:
        Context manager installing the observer when a `load_tensor` hook is
        configured, or a zero-overhead `contextlib.nullcontext()` otherwise.
    """
    resolved = _resolve_load_tensor_hook(vllm_config)
    if resolved is None:
        return contextlib.nullcontext()
    orig_loader_lw = getattr(model_loader, "load_weights", None)
    if not callable(orig_loader_lw):
        return contextlib.nullcontext()
    fn, options = resolved
    return _ModelLoaderObserver(model_loader, orig_loader_lw, fn, options)


__all__ = [
    "StepContext",
    "is_suppressed",
    "load_tensor_stream",
    "observe_model_loader",
    "suppressed",
]
