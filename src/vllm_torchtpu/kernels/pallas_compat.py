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
"""Pallas TPU shims for DMA scheduling hints only an internal JAX build has.

Some kernels are tuned against an internal Pallas build whose DMA APIs take
scheduling hints that no public JAX release accepts:

* ``pltpu.make_async_copy(..., priority=, strict_ordering=, wait_threshold=)``
  and ``pltpu.make_async_remote_copy(..., priority=, strict_ordering=)``;
* ``<descriptor>.start(strict_ordering=)`` and
  ``<descriptor>.wait(strict_ordering=, wait_threshold=)``;
* ``pltpu.emit_pipeline(..., priority=, strict_ordering=)``;
* ``pltpu.BufferedRef.create(..., priority=, strict_ordering=)`` and the
  ``priority`` / ``strict_ordering`` fields of ``BufferedRef``;
* ``pltpu.repeat``.

On public JAX each of those raises ``TypeError`` / ``AttributeError`` while
tracing. The wrappers here take the internal signatures and forward a hint
only when the installed Pallas accepts it; support is read once from the
installed signatures, never by catching errors. Dropping a hint keeps the
results: public Pallas lowers every DMA with ``strict_ordering=False`` and the
DMAs its own pipelines issue at priority 0, which is what the internal build
does with the hints at their defaults. ``priority`` on ``start`` is public, so
a priority given to a descriptor is still honoured when it is started.

``wait_threshold`` is not a hint (the internal build turns such a wait into a
plain semaphore wait for that value), so a non-``None`` threshold raises
``NotImplementedError`` on JAX that lacks it instead of waiting for something
else.

``AsyncCopyDescriptor.start(when=)`` is this module's own addition (neither
build has it): it issues the DMA under ``pl.when(when)``.

Importing this module only inspects signatures and defines the classes below;
it neither traces nor touches a device.
"""

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _keyword_params(fn: Any) -> frozenset[str]:
    """The parameter names `fn` accepts by keyword (a `**kwargs` catch-all
    does not count: it may forward to something that rejects the name)."""
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(p.name for p in parameters
                     if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))


def _accepted(hints: dict[str, Any], params: frozenset[str]) -> dict[str, Any]:
    return {name: value for name, value in hints.items() if name in params}


def _or(value: Any, default: Any) -> Any:
    return default if value is None else value


_MAKE_ASYNC_COPY_PARAMS = _keyword_params(pltpu.make_async_copy)
_MAKE_ASYNC_REMOTE_COPY_PARAMS = _keyword_params(pltpu.make_async_remote_copy)
_EMIT_PIPELINE_PARAMS = _keyword_params(pltpu.emit_pipeline)
_BUFFERED_REF_CREATE_PARAMS = _keyword_params(pltpu.BufferedRef.create)
# Read from the module dict so that a deprecation hook in `pltpu.__getattr__`
# can never fire (and warn) at import.
_NATIVE_REPEAT: Callable[..., jax.Array] | None = vars(pltpu).get("repeat")

# What the installed Pallas supports. All False on public JAX (checked on
# 0.10.2 and 0.11.1); all True on the internal build.
#: `make_async_copy` takes `priority` (the default for `start`).
HAS_DMA_PRIORITY = "priority" in _MAKE_ASYNC_COPY_PARAMS
#: Local DMAs take `strict_ordering` (descriptor, `start` and `wait`).
HAS_DMA_STRICT_ORDERING = "strict_ordering" in _MAKE_ASYNC_COPY_PARAMS
#: Local DMA waits take `wait_threshold`.
HAS_DMA_WAIT_THRESHOLD = "wait_threshold" in _MAKE_ASYNC_COPY_PARAMS
#: `make_async_remote_copy` takes `priority` and `strict_ordering`.
HAS_REMOTE_DMA_STRICT_ORDERING = ("strict_ordering"
                                  in _MAKE_ASYNC_REMOTE_COPY_PARAMS)
#: `emit_pipeline` takes `priority` (and `strict_ordering`).
HAS_EMIT_PIPELINE_PRIORITY = "priority" in _EMIT_PIPELINE_PARAMS
#: `BufferedRef.create` takes `priority` / `strict_ordering`, and the
#: pipeline's own DMAs use them.
HAS_BUFFERED_REF_PRIORITY = "priority" in _BUFFERED_REF_CREATE_PARAMS
#: `pltpu.repeat` exists.
HAS_REPEAT = _NATIVE_REPEAT is not None


@functools.cache
def _method_params(cls: type, method: str) -> frozenset[str]:
    return _keyword_params(getattr(cls, method, None))


class AsyncCopyDescriptor:
    """A Pallas DMA descriptor whose `start` / `wait` take the internal hints.

    Wraps the descriptor the installed `pltpu.make_async_copy` (or
    `make_async_remote_copy`) returned, kept as `.descriptor`; any other
    attribute is read from it. A hint left as `None` in a call falls back to
    the value given when the descriptor was made, as in the internal build.
    """

    __slots__ = ("descriptor", "priority", "strict_ordering", "wait_threshold")

    def __init__(self,
                 descriptor: Any,
                 *,
                 priority: int = 0,
                 strict_ordering: bool = False,
                 wait_threshold: int | None = None):
        self.descriptor = descriptor
        self.priority = priority
        self.strict_ordering = strict_ordering
        self.wait_threshold = wait_threshold

    def __getattr__(self, name: str) -> Any:
        if name in AsyncCopyDescriptor.__slots__:  # not set yet
            raise AttributeError(name)
        return getattr(self.descriptor, name)

    def __repr__(self) -> str:
        return (f"pallas_compat.AsyncCopyDescriptor({self.descriptor!r}, "
                f"priority={self.priority}, "
                f"strict_ordering={self.strict_ordering}, "
                f"wait_threshold={self.wait_threshold})")

    def start(self,
              priority: int | None = None,
              *,
              add: bool = False,
              strict_ordering: bool | None = None,
              when: Any = None) -> None:
        """Starts the DMA.

        `priority` is forwarded wherever `start` takes it (public JAX does);
        `strict_ordering` only on the internal build. `add` changes the
        result, so it is always passed through when set. `when`, if given,
        issues the DMA only where it holds (`pl.when` semantics).
        """
        hints = {
            "priority": _or(priority, self.priority),
            "strict_ordering": _or(strict_ordering, self.strict_ordering),
        }
        params = _method_params(type(self.descriptor), "start")
        kwargs = _accepted(hints, params)
        if add:
            kwargs["add"] = True
        if when is None:
            self.descriptor.start(**kwargs)
        else:
            pl.when(when)(lambda: self.descriptor.start(**kwargs))

    def wait(self,
             *,
             strict_ordering: bool | None = None,
             wait_threshold: int | None = None) -> None:
        """Waits for the DMA (read and write, like `descriptor.wait()`)."""
        self._wait("wait", strict_ordering, wait_threshold)

    def wait_send(self,
                  *,
                  strict_ordering: bool | None = None,
                  wait_threshold: int | None = None) -> None:
        """Waits for the read side of a remote DMA."""
        self._wait("wait_send", strict_ordering, wait_threshold)

    def wait_recv(self,
                  *,
                  strict_ordering: bool | None = None,
                  wait_threshold: int | None = None) -> None:
        """Waits for the write side of a remote DMA."""
        self._wait("wait_recv", strict_ordering, wait_threshold)

    def _wait(self, method: str, strict_ordering: bool | None,
              wait_threshold: int | None) -> None:
        params = _method_params(type(self.descriptor), method)
        kwargs = _accepted(
            {"strict_ordering": _or(strict_ordering, self.strict_ordering)},
            params)
        wait_threshold = _or(wait_threshold, self.wait_threshold)
        if wait_threshold is not None:
            if "wait_threshold" not in params:
                raise NotImplementedError(
                    f"{method}(wait_threshold={wait_threshold}) needs a Pallas "
                    "with DMA wait thresholds; the installed JAX "
                    f"{jax.__version__} has none, and waiting on the "
                    "descriptor instead would not be equivalent.")
            kwargs["wait_threshold"] = wait_threshold
        getattr(self.descriptor, method)(**kwargs)


def make_async_copy(src_ref: Any,
                    dst_ref: Any,
                    sem: Any,
                    *,
                    priority: int = 0,
                    strict_ordering: bool = False,
                    wait_threshold: int | None = None) -> AsyncCopyDescriptor:
    """`pltpu.make_async_copy` with the internal build's hint arguments."""
    hints = {
        "priority": priority,
        "strict_ordering": strict_ordering,
        "wait_threshold": wait_threshold,
    }
    descriptor = pltpu.make_async_copy(
        src_ref, dst_ref, sem, **_accepted(hints, _MAKE_ASYNC_COPY_PARAMS))
    return AsyncCopyDescriptor(descriptor, **hints)


def make_async_remote_copy(
        src_ref: Any,
        dst_ref: Any,
        send_sem: Any,
        recv_sem: Any,
        device_id: Any,
        device_id_type: Any = None,
        *,
        priority: int = 0,
        strict_ordering: bool = False) -> AsyncCopyDescriptor:
    """`pltpu.make_async_remote_copy` with the internal build's hint
    arguments. `device_id_type=None` keeps the installed default (MESH)."""
    hints = {"priority": priority, "strict_ordering": strict_ordering}
    kwargs = _accepted(hints, _MAKE_ASYNC_REMOTE_COPY_PARAMS)
    if device_id_type is not None:
        kwargs["device_id_type"] = device_id_type
    descriptor = pltpu.make_async_remote_copy(src_ref, dst_ref, send_sem,
                                              recv_sem, device_id, **kwargs)
    return AsyncCopyDescriptor(descriptor, **hints)


def emit_pipeline(body: Callable[..., Any],
                  *,
                  priority: int = 0,
                  strict_ordering: bool = False,
                  **kwargs: Any) -> Any:
    """`pltpu.emit_pipeline` taking the internal build's `priority` and
    `strict_ordering` for the DMAs the pipeline issues itself."""
    hints = {"priority": priority, "strict_ordering": strict_ordering}
    return pltpu.emit_pipeline(body, **kwargs,
                               **_accepted(hints, _EMIT_PIPELINE_PARAMS))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRef(pltpu.BufferedRef):
    """`pltpu.BufferedRef` with the internal build's DMA hint fields.

    Subclass this instead of `pltpu.BufferedRef` and build through its
    `create` / `input` / `output` / `input_output`, which take `priority` and
    `strict_ordering`. Overridden copy/wait methods read `self.priority` /
    `self.strict_ordering` and pass them to this module's descriptors. On
    public JAX the inherited copy/wait methods ignore both (as public Pallas
    always does); on the internal build they are forwarded to `create` and
    used there too.
    """

    priority: int = dataclasses.field(default=0, metadata=dict(static=True))
    strict_ordering: bool = dataclasses.field(default=False,
                                              metadata=dict(static=True))

    @classmethod
    def create(cls,
               *args: Any,
               priority: int = 0,
               strict_ordering: bool = False,
               **kwargs: Any) -> Any:
        """`pltpu.BufferedRef.create` (same arguments) plus the hints."""
        hints = {"priority": priority, "strict_ordering": strict_ordering}
        forwarded = _accepted(hints, _BUFFERED_REF_CREATE_PARAMS)
        ref = super().create(*args, **kwargs, **forwarded)
        if len(forwarded) < len(hints):
            ref = dataclasses.replace(ref, **hints)
        return ref


def repeat(x: jax.Array, repeats: int, axis: int) -> jax.Array:
    """`pltpu.repeat`: `x` tiled `repeats` times along `axis`.

    Note this is `jnp.tile` along one axis (``[a, b] -> [a, b, a, b]``), not
    `jnp.repeat`. Without `pltpu.repeat` it is computed with exactly the
    internal definition.
    """
    if _NATIVE_REPEAT is not None:
        return _NATIVE_REPEAT(x, repeats, axis)
    ndim = jnp.ndim(x)
    if not -ndim <= axis < ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {ndim}")
    axis %= ndim
    return jnp.tile(x, [repeats if i == axis else 1 for i in range(ndim)])
