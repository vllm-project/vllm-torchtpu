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
"""Which token buckets the compiled module being built or loaded serves.

A model can route some token buckets to a second compiled module (the
sequence-parallel backbone of a model takes the large prefill buckets). Each
module then compiles, and loads from the cache, only its own buckets, so the
two never build the same bucket and the total number of TPU programs stays one
per bucket. The router enters each module's scope around its call; compile
and load happen inside that call on the first one. Outside any such call
(every other compiled module, the MTP drafter included) every bucket is served.
The router first assigns each module its predicate, so a shape-variant retrace
resets only the module that owns the refused bucket.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable, Iterator
from contextvars import ContextVar

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

# (compiled module, its predicate) of the routed call in progress.
_SERVES: ContextVar[tuple[object, Callable[[int], bool]] | None] = ContextVar(
    "vllm_torchtpu_bucket_filter", default=None
)


@contextlib.contextmanager
def serving(module: object) -> Iterator[None]:
    """Scope in which the assigned compiled ``module`` is being called.

    The module's buckets are the predicate given to :func:`assign`, so the
    compile filter, deferred compilation and retraces all key on the module.
    """
    predicate = assigned(module)
    if predicate is None:
        raise ValueError(
            f"{type(module).__name__} was never given its buckets with assign()"
        )
    token = _SERVES.set((module, predicate))
    try:
        yield
    finally:
        _SERVES.reset(token)


def current() -> object | None:
    """The compiled module of the routed call in progress, or None outside one."""
    routed = _SERVES.get()
    return None if routed is None else routed[0]


def serves(size: int) -> bool:
    """Whether the module being compiled or loaded serves ``size`` tokens."""
    routed = _SERVES.get()
    return True if routed is None else bool(routed[1](size))


# Compiled module -> the buckets it serves. Weak keys: a model owns its modules.
_ASSIGNED: weakref.WeakKeyDictionary[object, Callable[[int], bool]] = (
    weakref.WeakKeyDictionary()
)


def assign(module: object, predicate: Callable[[int], bool]) -> None:
    """Record that the compiled ``module`` serves only ``predicate`` buckets.

    ``module`` must be a torch.compile wrapper (a ``support_torch_compile``
    module): shape_variants.retrace resets only the wrapper a refused bucket
    belongs to, and resetting the other one too would leave it untraced once
    the warmup ladder has moved past its buckets.
    """
    if not isinstance(module, TorchCompileWithNoGuardsWrapper):
        raise TypeError(
            f"assign() needs a compiled module (TorchCompileWithNoGuardsWrapper), "
            f"got {type(module).__name__}"
        )
    _ASSIGNED[module] = predicate


def assigned(module: object) -> Callable[[int], bool] | None:
    """The predicate given to :func:`assign`, if any."""
    return _ASSIGNED.get(module)


def module_serves(module: object, size: int) -> bool:
    """Whether ``module`` serves ``size`` tokens; True if never assigned."""
    predicate = assigned(module)
    return True if predicate is None else bool(predicate(size))
