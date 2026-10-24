# SPDX-License-Identifier: Apache-2.0
"""Which token buckets the compiled module being built or loaded serves.

A model can route some token buckets to a second compiled module (the
qwen4_exp sequence-parallel backbone takes the large prefill buckets). Each
module then compiles, and loads from the cache, only its own buckets, so the
two never build the same bucket and the total number of TPU programs stays one
per bucket. The router sets the predicate around each module's call; compile
and load happen inside that call on the first one. Outside any such call
(every other compiled module, the MTP drafter included) every bucket is served.
The router also assigns each module its predicate, so a shape-variant retrace
resets only the module that owns the refused bucket.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar

_SERVES: ContextVar[Callable[[int], bool] | None] = ContextVar(
    "vllm_torchtpu_bucket_filter", default=None
)


@contextlib.contextmanager
def serving(predicate: Callable[[int], bool]) -> Iterator[None]:
    """Scope in which the module being called serves ``predicate`` buckets."""
    token = _SERVES.set(predicate)
    try:
        yield
    finally:
        _SERVES.reset(token)


def serves(size: int) -> bool:
    """Whether the module being compiled or loaded serves ``size`` tokens."""
    predicate = _SERVES.get()
    return True if predicate is None else bool(predicate(size))


_ASSIGNED = "_vllm_torchtpu_serves_buckets"


def assign(module: object, predicate: Callable[[int], bool]) -> None:
    """Record that the compiled ``module`` serves only ``predicate`` buckets.

    shape_variants.retrace reads it to reset only the module a refused bucket
    belongs to: resetting the other one too would leave it untraced once the
    warmup ladder has moved past its buckets.
    """
    object.__setattr__(module, _ASSIGNED, predicate)


def assigned(module: object) -> Callable[[int], bool] | None:
    """The predicate given to :func:`assign`, if any."""
    return getattr(module, _ASSIGNED, None)


def module_serves(module: object, size: int) -> bool:
    """Whether ``module`` serves ``size`` tokens; True if never assigned."""
    predicate = assigned(module)
    return True if predicate is None else bool(predicate(size))
