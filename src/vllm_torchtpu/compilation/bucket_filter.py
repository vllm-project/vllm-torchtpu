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
sequence-parallel backbone of a model takes the large prefill buckets). The
model gives :func:`assign` one routing function that returns the module for a
bucket, and each module's buckets are derived from it, so every bucket has
exactly one module. Each module then compiles only its own buckets, so the
modules never build the same bucket and the total number of TPU programs stays
one per bucket. The router enters the module's scope with :func:`serving`
around its call; compile happens inside that call on the first one. Outside
any such call (every other compiled module, the MTP drafter included) every
bucket is served. A shape-variant retrace resets only the module that owns the
refused bucket.

Why not one compiled module with a branch on the token count, partitioned by
its guards (``runnable_for`` in compilation/shape_variants.py): that partition
follows the warmup ladder, where a trace serves the run of ascending buckets
its guards accept and a structure change forces a retrace of the whole module.
When the route flips back and forth along the ladder (a bucket whose per-rank
shard is not row-aligned stays replicated between two sharded ones), every
flip costs one more trace of the whole backbone, and each bucket group needs
the hand-over between traces. With one module per route each module is traced
once when neither branches on the token count, and each module keeps its own
compile prefix (``self.prefix`` under ``_patch_vllm_compile_prefix_isolation``),
so the two graphs keep separate cache directories.

KV cache budget: ``profile_run`` compiles and runs only the largest bucket
(``TPUWorker.determine_available_memory`` calls it with ``max_num_tokens``),
and the KV budget is set from the HBM in use after that run. With routed
buckets that run measures only the module that serves the largest bucket; the
largest bucket of the other module is not included in the budget.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The compiled module of the routed call in progress.
_SERVING: ContextVar[object | None] = ContextVar(
    "vllm_torchtpu_bucket_filter", default=None
)

# Compiled module -> the routing function of its model (bucket -> module).
# Weak keys: a model owns its modules.
_ROUTES: weakref.WeakKeyDictionary[object, Callable[[int], object]] = (
    weakref.WeakKeyDictionary()
)


def assign(route: Callable[[int], object], modules: Sequence[object]) -> None:
    """Give each of ``modules`` the buckets for which ``route`` returns it.

    ``route(size)`` returns the compiled module that serves ``size`` tokens,
    one of ``modules``. Each module must be a torch.compile wrapper (a
    ``support_torch_compile`` module): shape_variants.retrace resets only the
    wrapper a refused bucket belongs to, and resetting the other one too would
    leave it untraced once the warmup ladder has moved past its buckets.
    """
    for module in modules:
        if not isinstance(module, TorchCompileWithNoGuardsWrapper):
            raise TypeError(
                "assign() needs compiled modules (TorchCompileWithNoGuardsWrapper), "
                f"got {type(module).__name__}"
            )
    for module in modules:
        _ROUTES[module] = route


def is_assigned(module: object) -> bool:
    """Whether :func:`assign` gave ``module`` its buckets."""
    return module in _ROUTES


def module_serves(module: object, size: int) -> bool:
    """Whether ``module`` serves ``size`` tokens; True if never assigned."""
    route = _ROUTES.get(module)
    return True if route is None else route(size) is module


@contextlib.contextmanager
def serving(module: object) -> Iterator[None]:
    """Scope in which the assigned compiled ``module`` is being called.

    The module's buckets come from the routing function given to
    :func:`assign`, so the compile filter, deferred compilation and retraces
    all key on the module.
    """
    if not is_assigned(module):
        raise ValueError(
            f"{type(module).__name__} was never given its buckets with assign()"
        )
    token = _SERVING.set(module)
    try:
        yield
    finally:
        _SERVING.reset(token)


def current() -> object | None:
    """The compiled module of the routed call in progress, or None outside one."""
    return _SERVING.get()


def serves(size: int) -> bool:
    """Whether the module being compiled serves ``size`` tokens."""
    module = _SERVING.get()
    return True if module is None else module_serves(module, size)


def patch_bucket_filter() -> None:
    """Compile only the buckets the module being built serves.

    Wraps whichever ``PiecewiseBackend.compile_all_ranges`` is installed, so
    it holds with or without TPU_PARALLEL_PRECOMPILE (registered after
    ``_patch_vllm_compile_all_ranges``). Outside a routed call every bucket is
    kept. ``load_all_ranges`` is left alone: it runs only for vLLM's mega AOT
    artifact, which env_override.py turns off; a warm start loads each bucket
    through ``compile_all_ranges``.
    """
    from vllm.compilation.piecewise_backend import PiecewiseBackend

    original = PiecewiseBackend.compile_all_ranges
    marker = "_tpu_bucket_filter_patch"
    if original.__dict__.get(marker, False):
        return

    def compile_all_ranges(self, *args, **kwargs):
        others = {
            r: e
            for r, e in self.range_entries.items()
            if r.is_single_size() and not serves(r.start)
        }
        order = list(self.range_entries)
        for r in others:
            del self.range_entries[r]
        try:
            return original(self, *args, **kwargs)
        finally:
            # Put the entries back in their original order.
            merged = {**self.range_entries, **others}
            self.range_entries.clear()
            self.range_entries.update((r, merged[r]) for r in order)

    compile_all_ranges.__dict__[marker] = True
    PiecewiseBackend.compile_all_ranges = compile_all_ranges
    logger.info("Applied TPU patch: per-module bucket filter.")
