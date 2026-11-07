# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/compilation.py
"""Compile independent torch_tpu graphs before their ordered warmup execution."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_MAX_PENDING_COMPILES = 8
_POLL_INTERVAL_SECONDS = 0.01


class _CompileHandle(Protocol):
    """Protocol for an async ``torch_tpu`` compilation handle."""

    def is_ready(self) -> bool:
        """Return ``True`` when the async compilation has completed."""
        ...

    def get(self) -> object:
        """Materialize and raise any compilation error from the handle."""
        ...


@dataclass(frozen=True)
class CompileCall:
    """One tuple-returning op and the exact eager compiler options it uses."""

    op: Callable[..., tuple[object, ...]]
    operands: tuple[object, ...]
    options: dict[str, str]
    priority: int = 0  # Larger independent graphs are submitted first.


def _pop_ready_compile(
    pending: list[tuple[_CompileHandle, tuple[object, ...], tuple[object, ...]]],
) -> None:
    """Wait until any pending compile handle is ready, pop it, and materialize it.

    The first submitted graph may be the slowest; reusing any completed slot
    keeps it from holding back the rest of the independent schedule.

    Args:
        pending: In-flight ``(handle, inputs, outputs)`` compilation entries.
    """
    while True:
        for index, (handle, _inputs, _outputs) in enumerate(pending):
            if handle.is_ready():
                entry = pending.pop(index)
                entry[0].get()
                return
        time.sleep(_POLL_INTERVAL_SECONDS)


def _drain_pending_compiles(
    pending: list[tuple[_CompileHandle, tuple[object, ...], tuple[object, ...]]],
) -> Exception | None:
    """Drain all remaining in-flight compiles and return the first error, if any.

    Args:
        pending: In-flight ``(handle, inputs, outputs)`` compilation entries.

    Returns:
        First ``Exception`` raised while draining handles, or ``None``.
    """
    drain_error: Exception | None = None
    while pending:
        try:
            _pop_ready_compile(pending)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            LookupError,
            AssertionError,
        ) as exc:
            if drain_error is None:
                drain_error = exc
    return drain_error


def compile_ops(calls: Iterable[CompileCall]) -> None:
    """Populate the eager executable cache with at most eight pending compiles.

    Graphs are built on the caller's thread from fresh placeholders, so
    compiling cannot consume donated buffers or run collectives. This does not
    replace the warmup, which executes the graphs and the programs left out of
    this schedule. The bound also caps compiler memory on hosts shared by eight
    workers. Submitted work drains before a failure propagates.

    Args:
        calls: Iterable of :class:`CompileCall` specifications to compile.
    """
    execution_mode = importlib.import_module("torch_tpu._internal.execution_mode")
    tpu_torch_compile = importlib.import_module(
        "torch_tpu._internal.compile.tpu_torch_compile"
    )
    compiler_options = importlib.import_module("torch_tpu._internal.compiler_options")
    custom_compiler_options = compiler_options.custom_compiler_options

    pending: list[tuple[_CompileHandle, tuple[object, ...], tuple[object, ...]]] = []
    eager_members = execution_mode.EagerMode.__members__
    eager_mode = (
        eager_members.get("INTERNAL_COMPILE_FX_GRAPH")
        or eager_members["_INTERNAL_COMPILE_FX_GRAPH"]
    )
    body_succeeded = False
    try:
        for call in calls:
            if len(pending) == _MAX_PENDING_COMPILES:
                _pop_ready_compile(pending)
            inputs = tuple(
                tpu_torch_compile.placeholder_like(operand) for operand in call.operands
            )
            with execution_mode.set_eager_mode(eager_mode):
                outputs = call.op(*inputs)
            with custom_compiler_options(call.options):
                handle = tpu_torch_compile.async_traverse_and_compile(
                    list(outputs), list(inputs), fast_compile=True
                )
            pending.append((handle, inputs, outputs))
        body_succeeded = True
    finally:
        drain_error = _drain_pending_compiles(pending)
    if body_succeeded and drain_error is not None:
        raise drain_error
