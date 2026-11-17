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
"""torch_tpu private-structure helpers for VLLM_TPU_RELOAD_COMPILED_EXECUTABLE.

Everything here encodes `torch_tpu==0.2.0` internals (`_SplitCompiledExecutable`,
`TorchTpuCompiledExecutable._fast_runner` / `_executable`,
`torch.tpu._clear_cache()`), so the reload workaround is removed by deleting
this file, `tests/compilation/test_executable_reload.py`, and its callers in
`TpuCompilerAdaptor` and `TpuPlatform`.

TODO:
1. On the next `torch_tpu` bump (`> 0.2.0`): `CompiledArtifact` exposes
   `compiled_executables` on the base class and `TorchTpuCompiledExecutable`
   provides `release()`, which clears `_executable` and `_fast_runner`. Update
   `release_entry_executables` to iterate `artifact.compiled_executables` and
   call `leaf.release()` on each `TorchTpuCompiledExecutable` leaf (still
   skipping `NoOpCompiledArtifact`, which has no `release()`), and drop the
   `_SplitCompiledExecutable` import.
2. Once `torch_tpu` exposes single-entry eviction on its Tier-1
   `CompilationCache`, replace the process-wide `torch.tpu._clear_cache()`
   call in `clear_tpu_tier1_cache`.
3. Remove this module altogether once upstream `torch_tpu` / `libtpu` drops
   compile-only host metadata natively after `CompileAndLoad` for every
   `backend="tpu"` graph.
"""

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any

import torch
from torch_tpu._internal.compile.split_compiler import _SplitCompiledExecutable
from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
    NoOpCompiledArtifact,
    TorchTpuCompiledExecutable,
)

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class CompilationTracker:
    """Records whether the wrapped backend call ran `CompileAndLoad`."""

    compiled_now: bool = False


def get_num_compilations() -> int:
    """Return torch_tpu's cumulative CompileAndLoad count."""
    # pyrefly: ignore[missing-attribute]
    return int(torch.tpu._get_cache_stats().num_compilations)


@contextlib.contextmanager
def track_compilation() -> Iterator[CompilationTracker]:
    """Track whether a synchronous backend call incremented `num_compilations`.

    No-op (`compiled_now=False`) when `VLLM_TPU_RELOAD_COMPILED_EXECUTABLE` is
    disabled. Assumes synchronous compilation (`async_compile=False`, which
    `vllm-torchtpu` always uses).
    """
    tracker = CompilationTracker()
    if not envs.VLLM_TPU_RELOAD_COMPILED_EXECUTABLE:
        yield tracker
        return
    before = get_num_compilations()
    yield tracker
    after = get_num_compilations()
    tracker.compiled_now = after > before
    if tracker.compiled_now:
        logger.info(
            "[TpuCompilerAdaptor] CompileAndLoad ran (compilations %d -> %d)",
            before,
            after,
        )


def clear_tpu_tier1_cache() -> None:
    """Evict every entry of torch_tpu's in-memory Tier-1 CompilationCache."""
    # pyrefly: ignore[missing-attribute]
    torch.tpu._clear_cache()


def release_entry_executables(entry: Any) -> None:
    """Drop the PjRtLoadedExecutable refs of every leaf inside a bundled entry.

    Validates the whole forward/backward artifact tree before clearing any
    leaf, so an unsupported `CompiledArtifact` type raises `TypeError` without
    leaving the entry half released.
    """
    leaves: list[TorchTpuCompiledExecutable] = []
    seen: set[int] = set()

    def collect(artifact: Any) -> None:
        if artifact is None or id(artifact) in seen:
            return
        seen.add(id(artifact))
        if isinstance(artifact, _SplitCompiledExecutable):
            for sub in artifact.compiled_executables:
                collect(sub)
        elif isinstance(artifact, TorchTpuCompiledExecutable):
            leaves.append(artifact)
        elif not isinstance(artifact, NoOpCompiledArtifact):
            raise TypeError(
                f"Unsupported CompiledArtifact type {type(artifact).__qualname__!r} "
                "for VLLM_TPU_RELOAD_COMPILED_EXECUTABLE."
            )

    for loadable in (entry.compiled_fw, entry.compiled_bw):
        if loadable is not None:
            collect(loadable.result)
    for leaf in leaves:
        leaf._fast_runner = None
        leaf._executable = None
