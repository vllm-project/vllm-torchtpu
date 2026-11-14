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
this file and its callers in `TpuCompilerAdaptor`.

TODO: Remove this module once upstream `torch_tpu` / `libtpu` drops
compile-only metadata natively for every `backend="tpu"` graph (including
standalone `@torch.compile(backend="tpu")` helpers that bypass
`TpuCompilerAdaptor`), or provides `TorchTpuCompiledExecutable.release()`
together with a per-entry Tier-1 cache eviction API, so the process-wide
`torch.tpu._clear_cache()` below is no longer needed.
"""

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


def get_num_compilations() -> int:
    """Return torch_tpu's cumulative CompileAndLoad count."""
    # pyrefly: ignore[missing-attribute]
    return int(torch.tpu._get_cache_stats().num_compilations)


def clear_tpu_tier1_cache() -> None:
    """Evict every entry of torch_tpu's in-memory Tier-1 CompilationCache.

    TODO: Replace with a per-entry Tier-1 eviction API (by cache key or
    executable) once `torch_tpu` provides one. This process-wide clear also
    evicts earlier eager (`DEFER_AND_FUSE`) programs, which then reload from
    the Tier-2/Tier-3 caches, or recompile when both are disabled.
    """
    if envs.VLLM_XLA_CHECK_RECOMPILATION:
        logger.warning_once(
            "[TpuCompilerAdaptor] VLLM_TPU_RELOAD_COMPILED_EXECUTABLE clears "
            "torch_tpu's Tier-1 CompilationCache (per_entry_stats), which "
            "resets VLLM_XLA_CHECK_RECOMPILATION's cached-graph counter; "
            "inspect TpuCompilerAdaptor reload logs or stats.num_compilations "
            "instead."
        )
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
