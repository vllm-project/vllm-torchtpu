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
"""Persistent compile cache for the runner's helper graphs.

The runner's helpers (logits, sampling, logprobs, placeholder substitution) are
`torch.compile`d per input shape on first call. With the plain "tpu" backend
nothing persists them across starts but torch_tpu's XLA binary cache, so every
start traces, lowers and loads each graph again, and every rank compiles every
graph on a cold start.

When `additional_config={"helper_compile_cache": True}`, `helper_compile(name)`
compiles through `TpuCompilerAdaptor.compile_keyed` instead, keyed by the graph
and its input shapes, in a directory per run configuration:

    <VLLM_CACHE_ROOT>/tpu_helper_cache/<hash>/rank_<rank>_<dp_index>/<name>/

so a start replays the artifacts an earlier start saved, and under
TPU_PRECOMPILE_PEER_REPLAY a rank replays a graph another rank of the host
already compiled. `<hash>` covers the TPU compilation hash (kernel sources,
compiler versions, TPU env — excluding KV cache size), model/parallel/compile
configs; the key covers the graph fingerprint (code and constants) and input
metadata. Dynamo still traces each shape; lowering and XLA compile are skipped.

Until `configure` runs, and for a graph that cannot be fingerprinted, the
helpers compile with the plain "tpu" backend as before.
"""

import hashlib
import json
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
import vllm.envs as vllm_envs

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.utils import HELPER_COMPILE_CACHE_KEY

logger = init_logger(__name__)

_cache_root: str | None = None


def helper_compile_cache_enabled(vllm_config: Any) -> bool:
    """Return whether `additional_config.helper_compile_cache` is enabled."""
    extra = vllm_config.additional_config
    if isinstance(extra, dict) and HELPER_COMPILE_CACHE_KEY in extra:
        enabled = extra[HELPER_COMPILE_CACHE_KEY]
        if not isinstance(enabled, bool):
            raise ValueError(
                f"additional_config.{HELPER_COMPILE_CACHE_KEY} must be a bool"
            )
        return enabled
    return False


def configure(vllm_config: Any) -> None:
    """Enable the helper cache for this process and fix its root directory now.

    Called from `TPUWorker.compile_or_warm_up_model` after
    `update_block_size_for_backend` has finalized `mamba_block_size` and after
    `profile_run` (which stays on the plain backend), before `capture_model`
    mutates `compilation_config.cache_dir`.
    """
    global _cache_root
    from vllm_torchtpu.compilation.tpu_compiler import tpu_compilation_cache_subdir

    subdir = tpu_compilation_cache_subdir(
        vllm_config, "tpu_helper_cache", include_kv=False
    )
    pc = vllm_config.parallel_config
    _cache_root = os.path.join(subdir, f"rank_{pc.rank}_{pc.data_parallel_index}")


def configure_if_enabled(vllm_config: Any) -> bool:
    """Configure the helper cache when enabled and compile caching is on."""
    enabled = (
        helper_compile_cache_enabled(vllm_config)
        and not vllm_envs.VLLM_DISABLE_COMPILE_CACHE
    )
    if enabled:
        configure(vllm_config)
    return enabled


def reset() -> None:
    """Disable the cache and drop in-process runnables (tests)."""
    global _cache_root
    from vllm_torchtpu.compilation.tpu_compiler import _KEYED_RUNNABLES

    _cache_root = None
    _KEYED_RUNNABLES.clear()


def _input_meta(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            str(value.dtype),
            tuple(int(d) for d in value.shape),
            tuple(int(d) for d in value.stride()),
            int(value.storage_offset()),
        )
    if isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _plain_backend(graph: fx.GraphModule, example_inputs: list[Any]):
    return torch._dynamo.lookup_backend("tpu")(graph, example_inputs)


def backend(name: str) -> Callable[[fx.GraphModule, list[Any]], Any]:
    """A `torch.compile` backend that caches the helper `name`'s graphs."""

    def compile_helper(graph: fx.GraphModule, example_inputs: list[Any]):
        root = _cache_root
        if root is None:
            return _plain_backend(graph, example_inputs)

        from vllm_torchtpu.compilation.graph_fingerprint import _graph_fingerprint
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        fingerprint = _graph_fingerprint(graph)
        if fingerprint is None:
            return _plain_backend(graph, example_inputs)
        key = (
            "helper_"
            + hashlib.sha256(
                json.dumps(
                    [fingerprint, [_input_meta(v) for v in example_inputs]],
                    default=str,
                ).encode()
            ).hexdigest()[:24]
        )

        adaptor = TpuCompilerAdaptor()
        adaptor.initialize_cache(cache_dir=os.path.join(root, name))
        return adaptor.compile_keyed(graph, example_inputs, key)

    compile_helper.__name__ = f"tpu_helper_{name}"
    return compile_helper


def helper_compile(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """`torch.compile(backend="tpu", fullgraph=True, dynamic=False)`, with the
    helper's graphs cached across starts and ranks (see the module docstring)."""
    return torch.compile(backend=backend(name), fullgraph=True, dynamic=False)
