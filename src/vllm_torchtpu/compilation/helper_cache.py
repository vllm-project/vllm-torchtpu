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

`helper_compile(name)` compiles through `TpuCompilerAdaptor` instead, keyed by
the graph and its input shapes, in a directory per run configuration:

    <VLLM_CACHE_ROOT>/tpu_helper_cache/<hash>/rank_<rank>_<dp_rank>/<name>/

so a start replays the artifacts an earlier start saved, and under
TPU_PRECOMPILE_PEER_REPLAY a rank replays a graph another rank of the host
already compiled. `<hash>` covers the TPU compilation hash (kernel sources,
compiler versions, TPU env), vLLM's config hash and vLLM's compile env; the key
covers the graph fingerprint (code and constants) and the input metadata.
Dynamo still traces each shape; lowering and XLA compile are what is skipped.

Until `configure` runs, and for a graph that cannot be fingerprinted, the
helpers compile with the plain "tpu" backend as before.
"""

import copy
import hashlib
import json
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Not a single size, so TpuCompilerAdaptor leaves shape_variants out of it.
_HELPER_RANGE_START, _HELPER_RANGE_END = 0, 1

_config: Any = None
_cache_root: str | None = None


def configure(vllm_config: Any) -> None:
    """Enable the cache for this process. The directory is fixed at the first
    helper compile, from the configuration at that point."""
    global _config, _cache_root
    _config = vllm_config
    _cache_root = None


def reset() -> None:
    """Disable the cache (tests)."""
    global _config, _cache_root
    _config = None
    _cache_root = None


def _root() -> str | None:
    """`<VLLM_CACHE_ROOT>/tpu_helper_cache/<hash>/rank_<r>_<dp>`, or None."""
    global _cache_root
    if _config is None:
        return None
    if _cache_root is None:
        import vllm.envs as vllm_envs
        from vllm.compilation.caching import aot_compile_hash_factors

        from vllm_torchtpu.compilation.tpu_compiler import (
            compute_tpu_compilation_hash,
        )

        # aot_compile_hash_factors is patched at platform activation to append
        # compute_tpu_compilation_hash and shape_variants.aot_cache_tag(); the
        # root is fixed on first helper compile (outside any shape-variant
        # trace), and compute_tpu_compilation_hash is appended explicitly so
        # unpatched processes still include it.
        factors = [
            *aot_compile_hash_factors(_config),
            compute_tpu_compilation_hash(_config),
        ]
        digest = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
        cc = getattr(_config, "compilation_config", None)
        base_cache_dir = getattr(cc, "cache_dir", None) or vllm_envs.VLLM_CACHE_ROOT
        pc = _config.parallel_config
        _cache_root = os.path.join(
            base_cache_dir,
            "tpu_helper_cache",
            digest,
            f"rank_{pc.rank}_{pc.data_parallel_rank}",
        )
    return _cache_root


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
        root = _root()
        if root is None:
            return _plain_backend(graph, example_inputs)

        from vllm.config.utils import Range

        from vllm_torchtpu.compilation.graph_fingerprint import _graph_fingerprint
        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilerAdaptor,
            _ensure_tuple_output,
        )

        # The adaptor fingerprints the graph after its output-tuple rewrite,
        # so the key does too.
        wrapped, was_wrapped = _ensure_tuple_output(copy.deepcopy(graph))
        fingerprint = _graph_fingerprint(wrapped)
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
        compile_range = Range(start=_HELPER_RANGE_START, end=_HELPER_RANGE_END)
        path = os.path.join(root, name, key)
        if os.path.exists(path):
            # Rebuilds the runnable from the artifact; recompiles and rewrites
            # it if the artifact does not fit.
            return adaptor.load(
                (key, path, was_wrapped), graph, example_inputs, 0, compile_range
            )
        compiled_fn, _ = adaptor.compile(
            graph, example_inputs, {}, compile_range, key=key
        )
        return compiled_fn

    compile_helper.__name__ = f"tpu_helper_{name}"
    return compile_helper


def helper_compile(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """`torch.compile(backend="tpu", fullgraph=True, dynamic=False)`, with the
    helper's graphs cached across starts and ranks (see the module docstring)."""
    return torch.compile(backend=backend(name), fullgraph=True, dynamic=False)
