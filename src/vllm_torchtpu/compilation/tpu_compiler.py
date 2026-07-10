# SPDX-License-Identifier: Apache-2.0
"""TPU compiler adaptor for vLLM's VLLM_COMPILE mode.

This module implements the CompilerInterface for TPU, enabling
PiecewiseBackend to compile FX graphs via TorchTPU's FX->MLIR->PjRt
pipeline. Each shape bucket gets its own compiled executable.
"""

import copy
import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Callable

import torch
import torch._guards
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch_tpu._internal.compile._backend import TpuBackend
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Type alias for the compile range tuple
Range = tuple[int, int]

_tpu_backend = TpuBackend()

_RUNTIME_CACHE_KEY_PATHS = (
    "vllm_torchtpu/layers/vllm/fused_moe.py",
    "vllm_torchtpu/layers/common/fused_moe_gmm.py",
    "vllm_torchtpu/kernels/megablox",
    "vllm_torchtpu/kernels/sparse_core",
    "vllm_torchtpu/layers/vllm/attention.py",
    "vllm_torchtpu/layers/common/attention_interface.py",
    "vllm_torchtpu/kernels/ragged_paged_attention/v3",
    "vllm_torchtpu/kernels/experimental/batched_rpa",
    "vllm_torchtpu/layers/vllm/custom_ops/gdn_attention_op.py",
    "vllm_torchtpu/layers/common/gdn_attention.py",
    "vllm_torchtpu/layers/common/ragged_gated_delta_rule_wrapper.py",
    "vllm_torchtpu/layers/common/ragged_gated_delta_rule_ref.py",
    "vllm_torchtpu/kernels/gdn",
    "vllm_torchtpu/kernels/causal_conv1d",
    "vllm_torchtpu/kernels/experimental/pcp_streaming_rpa",
)


def _iter_runtime_cache_key_files(repo_root: Path) -> list[Path]:
    """Return runtime source files that affect TPU custom-op lowering."""
    files_by_relpath: dict[str, Path] = {}

    for rel_path in _RUNTIME_CACHE_KEY_PATHS:
        path = repo_root / rel_path
        if path.is_file():
            files_by_relpath[rel_path] = path
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                child_relpath = child.relative_to(repo_root).as_posix()
                files_by_relpath[child_relpath] = child
            continue
        logger.warning(
            "[TpuCompilerAdaptor] Cache key source path missing: %s", path)

    return [files_by_relpath[key] for key in sorted(files_by_relpath)]


# TODO(geyuhao): Switch this cache-key hashing to upstream vllm.ir.util.hash_source
# once our local vLLM checkout exposes that helper.
def _ensure_tuple_output(graph: fx.GraphModule) -> tuple[fx.GraphModule, bool]:
    """Wrap a graph module to return a tuple if it returns a single tensor.

    aot_autograd requires graph outputs to be tuples. PiecewiseBackend
    sub-graphs may return a single tensor.

    Returns (wrapped_graph, was_wrapped).
    """
    output_node = None
    for node in graph.graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        return graph, False

    output_args = output_node.args[0]
    if isinstance(output_args, (tuple, list)):
        return graph, False

    # Single return value — wrap in a tuple
    with graph.graph.inserting_before(output_node):
        output_node.args = ((output_args, ), )
    graph.graph.lint()
    graph.recompile()
    return graph, True


class TpuCompilerAdaptor(CompilerInterface):
    """CompilerInterface implementation for TPU.

    Delegates to TpuBackend which handles aot_autograd decomposition
    and FX->MLIR->PjRt compilation. Supports per-shape disk caching
    via pickle serialization of compiled executables.
    """

    name = "tpu"

    def __init__(self):
        self.cache_dir: str | None = None
        self._disable_cache = False

    def initialize_cache(self,
                         cache_dir: str,
                         disable_cache: bool = False,
                         prefix: str = "") -> None:
        vllm_xla_cache_path = os.getenv("VLLM_XLA_CACHE_PATH")
        if vllm_xla_cache_path:
            try:
                from vllm.envs import VLLM_CACHE_ROOT
                rel_path = os.path.relpath(cache_dir, VLLM_CACHE_ROOT)
                cache_dir = os.path.join(vllm_xla_cache_path, rel_path)
            except Exception as e:
                logger.warning(
                    "[TpuCompilerAdaptor] Failed to relocate cache directory "
                    "using VLLM_XLA_CACHE_PATH: %s", e)

        self.cache_dir = cache_dir
        self._disable_cache = disable_cache

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        """Hash TPU runtime sources that sit behind custom-op boundaries."""
        import torch_tpu

        repo_root = Path(__file__).resolve().parents[2]
        hash_obj = hashlib.sha256()

        hash_obj.update(f"torch={torch.__version__}".encode())
        hash_obj.update(
            f"torch_tpu={getattr(torch_tpu, '__version__', 'unknown')}".encode(
            ))
        # Kernel choice (default vs batched RPA) no longer needs to be in the
        # cache key — each Impl registers under a distinct op name
        # (`pallas::rpa_kernel_*` vs `pallas::rpa_kernel_batched_*`), so the
        # FX graph already discriminates.

        # KV-cache budget inputs. The compiled FX graphs embed the resolved
        # num_gpu_blocks as a literal, but num_gpu_blocks isn't known until
        # after profile_run computes the KV-cache budget from these env+config
        # values. Hashing them catches changes that would otherwise produce a
        # stale-cache "tensor X vs Y expanded size" error on first execution
        # instead of a clean recompile.
        hash_obj.update(
            f"kv_headroom_mib={envs.TPU_KV_CACHE_HEADROOM_MIB}".encode())
        hash_obj.update(
            f"gpu_mem_util={vllm_config.cache_config.gpu_memory_utilization}".
            encode())
        hash_obj.update(
            f"max_num_batched_tokens="
            f"{vllm_config.scheduler_config.max_num_batched_tokens}".encode())
        hash_obj.update(
            f"num_gpu_blocks_override="
            f"{vllm_config.cache_config.num_gpu_blocks_override}".encode())
        spec = vllm_config.speculative_config
        hash_obj.update(f"spec_draft="
                        f"{getattr(spec, 'model', None)}".encode())
        hash_obj.update(
            f"spec_draft_tp="
            f"{getattr(spec, 'draft_tensor_parallel_size', None)}".encode())
        hash_obj.update(
            f"spec_num_tokens="
            f"{getattr(spec, 'num_speculative_tokens', None)}".encode())

        for path in _iter_runtime_cache_key_files(repo_root):
            relpath = path.relative_to(repo_root).as_posix()
            hash_obj.update(relpath.encode())
            try:
                hash_obj.update(path.read_bytes())
            except OSError as exc:
                logger.warning(
                    "[TpuCompilerAdaptor] Failed to read cache key source %s: %s",
                    path,
                    exc,
                )

        return hash_obj.hexdigest()[:10]

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        """Compile a Dynamo-level FX graph to a TPU PjRt executable.

        Delegates to TpuBackend which handles aot_autograd + fx_to_mlir
        + compile_mlir internally.
        """
        logger.info(
            "[TpuCompilerAdaptor] Compiling FX graph for range %s",
            compile_range,
        )

        # aot_autograd (inside TpuBackend) requires tuple outputs.
        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        # `_tpu_backend` calls detect_fake_mode(), which asserts a single
        # FakeTensorMode. The Dynamo tracing context's mode differs from the
        # example inputs' mode, so run under a context built from the example
        # inputs' own fake mode (consistent for detect_fake_mode). We also ensure
        # that mode has a ShapeEnv: with enable_serialization the bundled
        # AOTAutogradCache key computation (FxGraphCache._check_can_cache)
        # bypasses with "No shape env" when the tracing context has none, which
        # is fatal for these static graphs. An (empty) ShapeEnv yields empty
        # guards -- correct for static shapes -- and lets the artifact cache work.
        from torch._subclasses.fake_tensor import FakeTensor
        fake_mode = next(
            (t.fake_mode for t in example_inputs if isinstance(t, FakeTensor)),
            None)
        if fake_mode is not None and fake_mode.shape_env is None:
            fake_mode.shape_env = ShapeEnv()
        tracing_ctx = (torch._guards.TracingContext(fake_mode)
                       if fake_mode is not None else None)
        with torch._guards.tracing(tracing_ctx):
            compiled_fn = _tpu_backend(graph, example_inputs)

        if was_wrapped:
            inner_fn = compiled_fn

            def unwrap_fn(*args, **kwargs):
                result = inner_fn(*args, **kwargs)
                if isinstance(result, (tuple, list)) and len(result) == 1:
                    return result[0]
                return result

            compiled_fn = unwrap_fn

        # Save the per-shape PjRt executable to disk.
        # We pickle the inner _TorchTpuCompiledExecutable (which has
        # __reduce__), not the aot_autograd wrapper (which has
        # unpicklable closures).
        handle = None
        if (key is not None and self.cache_dir is not None
                and not self._disable_cache
                and _tpu_backend._compiled_executables):
            try:
                inner_exe = _tpu_backend._compiled_executables[-1]
                save_path = os.path.join(self.cache_dir, key)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump(inner_exe, f)
                handle = (key, save_path, was_wrapped)
                logger.info(
                    "[TpuCompilerAdaptor] Saved compiled executable to %s",
                    save_path,
                )
            except Exception as e:
                logger.warning(
                    "[TpuCompilerAdaptor] Failed to save executable: %s", e)

        return compiled_fn, handle

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]:
        """Load a compiled executable from disk.

        Unpickles the _TorchTpuCompiledExecutable, then wraps it with
        aot_autograd (using the same graph) to reconstruct the full
        callable. This skips FX->MLIR->compile but still sets up the
        aot_autograd runtime wrapper that handles input mutations.
        """
        assert isinstance(handle, tuple) and len(handle) == 3
        key, path, was_wrapped = handle
        logger.info("[TpuCompilerAdaptor] Loading compiled executable from %s",
                    path)

        with open(path, "rb") as f:
            inner_exe = pickle.load(f)

        # Re-create the aot_autograd wrapper around the cached executable.
        # The current torch_tpu executable already handles non-tensor argument
        # filtering internally, so the load path can return the unpickled
        # executable directly.
        from torch._dynamo.backends.common import aot_autograd

        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        def _cached_compiler(*_args, **_kwargs):
            return inner_exe

        # The tracing context has a FakeTensorMode from Dynamo, but the example
        # inputs have fake tensors from a different FakeTensorMode.
        # `aot_autograd` calls detect_fake_mode() which asserts all
        # FakeTensorModes match, causing a crash.
        # Clear the tracing context and let `aot_autograd` create its own.
        with torch._guards.tracing(None):
            compiled_fn = aot_autograd(
                fw_compiler=_cached_compiler,
                keep_inference_input_mutations=False,
            )(graph, example_inputs)

        if was_wrapped:
            inner_fn = compiled_fn

            def unwrap_fn(*args, **kwargs):
                result = inner_fn(*args, **kwargs)
                if isinstance(result, (tuple, list)) and len(result) == 1:
                    return result[0]
                return result

            compiled_fn = unwrap_fn

        return compiled_fn
