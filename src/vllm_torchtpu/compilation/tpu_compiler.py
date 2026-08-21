# SPDX-License-Identifier: Apache-2.0
"""TPU compiler adaptor for vLLM's VLLM_COMPILE mode.

This module implements the CompilerInterface for TPU, enabling
PiecewiseBackend to compile FX graphs via TorchTPU's FX->MLIR->PjRt
pipeline. Each shape bucket gets its own compiled executable.
"""

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import pickle
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

import torch
import torch._guards
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch_tpu._internal.compile._backend import TpuBackend
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range

from vllm_torchtpu import envs, utils
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class TpuCompilationHandle:
    """Metadata handle saved to vLLM's compilation cache directory."""
    key: str
    executable: Any | None = None


_tpu_backend = TpuBackend()

_TPU_COMPILE_ENV_IGNORED = {
    # Startup, diagnostics, and orchestration only; these do not affect the
    # compiled model or custom-kernel lowering.
    "DP_SCHED_ENABLED",
    "PYTHON_TRACER_LEVEL",
    "RAY_USAGE_STATS_ENABLED",
    "TPU_NAME",
    "USE_PHASED_PROFILER",
    # Raiden KV-transfer orchestration/identity only; consumed at serving time
    # by the KV connector and never affect the compiled graph.
    "TPU_RAIDEN_CONTROLLER_ADDRESS",
    "TPU_RAIDEN_ENGINE_ID",
    "TPU_RAIDEN_JOB_NAME",
    "TPU_RAIDEN_TRANSFER_PARALLELISM",
    "TPU_WORKER_ID",
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE",
    "VLLM_XLA_CHECK_RECOMPILATION",
    "TPU_PARALLEL_PRECOMPILE",
}

_NATIVE_TPU_COMPILE_ENV_VARS = (
    "LIBTPU_INIT_ARGS",
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
    "TORCH_TPU_INTERNAL_XLA_OPTIONS",
    "XLA_FLAGS",
)

_RUNTIME_CACHE_KEY_PATHS = (
    # Pallas implementations execute behind torch custom-op boundaries, so
    # Dynamo's traced-file validation cannot see their source changes.
    "vllm_torchtpu/kernels",
    "vllm_torchtpu/layers/common",
    "vllm_torchtpu/layers/vllm/attention.py",
    "vllm_torchtpu/layers/vllm/custom_ops",
    "vllm_torchtpu/layers/vllm/fused_moe.py",
    "vllm_torchtpu/layers/vllm/linear_common.py",
    "vllm_torchtpu/layers/vllm/quantization",
    "vllm_torchtpu/layers/vllm/vision_attention.py",
    "vllm_torchtpu/models/vllm/deepseek_v4/attention.py",
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

    if envs.TPU_KERNEL_ITER_MODE:
        for relpath in _reloadable_kernel_relpaths(repo_root):
            files_by_relpath.pop(relpath, None)

    return [files_by_relpath[key] for key in sorted(files_by_relpath)]


def _reloadable_kernel_relpaths(repo_root: Path) -> list[str]:
    """Repo-relative paths of the hot-reloadable kernel modules.

    In kernel-iteration mode these execute eagerly outside the compiled
    pieces (splitting_ops in tpu_platform.py) and are re-lowered from source
    per process / per reload, so their bytes no longer affect the cached
    executables and are dropped from the cache key. Everything else in
    _RUNTIME_CACHE_KEY_PATHS stays hashed.
    """
    from vllm_torchtpu.compilation import kernel_reload

    relpaths = []
    for mod_name in kernel_reload.default_reload_modules():
        spec = importlib.util.find_spec(mod_name)
        origin = spec.origin if spec else None
        if not origin:
            logger.warning(
                "[TpuCompilerAdaptor] Cannot resolve reload module %s; its "
                "file stays in the cache key.", mod_name)
            continue
        try:
            relpaths.append(
                Path(origin).resolve().relative_to(repo_root).as_posix())
        except ValueError:
            # Outside the repo root: never part of the hashed set anyway.
            continue
    return relpaths


def _tpu_compile_env_factors() -> dict[str, Any]:
    """Return TorchTPU plugin env values that can affect compilation."""
    factors = {}
    for name, getter in envs.environment_variables.items():
        if name in _TPU_COMPILE_ENV_IGNORED:
            continue
        try:
            factors[name] = getter()
        except Exception as exc:
            logger.warning("Skipping TPU compile environment variable %s: %s",
                           name, exc)
    return factors


def compute_tpu_compilation_hash(vllm_config: VllmConfig) -> str:
    """Hash TPU-specific factors used by both AOT and piecewise caches."""
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    spec = vllm_config.speculative_config
    factors = {
        "torch": torch.__version__,
        "torch_tpu": importlib_metadata.version("torch-tpu"),
        "jax": importlib_metadata.version("jax"),
        "jaxlib": importlib_metadata.version("jaxlib"),
        "libtpu": importlib_metadata.version("libtpu"),
        "env": _tpu_compile_env_factors(),
        "native_env": {
            name: os.getenv(name)
            for name in _NATIVE_TPU_COMPILE_ENV_VARS
        },
        "parallel": {
            "effective_data_parallel_size":
            utils.get_dp_size(vllm_config.parallel_config),
        },
        # TPU AOT graphs embed the KV tensor extent as a literal. Hash the
        # resolved block count, which every rank shares via kv_cache_config.
        # The raw per-worker num_gpu_blocks_override is a pre-alignment HBM
        # measurement that diverges by a few blocks across ranks (the executor
        # aligns the value actually used to min()); hashing it would split
        # ranks into separate cache dirs and force asymmetric recompiles.
        "kv_cache": {
            "gpu_memory_utilization": cache_config.gpu_memory_utilization,
            "num_gpu_blocks": cache_config.num_gpu_blocks,
        },
        "scheduler": {
            "max_model_len": vllm_config.model_config.max_model_len,
            "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
            "max_num_seqs": scheduler_config.max_num_seqs,
        },
        "speculative": {
            "model":
            spec.model if spec is not None else None,
            "draft_tensor_parallel_size":
            spec.draft_tensor_parallel_size if spec is not None else None,
            "num_speculative_tokens":
            spec.num_speculative_tokens if spec is not None else None,
        },
    }

    repo_root = Path(__file__).resolve().parents[2]
    hash_obj = hashlib.sha256(
        json.dumps(factors, sort_keys=True, separators=(",", ":")).encode())
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


def _single_size(compile_range: Range) -> int | None:
    """The token bucket a compile range pins, or None for a size range."""
    return compile_range.start if compile_range.is_single_size() else None


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
        self.cache_dir = cache_dir
        self._disable_cache = disable_cache

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        """Hash TPU runtime sources that sit behind custom-op boundaries."""
        return compute_tpu_compilation_hash(vllm_config)

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
        # A bucket this graph cannot serve, or one an earlier trace of the same
        # model already built, is not compiled here (see shape_variants).
        size = _single_size(compile_range)
        if key is not None and size is not None:
            elsewhere = shape_variants.runnable_for(self.cache_dir, key, graph,
                                                    size)
            if elsewhere is not None:
                return elsewhere, None

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
        # When Tier-3 C++ persistent cache is active, torch_tpu handles PJRT
        # binary persistence natively in C++ (torch_tpu_tier3/*.bin).
        # We store a lightweight metadata handle instead of duplicate compiled binary executables.
        handle = None
        if (key is not None and self.cache_dir is not None
                and not self._disable_cache
                and _tpu_backend._compiled_executables):
            try:
                inner_exe = _tpu_backend._compiled_executables[-1]
                save_path = os.path.join(self.cache_dir, key)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                handle_data = TpuCompilationHandle(
                    key=key,
                    executable=inner_exe,
                )
                with open(save_path, "wb") as f:
                    pickle.dump(handle_data, f)
                handle = (key, save_path, was_wrapped)
                logger.info(
                    "[TpuCompilerAdaptor] Saved compiled executable metadata to %s",
                    save_path,
                )
            except Exception as e:
                logger.warning(
                    "[TpuCompilerAdaptor] Failed to save executable: %s", e)

        if key is not None and size is not None:
            shape_variants.remember(self.cache_dir, key, graph, compiled_fn)
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

        Unpickles the _TorchTpuCompiledExecutable (or uses Tier-3 C++ cache),
        then wraps it with aot_autograd (using the same graph) to reconstruct
        the full callable.
        """
        assert isinstance(handle, tuple) and len(handle) == 3
        key, path, was_wrapped = handle
        size = _single_size(compile_range)
        elsewhere = shape_variants.runnable_for(self.cache_dir, key, graph,
                                                size)
        if elsewhere is not None:
            return elsewhere

        logger.info("[TpuCompilerAdaptor] Loading compiled executable from %s",
                    path)

        with open(path, "rb") as f:
            loaded_data = pickle.load(f)

        # Re-create the aot_autograd wrapper around the cached executable.
        from torch._dynamo.backends.common import aot_autograd

        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        inner_exe = None
        if isinstance(loaded_data, TpuCompilationHandle):
            inner_exe = loaded_data.executable
        elif isinstance(loaded_data, dict):
            inner_exe = loaded_data.get("executable")
        else:
            inner_exe = loaded_data

        if inner_exe is not None:

            def _cached_compiler(*_args, **_kwargs):
                return inner_exe
        else:
            # Tier-3 C++ cache is active: _tpu_backend will hit native C++ Tier-3 binary cache (<1ms)
            def _cached_compiler(g, example_inputs):
                clean_inputs = [
                    int(x) if isinstance(x, torch.SymInt) else x
                    for x in example_inputs
                ]
                return _tpu_backend(g, clean_inputs)

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

        if size is not None:
            shape_variants.remember(self.cache_dir, key, graph, compiled_fn)
        return compiled_fn
