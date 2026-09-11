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
    """Metadata handle saved to vLLM's compilation cache directory.

    `entry` is torch's bundled AOTAutograd cache entry: the aot_autograd
    wrapper metadata and the compiled PjRt executable as a single serializable
    object. Keeping them together is the point -- a wrapper rebuilt from a
    fresh trace and an executable fetched separately can disagree, and the
    disagreement is only observable once the program runs.

    `signature` pins the graph the entry was built from, so a replay against a
    structurally different graph is refused instead of silently executing.

    Defaults are deliberately "unusable": an older pickle restored into this
    class keeps its own __dict__ and picks up these class defaults for the
    fields it lacks, so it fails the version check rather than being replayed.
    """
    key: str
    entry: Any | None = None
    signature: tuple | None = None
    was_wrapped: bool = False


_tpu_backend = TpuBackend()


def _deserialize_entry(entry: Any) -> Callable[..., Any]:
    """Rebuild a runnable from a bundled AOTAutograd entry.

    Restores the aot_autograd wrapper *and* the executable from the one entry,
    and returns a varargs callable over the graph's placeholders -- the same
    calling convention TpuBackend hands back, so `compile` and `load` produce
    interchangeable runnables.

    Module-level, and paired with `_tpu_backend`: a test that swaps the backend
    for a fake has to swap this too, since the fake's "entry" never came from
    torch and cannot go back through it.
    """
    from torch._functorch._aot_autograd.aot_autograd_result import \
        deserialize_bundled_cache_entry
    return deserialize_bundled_cache_entry(entry)


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
    # KV-transfer wiring: ports, socket paths, staging pool, pool sizes,
    # timeouts. Same reasoning as the Raiden group above. Every registry
    # entry not named here joins the cache key, so omitting these would
    # recompile the world whenever an operator retuned a port.
    "TPU_IPC_SOCKET_DIR",
    "TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS",
    "TPU_KV_COORD_EXECUTOR_MAX_WORKERS",
    "TPU_KV_LATENCY_LOG_INTERVAL",
    "TPU_KV_PIN_SHM",
    "TPU_KV_SHM_POOL_GB",
    "TPU_KV_STAGE_WAITER_POOL_SIZE",
    "TPU_KV_STAGE_WAIT_TIMEOUT_SECS",
    "TPU_KV_TRANSFER_CHANNEL_NUMBER",
    "TPU_KV_TRANSFER_NAMESPACE",
    "TPU_KV_TRANSFER_PORT",
    # Compiles the same artifacts, just eagerly at startup.
    "TPU_KV_WARMUP_ENABLED",
    "TPU_NODE_ID",
    "TPU_P2P_WAIT_PULL_TIMEOUT",
    "TPU_RAIDEN_INLINE_LOAD",
    "TPU_RAIDEN_POOL_STAGING_LEASES",
    "TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT",
    "TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S",
    "TPU_RAIDEN_STAGE3_STATUS_PROBE_S",
    "TPU_RAIDEN_TEST_REGISTRATION_DELAY_S",
    "TPU_RAIDEN_TRANSFER_NUM_SLOTS",
    "TPU_SIDE_CHANNEL_PORT",
    # The HBM reserve that feeds block-count sizing comes from
    # kv_transfer_config, not from this variable.
    "TPU_USE_RAIDEN_CONNECTOR",
    # Pipeline chunk sizing: scheduling and the extra token buckets, which
    # are keyed by shape inside the same cache.
    "TPU_PP_DYNAMIC_CHUNKS",
    "TPU_PP_CHUNK_SLACK",
    # Startup and host-side knobs: ports, an IPC secret, two raiden startup
    # checks, offload retry and drain limits, a weight-load sync cadence, a
    # debug log switch, and the local-rank offsets, which move a worker to a
    # different chip without changing the executable it runs.
    "DEBUG_TPU_LOCAL_RANK_OFFSET",
    "RAIDEN_DISABLE_SINGLETON_WORKER",
    "RAIDEN_SHM_KEY",
    # Compiles the same spec-decode artifacts, just eagerly at startup.
    "SPEC_WARMUP",
    "TORCH_TPU_BASE_PORT",
    "TORCH_TPU_MP_RENDEZVOUS_PORT",
    "TPU_LOCAL_RANK_OFFSET",
    "TPU_SHARDED_LOAD_SYNC_EVERY",
    "VLLM_TORCHTPU_IPC_KEY",
    "VLLM_TPU_DEBUG_PCP_LAYOUT",
    "VLLM_TPU_OFFLOAD_SAVE_RETRIES",
    "VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S",
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
    "vllm_torchtpu/layers/core",
    "vllm_torchtpu/layers/adapter/attention.py",
    "vllm_torchtpu/layers/adapter/cp_attention.py",
    "vllm_torchtpu/layers/adapter/custom_ops",
    "vllm_torchtpu/layers/adapter/fused_moe.py",
    # The fused EP MoE op and the helper that registers it. Both sit behind a
    # torch custom-op boundary Dynamo cannot see through, and the kernel they
    # export is chosen from their source, so an edit here has to invalidate
    # the compiled executables like any other kernel change.
    "vllm_torchtpu/layers/adapter/fused_moe_ep.py",
    "vllm_torchtpu/distributed/sharded_jax_op.py",
    # And the mesh they export with: `build_ep_mesh` fixes the device ORDER
    # the kernel's remote DMAs resolve their peer coordinates against, so an
    # edit to the ordering changes where every routed row goes without
    # changing a byte of the kernel.
    "vllm_torchtpu/distributed/ep_mesh.py",
    "vllm_torchtpu/layers/adapter/linear_common.py",
    "vllm_torchtpu/layers/adapter/router_topk.py",
    "vllm_torchtpu/layers/adapter/quantization",
    "vllm_torchtpu/layers/adapter/vision_attention.py",
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

    @staticmethod
    def _unwrap_compiled_fn(compiled_fn):
        inner_fn = compiled_fn

        def unwrap_fn(*args, **kwargs):
            result = inner_fn(*args, **kwargs)
            if isinstance(result, (tuple, list)) and len(result) == 1:
                return result[0]
            return result

        return unwrap_fn

    def _run_backend(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
    ) -> tuple[Callable[..., Any], Any | None]:
        """Compile `graph` with TpuBackend, and grab its serializable entry.

        Shared by `compile` and by `load`'s fallback, so a cache miss and a
        cold start reach the backend by exactly the same route.

        Returns (runnable, entry). The runnable still returns a tuple -- the
        caller applies the `_ensure_tuple_output` unwrap. `entry` is None when
        the backend attached no `serialize`, which it skips for dynamic-symint
        graphs and in debug mode (see TpuBackend.__call__).
        """
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

        # Read `serialize` off the object TpuBackend returned, before any
        # wrapping: the unwrap below replaces it with a plain closure that does
        # not carry attributes, and the entry would be unreachable after that.
        entry = None
        serialize = getattr(compiled_fn, "serialize", None)
        if serialize is not None:
            try:
                entry = serialize()
            except Exception as e:
                logger.warning(
                    "[TpuCompilerAdaptor] serialize() failed, this graph will "
                    "not be cached: %s", e)
        return compiled_fn, entry

    def _save_handle(
        self,
        key: str,
        path: str,
        entry: Any,
        graph: fx.GraphModule,
        was_wrapped: bool,
    ) -> bool:
        """Write the bundled entry for `graph` to `path`. True if it landed."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = TpuCompilationHandle(
                key=key,
                entry=entry,
                signature=shape_variants.graph_signature(graph),
                was_wrapped=was_wrapped,
            )
            # Write-then-rename: a crash or a full disk partway through
            # pickling a 400 MB entry would otherwise leave a truncated file
            # that the next run reads back as a corrupt artifact.
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp_path, path)
            logger.info("[TpuCompilerAdaptor] Saved compiled artifact to %s",
                        path)
            return True
        except Exception as e:
            # Non-fatal by design: a failed save costs a recompile next run,
            # which is strictly better than persisting something unloadable.
            logger.warning("[TpuCompilerAdaptor] Failed to save artifact: %s",
                           e)
            return False

    def _replay(
        self,
        path: str,
        graph: fx.GraphModule,
        was_wrapped: bool,
    ) -> Callable[..., Any] | None:
        """Rebuild the runnable saved at `path`, or None to recompile.

        Every rejection here costs one recompile and nothing else, so this is
        deliberately strict: an artifact is replayed only if it is the current
        format and was built from a graph with this one's input signature.
        """
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("[TpuCompilerAdaptor] Unreadable artifact %s: %s",
                           path, e)
            return None

        reason = None
        if not isinstance(payload, TpuCompilationHandle):
            reason = f"unexpected payload type {type(payload).__name__}"
        elif payload.entry is None:
            reason = "no bundled entry"
        elif payload.signature != shape_variants.graph_signature(graph):
            # Not what makes the bundled entry safe -- that comes from wrapper
            # and executable being one object, with no lookup left to drift.
            # This is the same check BucketExecutables.get already applies to
            # the in-memory hand-over, extended across processes, and it earns
            # its place because nothing else validates the entry against the
            # graph: torch_tpu injects a synthetic cache_info and stubs out
            # generate_guards_expression, and replaying a pickle we wrote
            # ourselves never goes through AOTAutogradCache.try_load.
            # Placeholders only, so a changed graph *body* still passes -- that
            # is compute_tpu_compilation_hash's job, not this one.
            reason = (f"graph signature changed ({payload.signature} vs "
                      f"{shape_variants.graph_signature(graph)})")
        elif payload.was_wrapped != was_wrapped:
            reason = "output-tuple wrapping differs from the saved graph"
        if reason is not None:
            logger.warning("[TpuCompilerAdaptor] Ignoring artifact %s: %s",
                           path, reason)
            return None

        try:
            return _deserialize_entry(payload.entry)
        except Exception as e:
            logger.warning(
                "[TpuCompilerAdaptor] Could not replay artifact %s, "
                "recompiling: %s", path, e)
            return None

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

        Called once per (subgraph, token bucket). vLLM keeps the returned
        callable for this process and writes the returned handle into its
        compilation cache directory, where a later process reads it back and
        passes it to `load`. The handle is therefore an on-disk format.

        Returning `(fn, None)` means "usable now, but nothing was persisted" --
        vLLM will call `compile` again next run.
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

        # aot_autograd (inside TpuBackend) requires tuple outputs. Deep-copied
        # first because _ensure_tuple_output rewrites the output node in place
        # and vLLM keeps its own reference to `graph`. Only the output node
        # moves, so the placeholder signature is unaffected.
        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        compiled_fn, entry = self._run_backend(graph, example_inputs)

        if was_wrapped:
            # Undo _ensure_tuple_output on the result side, so callers see the
            # single tensor the original graph returned.
            compiled_fn = self._unwrap_compiled_fn(compiled_fn)

        # Persist torch's bundled AOTAutograd entry -- the aot_autograd wrapper
        # metadata and the compiled executable as one object. It has to be one
        # object: `load` cannot rebuild a wrapper from a fresh trace and pair
        # it with a separately-fetched executable, because nothing at that
        # point can check the two describe the same program, and a mismatch is
        # only observable once the program runs -- as e0102
        # RuntimeProgramInputMismatch if the padded input buffers differ in
        # size, and as a silently wrong result if they happen not to.
        #
        # When the Tier-3 C++ persistent cache is active torch_tpu also keeps
        # the PJRT binary itself (torch_tpu_tier3/*.bin); that covers the
        # recompile path in `load`, not this one.
        handle = None
        caching = (key is not None and self.cache_dir is not None
                   and not self._disable_cache)
        if caching and entry is not None:
            save_path = os.path.join(self.cache_dir, key)
            if self._save_handle(key, save_path, entry, graph, was_wrapped):
                handle = (key, save_path, was_wrapped)
        elif caching:
            logger.info(
                "[TpuCompilerAdaptor] No serializable artifact for range %s; "
                "it will be recompiled on the next start", compile_range)

        # Offer this bucket to later traces of the same model, so a graph that
        # refuses it reuses this runnable instead of building a second
        # executable for it (see shape_variants.BucketExecutables).
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
        """Rebuild a compiled runnable from the artifact `compile` saved.

        This is the warm-start path, and the reason a warm start is minutes
        rather than tens of minutes: FX->MLIR lowering and XLA compilation are
        both skipped. Replaying torch's bundled AOTAutograd entry restores the
        wrapper and the executable together, so there is no point at which a
        freshly traced wrapper could be paired with the wrong program.

        Falls back to a real compile whenever the artifact is missing, stale,
        or built from a different graph. That is never wrong, only slower --
        and with the Tier-3 C++ cache active it is not even much slower, since
        torch_tpu serves the PJRT binary natively.
        """
        assert isinstance(handle, tuple) and len(handle) == 3
        key, path, _saved_was_wrapped = handle
        size = _single_size(compile_range)
        # Screened on the load path too, and for a second reason beyond
        # "unsupported bucket": vLLM's key is (subgraph, bucket) with no graph
        # identity, so without this a re-traced variant would adopt an artifact
        # another trace compiled (see shape_variants.runnable_for).
        elsewhere = shape_variants.runnable_for(self.cache_dir, key, graph,
                                                size)
        if elsewhere is not None:
            return elsewhere

        logger.info("[TpuCompilerAdaptor] Loading compiled artifact from %s",
                    path)

        # Same deepcopy + output-tuple rewrite as `compile`, so the graph the
        # artifact is checked against is shaped the way it was when saved.
        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        compiled_fn = self._replay(path, graph, was_wrapped)
        if compiled_fn is None:
            compiled_fn, entry = self._run_backend(graph, example_inputs)
            # Self-heal: overwrite the artifact vLLM will hand us again next
            # run. Without this a single stale entry recompiles forever.
            if (entry is not None and self.cache_dir is not None
                    and not self._disable_cache):
                self._save_handle(key, path, entry, graph, was_wrapped)

        if was_wrapped:
            compiled_fn = self._unwrap_compiled_fn(compiled_fn)

        # A loaded bucket is offered to later traces exactly like a compiled
        # one, so a graph that refuses this bucket reuses this runnable instead
        # of replaying the artifact into a second HBM program region.
        if size is not None:
            shape_variants.remember(self.cache_dir, key, graph, compiled_fn)
        return compiled_fn
