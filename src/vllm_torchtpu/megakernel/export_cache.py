# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A persistent export cache for the megakernel's torch ops.

torch_tpu exports (traces and lowers) every op on its first call in a process
just to fingerprint the module and find the compiled executable. With
VLLM_XLA_CACHE_PATH set, ops built with an ``export_key`` (``megakernel.bridge``)
persist that call's serialized module, fingerprint and output placeholders
under ``<VLLM_XLA_CACHE_PATH>/k3_export`` (keyed in ``_ExportCache._path``),
and later processes register the module straight from disk.
"""

import hashlib
import os
import pathlib
import pickle
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import torch
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.distributed.sharded_jax_op import _placeholder
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_EXPORT_ENV_PREFIXES = ("JAX_",)
_SOURCE_FINGERPRINT: str | None = None


def _source_fingerprint() -> str:
    global _SOURCE_FINGERPRINT
    if _SOURCE_FINGERPRINT is None:
        import importlib.metadata

        import vllm_torchtpu

        digest = hashlib.blake2b(digest_size=16)
        root = pathlib.Path(vllm_torchtpu.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        for package in ("jax", "jaxlib", "torch_tpu", "libtpu"):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "-"
            digest.update(f"{package}={version};".encode())
        _SOURCE_FINGERPRINT = digest.hexdigest()
    return _SOURCE_FINGERPRINT


def _process_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:  # best effort
        pass
    return int(jax.process_index())


class _ExportCache:
    """The persistent export cache of one op (see the module docstring)."""

    def __init__(self, root: pathlib.Path, export_key: str):
        self.root, self.export_key = root, export_key
        self.shared_rank: int | None = None

    def _path(self, op, args, kwargs, *, rank: int | None = None) -> pathlib.Path:
        if rank is None:
            rank = self.shared_rank
        if rank is None:
            rank = _process_rank()
        signature: list[object] = []
        for index, arg in enumerate(args):
            if index in op.static_argnums:
                signature.append(("static", repr(arg)))
            elif arg is None:
                signature.append(None)
            else:
                signature.append((tuple(arg.shape), str(arg.dtype)))
        env = sorted(
            (k, v) for k, v in os.environ.items() if k.startswith(_EXPORT_ENV_PREFIXES)
        )
        mesh = (
            (tuple(int(d.id) for d in op.mesh.devices.flat), tuple(op.mesh.axis_names))
            if op.mesh
            else None
        )
        key = repr(
            (
                op.name,
                self.export_key,
                signature,
                sorted((k, repr(v)) for k, v in kwargs.items()),
                repr(op.input_partition_specs),
                repr(op.output_partition_specs),
                tuple(op.donate_argnums),
                op.static_argnums,
                mesh,
                rank,
                _source_fingerprint(),
                env,
            )
        )
        digest = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in op.name)[:80]
        return self.root / f"{safe}-{digest}.pkl"

    def call(self, op, call, args, kwargs):
        kernel_key = pallas_impl._get_kernel_invocation_key(
            op.trace_key, args, kwargs, op.static_argnums
        )
        if kernel_key in op.output_shapes:
            return call(op, *args, **kwargs)
        path = self._path(op, args, kwargs)
        if self._load(op, kernel_key, path):
            return call(op, *args, **kwargs)
        op._recorded_export = None
        result = call(op, *args, **kwargs)
        lowered = op._recorded_export
        fingerprint = op.kernel_key_to_mlir_fingerprint.get(kernel_key)
        if lowered is not None and fingerprint is not None:
            try:
                self._store(path, fingerprint, lowered)
            except OSError as exc:
                logger.warning("export cache: could not store %s: %s", path, exc)
        return result

    def _load(self, op, kernel_key, path: pathlib.Path) -> bool:
        try:
            with open(path, "rb") as handle:
                entry = pickle.load(handle)
            module = (self.root / f"module-{entry['fingerprint']}.mlirbc").read_bytes()
        except (OSError, EOFError, pickle.UnpicklingError, KeyError):
            return False
        started = time.perf_counter()
        pallas_impl.tpu_torch_pallas.register_custom_kernel(
            op.name, entry["fingerprint"], serialized_mlir_module=module
        )
        outputs = [
            _placeholder(jax.core.ShapedArray(shape, dtype), spec, op.mesh)
            if array
            else pallas_impl.torch_placeholder(
                jax.core.ShapedArray(shape, dtype), mesh=op.mesh
            )
            for array, shape, dtype, spec in entry["outputs"]
        ]
        op.kernel_key_to_mlir_fingerprint[kernel_key] = entry["fingerprint"]
        op.output_shapes[kernel_key] = (outputs, entry["out_tree"])
        logger.debug(
            "export cache: %s from %s (%.2f s)",
            op.name,
            path.name,
            time.perf_counter() - started,
        )
        return True

    def _store(self, path: pathlib.Path, fingerprint: str, lowered) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        shardings = getattr(lowered, "_out_named_shardings", None)
        if shardings is None or len(shardings) != len(lowered.out_avals):
            shardings = (None,) * len(lowered.out_avals)
        outputs = []
        for aval, sharding in zip(lowered.out_avals, shardings):
            spec = getattr(sharding, "spec", None)
            outputs.append(
                (
                    spec is not None and isinstance(aval, jax.core.ShapedArray),
                    tuple(aval.shape),
                    aval.dtype,
                    spec,
                )
            )
        module_path = self.root / f"module-{fingerprint}.mlirbc"
        if not module_path.exists():
            _atomic_write(module_path, bytes(lowered.mlir_module_serialized))
        _atomic_write(
            path,
            pickle.dumps(
                {
                    "fingerprint": fingerprint,
                    "outputs": outputs,
                    "out_tree": lowered.out_tree,
                }
            ),
        )


def prepare_distributed(calls: Sequence[Any]) -> None:
    """Export rank-invariant programs on different workers.

    Every worker passes the same ordered programs, whose bodies and export keys
    do not depend on the rank. Each program's owner traces it on fresh
    placeholders without compiling or executing it; the others later load its
    entry from the export cache. Without the cache, fall back to per-worker
    export. All workers see export errors before any proceeds to compilation.
    """
    if not calls or not torch.distributed.is_initialized():
        return
    from torch_tpu._internal import execution_mode
    from torch_tpu._internal.compile import tpu_torch_compile

    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    # CustomOpDef keeps the registered JaxCallable as its initial function;
    # ops without the export cache fall back to per-worker export.
    ops: list[Any] = [getattr(call.op, "_init_fn", None) for call in calls]
    eligible = all(
        getattr(op, "_export_cache", None) is not None
        and op.mesh is not None
        and op.mesh.size == world
        for op in ops
    )
    signatures = (
        [
            (
                op._export_cache._path(op, call.operands, {}, rank=index % world).name,
                sorted(call.options.items()),
            )
            for index, (op, call) in enumerate(zip(ops, calls, strict=True))
        ]
        if eligible
        else None
    )
    gathered: list[Any] = [None] * world
    torch.distributed.all_gather_object(gathered, signatures)
    if any(signature is None for signature in gathered):
        return
    if any(signature != signatures for signature in gathered):
        raise ValueError("distributed export requires identical program signatures")

    for index, op in enumerate(ops):
        op._export_cache.shared_rank = index % world
    failure: Exception | None = None
    try:
        for index, call in enumerate(calls):
            if index % world != rank:
                continue
            inputs = tuple(tpu_torch_compile.placeholder_like(t) for t in call.operands)
            with execution_mode.set_eager_mode(
                execution_mode.EagerMode.INTERNAL_COMPILE_FX_GRAPH
            ):
                call.op(*inputs)
    except Exception as exc:
        failure = exc
    torch.distributed.all_gather_object(
        gathered, None if failure is None else f"rank {rank}: {failure!r}"
    )
    if any(error is not None for error in gathered):
        raise RuntimeError(f"distributed export failed: {gathered}") from failure


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


_CACHED_CLASSES: dict[type, type] = {}


def _cached_class(cls: type) -> type:
    """``cls`` whose ``__call__`` goes through the op's export cache."""
    cached = _CACHED_CLASSES.get(cls)
    if cached is None:
        call = cls.__call__

        def __call__(self, *args, **kwargs):
            return self._export_cache.call(self, call, args, kwargs)

        cached = type(f"{cls.__name__}WithExportCache", (cls,), {"__call__": __call__})
        _CACHED_CLASSES[cls] = cached
    return cached


def hook(export_key: str | None) -> Callable[[Any], None] | None:
    """The ``sharded_jax_op(configure_callable=...)`` hook that puts an op
    with ``export_key`` on the cache, or None when there is no key or no
    cache path (only an explicitly set VLLM_XLA_CACHE_PATH: its default is a
    home directory)."""
    import vllm.envs as vllm_envs

    if not export_key or not vllm_envs.is_set("VLLM_XLA_CACHE_PATH"):
        return None
    root = pathlib.Path(vllm_envs.VLLM_XLA_CACHE_PATH) / "k3_export"

    def configure(op) -> None:
        exported = op.exported

        def recording(*args, **kwargs):
            lowered = exported(*args, **kwargs)
            op._recorded_export = lowered
            return lowered

        op.exported = recording
        op._export_cache = _ExportCache(root, export_key)
        op.__class__ = _cached_class(type(op))

    return configure
