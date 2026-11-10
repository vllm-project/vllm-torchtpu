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
# src/vllm_torchtpu/megakernel/export_cache.py
"""A persistent export cache for the megakernel's torch ops.

torch_tpu exports (traces and lowers) every op on its first call in a process
just to fingerprint the module and find the compiled executable. With
VLLM_XLA_CACHE_PATH set, ops built with an ``export_key`` (``megakernel.bridge``)
persist that call's serialized module, fingerprint and output placeholders
under ``<VLLM_XLA_CACHE_PATH>/k3_export`` (keyed in ``_ExportCache._path``),
and later processes register the module straight from disk.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import os
import pathlib
import pickle
import socket
import time
from typing import TYPE_CHECKING, Protocol, cast

import jax
import torch
from torch_tpu._internal.pallas import pallas as pallas_impl

import vllm_torchtpu
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from jax.sharding import Mesh, PartitionSpec


class _JaxCallableProtocol(Protocol):
    """Structural contract for ``torch_tpu`` ``JaxCallable`` instances."""

    name: str
    mesh: Mesh | None
    input_partition_specs: object
    output_partition_specs: object
    donate_argnums: Iterable[int]
    static_argnums: tuple[int, ...]
    trace_key: str
    output_shapes: dict[object, object]
    kernel_key_to_mlir_fingerprint: dict[object, str]
    exported: Callable[..., _LoweredExportProtocol]
    _export_cache: _ExportCache | None
    _recorded_export: _LoweredExportProtocol | None

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Invoke the wrapped callable."""
        ...


class _LoweredExportProtocol(Protocol):
    """Structural contract for lowered JAX export artifacts."""

    out_avals: Sequence[jax.core.ShapedArray]
    mlir_module_serialized: bytes
    out_tree: object


class _CompileCallOpProtocol(Protocol):
    """Callable wrapper inside a :class:`CompileCall` record."""

    _init_fn: _JaxCallableProtocol | None

    def __call__(self, *args: object) -> object: ...


class _CompileCallProtocol(Protocol):
    """Structural contract for ``compilation.CompileCall`` instances."""

    op: _CompileCallOpProtocol
    operands: tuple[object, ...]
    options: dict[str, object]


class _ShapeAndDtypeProtocol(Protocol):
    """Array/tensor shape and dtype protocol for cache key construction."""

    shape: Iterable[int]
    dtype: object


logger = init_logger(__name__)

_EXPORT_ENV_PREFIXES = ("JAX_",)


def _create_torch_placeholder(
    aval: jax.core.ShapedArray,
    spec: PartitionSpec | None,
    mesh: Mesh | None,
) -> torch.Tensor:
    """Build a local-shard torch placeholder tensor for ``aval`` under ``spec``.

    ``jax.export`` records ``shard_map`` outputs with a replicated aval; when an
    output ``PartitionSpec`` is present, this helper sizes the placeholder to the
    local shard on ``mesh`` instead of the global array shape.

    Args:
        aval: Exported JAX ``ShapedArray`` describing global shape and dtype.
        spec: Output ``PartitionSpec`` for the shard, or ``None`` if replicated.
        mesh: JAX device mesh over which ``spec`` partitions ``aval``.

    Returns:
        Empty ``torch.Tensor`` sized to this worker's local output shard.

    Raises:
        RuntimeError: If ``torch_placeholder`` returns ``None``.
        NotImplementedError: If ``aval.dtype`` has no ``torch.dtype`` mapping.
        ValueError: If a packed 4-bit output has an odd minor dimension.
    """
    if spec is None or not isinstance(aval, jax.core.ShapedArray):
        placeholder = pallas_impl.torch_placeholder(aval, mesh=mesh)
        if placeholder is None:
            msg = f"torch_placeholder returned None for aval {aval}"
            raise RuntimeError(msg)
        return placeholder
    dtype_map: dict[object, torch.dtype] = pallas_impl.JAX_TO_TORCH_DTYPE_MAP
    torch_dtype = dtype_map.get(aval.dtype)
    if torch_dtype is None:
        msg = f"Unsupported dtype for pallas kernels: {aval.dtype}"
        raise NotImplementedError(msg)
    shape = tuple(
        int(dim) for dim in pallas_impl.get_local_shape(aval.shape, mesh, spec)
    )
    if torch_dtype == torch.float4_e2m1fn_x2 and shape:
        if shape[-1] % 2:
            msg = (
                "expected the last dimension of the JAX local shape to be even "
                f"to pack into {torch_dtype}, got {shape[-1]} (local={shape}, "
                f"global={aval.shape}, spec={spec})"
            )
            raise ValueError(msg)
        shape = (*shape[:-1], shape[-1] // 2)
    try:
        return torch.empty(shape, dtype=torch_dtype, device="tpu")
    except RuntimeError:
        return torch.empty(shape, dtype=torch_dtype, device="meta")


@functools.cache
def _source_fingerprint() -> str:
    """Compute and memoize a Blake2b digest over ``vllm_torchtpu`` sources and versions.

    Returns:
        32-hex-character Blake2b digest string.
    """
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
    return digest.hexdigest()


def _process_rank() -> int:
    """Return the active distributed worker rank, falling back to ``jax.process_index``.

    Returns:
        Zero-based worker rank integer.
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except RuntimeError as exc:  # best effort
        logger.debug("export cache: rank lookup fallback to jax.process_index: %s", exc)
    return int(jax.process_index())


class _ExportCache:
    """Persistent on-disk MLIR export cache for one registered custom op.

    The persistent export cache of one op (see the module docstring).
    """

    def __init__(self, root: pathlib.Path, export_key: str) -> None:
        """Initialize the cache directory and static export key.

        Args:
            root: Directory storing serialized ``.pkl`` and ``.mlirbc`` artifacts.
            export_key: Static configuration key baked into the op's trace.
        """
        self.root, self.export_key = root, export_key
        self.shared_rank: int | None = None

    def path(
        self,
        op: _JaxCallableProtocol,
        args: tuple[_ShapeAndDtypeProtocol | object, ...],
        kwargs: dict[str, object],
        *,
        rank: int | None = None,
    ) -> pathlib.Path:
        """Return the cache metadata file path for ``op`` invoked with ``args``.

        Args:
            op: Registered ``JaxCallable`` instance.
            args: Positional arguments passed to the op call.
            kwargs: Keyword arguments passed to the op call.
            rank: Optional explicit rank override for distributed export sharing.

        Returns:
            Filesystem ``Path`` to the ``.pkl`` cache entry.
        """
        if rank is None:
            rank = self.shared_rank
        if rank is None:
            rank = _process_rank()
        static_argnums = op.static_argnums
        signature: list[object] = []
        for index, arg in enumerate(args):
            if index in static_argnums:
                signature.append(("static", repr(arg)))
            elif arg is None:
                signature.append(None)
            else:
                shaped = cast("_ShapeAndDtypeProtocol", arg)
                signature.append((tuple(shaped.shape), str(shaped.dtype)))
        env = sorted(
            (k, v) for k, v in os.environ.items() if k.startswith(_EXPORT_ENV_PREFIXES)
        )
        op_mesh = op.mesh
        mesh = (
            (
                tuple(int(dev.id) for dev in op_mesh.devices.flat),
                tuple(op_mesh.axis_names),
            )
            if op_mesh
            else None
        )
        op_name = str(op.name)
        key = repr(
            (
                op_name,
                self.export_key,
                signature,
                sorted((k, repr(v)) for k, v in kwargs.items()),
                repr(op.input_partition_specs),
                repr(op.output_partition_specs),
                tuple(op.donate_argnums),
                static_argnums,
                mesh,
                rank,
                _source_fingerprint(),
                env,
            )
        )
        digest = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in op_name)[:80]
        return self.root / f"{safe}-{digest}.pkl"

    _path = path

    def call(
        self,
        op: _JaxCallableProtocol,
        call: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Execute ``call(op, *args, **kwargs)``, loading or storing the export cache.

        Args:
            op: Registered ``JaxCallable`` instance.
            call: Underlying ``__call__`` method on ``op``'s class.
            args: Positional arguments to ``op``.
            kwargs: Keyword arguments to ``op``.

        Returns:
            Result of ``call(op, *args, **kwargs)``.
        """
        kernel_key = pallas_impl._get_kernel_invocation_key(
            op.trace_key,
            args,
            kwargs,
            op.static_argnums,
        )
        if kernel_key in op.output_shapes:
            return call(op, *args, **kwargs)
        cache_path = self.path(op, args, kwargs)
        if self._load(op, kernel_key, cache_path):
            return call(op, *args, **kwargs)
        op._recorded_export = None
        result = call(op, *args, **kwargs)
        lowered = op._recorded_export
        fingerprint = op.kernel_key_to_mlir_fingerprint.get(kernel_key)
        if lowered is not None and fingerprint is not None:
            try:
                self._store(cache_path, fingerprint, lowered)
            except OSError as exc:
                logger.warning("export cache: could not store %s: %s", cache_path, exc)
        return result

    def _load(
        self, op: _JaxCallableProtocol, kernel_key: object, path: pathlib.Path
    ) -> bool:
        """Load a cached MLIR module and output placeholders from ``path``.

        Args:
            op: Registered ``JaxCallable`` instance to populate.
            kernel_key: Invocation key for the current call signature.
            path: Path to the ``.pkl`` metadata file.

        Returns:
            ``True`` when the cached export was loaded, ``False`` on cache miss.
        """
        try:
            entry = pickle.loads(path.read_bytes())
            fingerprint = str(entry["fingerprint"])
            module = (self.root / f"module-{fingerprint}.mlirbc").read_bytes()
            raw_outputs = entry["outputs"]
            out_tree = entry["out_tree"]
        except (OSError, EOFError, pickle.UnpicklingError, KeyError, TypeError):
            return False
        if not isinstance(raw_outputs, list):
            return False
        started = time.perf_counter()
        op_name = str(op.name)
        op_mesh = op.mesh
        pallas_impl.tpu_torch_pallas.register_custom_kernel(
            op_name, fingerprint, serialized_mlir_module=module
        )
        outputs = [
            _create_torch_placeholder(
                jax.core.ShapedArray(shape, dtype),
                spec,
                op_mesh,
            )
            if is_array
            else pallas_impl.torch_placeholder(
                jax.core.ShapedArray(shape, dtype), mesh=op_mesh
            )
            for is_array, shape, dtype, spec in raw_outputs
        ]
        op.kernel_key_to_mlir_fingerprint[kernel_key] = fingerprint
        op.output_shapes[kernel_key] = (outputs, out_tree)
        logger.debug(
            "export cache: %s from %s (%.2f s)",
            op_name,
            path.name,
            time.perf_counter() - started,
        )
        return True

    def _store(
        self,
        path: pathlib.Path,
        fingerprint: str,
        lowered: _LoweredExportProtocol,
    ) -> None:
        """Persist lowered MLIR bytecode and output shape metadata to ``self.root``.

        Args:
            path: Target ``.pkl`` metadata path.
            fingerprint: MLIR module fingerprint string.
            lowered: Lowered JAX export object carrying ``out_avals`` and MLIR bytes.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        out_avals = lowered.out_avals
        shardings = getattr(lowered, "_out_named_shardings", None)
        if shardings is None or len(shardings) != len(out_avals):
            shardings = (None,) * len(out_avals)
        outputs = []
        for aval, sharding in zip(out_avals, shardings, strict=True):
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


type CacheDirIdentity = tuple[str, int, int]
"""Host and filesystem inode identity ``(hostname, st_dev, st_ino)`` of a cache root."""


def cache_dir_identity(root: pathlib.Path | None) -> CacheDirIdentity | None:
    """Return ``(hostname, st_dev, st_ino)`` for ``root``, or ``None`` if unavailable.

    Args:
        root: Export cache directory path, or ``None`` when caching is disabled.

    Returns:
        Filesystem identity tuple distinguishing host-local vs shared directories.
    """
    if root is None:
        return None
    with contextlib.suppress(OSError):
        root.mkdir(parents=True, exist_ok=True)
        stat_result = root.stat()
        return (socket.gethostname(), stat_result.st_dev, stat_result.st_ino)
    return None


def peer_ranks_for_cache(
    identities: Sequence[CacheDirIdentity | None],
    rank: int,
) -> tuple[int, ...]:
    """Return ordered peer ranks sharing ``rank``'s cache directory identity.

    Args:
        identities: Per-rank cache directory identities gathered across the world.
        rank: Current distributed worker rank.

    Returns:
        Sorted tuple of ranks writing to the same physical cache directory.
    """
    target = identities[rank]
    if target is None:
        return (rank,)
    return tuple(peer for peer, ident in enumerate(identities) if ident == target)


def assign_export_owner(index: int, peers: Sequence[int]) -> int:
    """Return the peer rank responsible for exporting program ``index``.

    Args:
        index: Zero-based index of the program in the compile schedule.
        peers: Non-empty ordered sequence of ranks sharing the cache directory.

    Returns:
        Assigned owner rank from ``peers``.

    Raises:
        ValueError: If ``peers`` is empty.
    """
    if not peers:
        msg = "peers must be non-empty"
        raise ValueError(msg)
    return peers[index % len(peers)]


def prepare_distributed(calls: Sequence[_CompileCallProtocol]) -> None:
    """Export rank-invariant programs across distributed workers in parallel.

    Every worker passes the same ordered programs, whose bodies and export keys
    do not depend on the rank. Each program's owner traces it on fresh
    placeholders without compiling or executing it; the others later load its
    entry from the export cache. Without the cache, fall back to per-worker
    export. All workers see export errors before any proceeds to compilation.

    Args:
        calls: Sequence of compile call objects with ``op``, ``operands``, and
            ``options`` attributes.

    Raises:
        ValueError: If program signatures differ across ranks.
        RuntimeError: If any rank fails while exporting its assigned program.
    """
    if not calls or not torch.distributed.is_initialized():
        return
    execution_mode = importlib.import_module("torch_tpu._internal.execution_mode")
    tpu_torch_compile = importlib.import_module(
        "torch_tpu._internal.compile.tpu_torch_compile"
    )

    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    # CustomOpDef keeps the registered JaxCallable as its initial function; ops without
    # the export cache fall back to per-worker export.
    ops: list[_JaxCallableProtocol | None] = [
        getattr(call.op, "_init_fn", None) for call in calls
    ]
    cached_ops: list[tuple[_JaxCallableProtocol, _ExportCache]] = []
    for op in ops:
        if op is None:
            break
        cache_obj = getattr(op, "_export_cache", None)
        mesh = getattr(op, "mesh", None)
        if cache_obj is None or mesh is None or mesh.size != world:
            break
        cached_ops.append((op, cache_obj))
    eligible = len(cached_ops) == len(ops)
    signatures = (
        [
            (
                cache_obj.path(op, call.operands, {}, rank=index % world).name,
                sorted(call.options.items()),
            )
            for index, ((op, cache_obj), call) in enumerate(
                zip(cached_ops, calls, strict=True)
            )
        ]
        if eligible
        else None
    )
    first_cache: _ExportCache | None = (
        cached_ops[0][1] if eligible and cached_ops else None
    )
    local_identity = (
        cache_dir_identity(first_cache.root) if first_cache is not None else None
    )
    gathered: list[tuple[object, CacheDirIdentity | None] | None] = [None] * world
    torch.distributed.all_gather_object(gathered, (signatures, local_identity))
    if any(item is None or item[0] is None for item in gathered):
        return
    if any(item[0] != signatures for item in gathered if item is not None):
        msg = "distributed export requires identical program signatures"
        raise ValueError(msg)

    identities = [item[1] if item is not None else None for item in gathered]
    peers = peer_ranks_for_cache(identities, rank)
    for index, (_, cache_obj) in enumerate(cached_ops):
        cache_obj.shared_rank = assign_export_owner(index, peers)
    eager_mode = execution_mode.EagerMode.INTERNAL_COMPILE_FX_GRAPH
    failure: Exception | None = None
    try:
        for index, call in enumerate(calls):
            if assign_export_owner(index, peers) != rank:
                continue
            inputs = tuple(
                tpu_torch_compile.placeholder_like(tensor) for tensor in call.operands
            )
            with execution_mode.set_eager_mode(eager_mode):
                call.op(*inputs)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        LookupError,
        AssertionError,
    ) as exc:
        failure = exc
    torch.distributed.all_gather_object(
        gathered, None if failure is None else f"rank {rank}: {failure!r}"
    )
    if any(error is not None for error in gathered):
        msg = f"distributed export failed: {gathered}"
        raise RuntimeError(msg) from failure


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via a PID-scoped temporary file.

    Args:
        path: Destination file path.
        data: Raw bytes to write.
    """
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


_CACHED_CLASSES: dict[type, type] = {}


def _cached_class[JaxCallableT: _JaxCallableProtocol](
    cls: type[JaxCallableT],
) -> type[JaxCallableT]:
    """Return a subclass of ``cls`` whose ``__call__`` routes through ``_export_cache``.

    ``cls`` whose ``__call__`` goes through the op's export cache.

    Args:
        cls: Original ``JaxCallable`` class.

    Returns:
        Subclass of ``cls`` dispatching ``__call__`` through ``self._export_cache``.
    """
    cached = _CACHED_CLASSES.get(cls)
    if cached is None:
        call = cls.__call__

        def _invoke_with_cache(
            self: _JaxCallableProtocol,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Dispatch callable invocation through ``self._export_cache``.

            Args:
                self: ``JaxCallable`` instance with an attached ``_export_cache``.
                *args: Positional arguments forwarded to ``call``.
                **kwargs: Keyword arguments forwarded to ``call``.

            Returns:
                Result of the cached or freshly exported callable invocation.
            """
            cache_obj: _ExportCache | None = self._export_cache
            assert cache_obj is not None
            return cache_obj.call(self, call, args, kwargs)

        cached = type(
            f"{cls.__name__}WithExportCache",
            (cls,),
            {"__call__": _invoke_with_cache},
        )
        _CACHED_CLASSES[cls] = cached
    return cast("type[JaxCallableT]", cached)


def hook(export_key: str | None) -> Callable[[_JaxCallableProtocol], None] | None:
    """Build the ``sharded_jax_op(configure_callable=...)`` hook for ``export_key``.

    The ``sharded_jax_op(configure_callable=...)`` hook that puts an op
    with ``export_key`` on the cache, or None when there is no key or no
    cache path (only an explicitly set, non-empty VLLM_XLA_CACHE_PATH: its
    default is a home directory; empty turns the cache off).

    Args:
        export_key: Static configuration key for the op, or ``None``.

    Returns:
        Configuration callback for ``sharded_jax_op``, or ``None``.
    """
    vllm_envs = importlib.import_module("vllm.envs")

    if (
        not export_key
        or not vllm_envs.is_set("VLLM_XLA_CACHE_PATH")
        or not vllm_envs.VLLM_XLA_CACHE_PATH
    ):
        return None
    root = pathlib.Path(vllm_envs.VLLM_XLA_CACHE_PATH) / "k3_export"

    def configure(op: _JaxCallableProtocol) -> None:
        """Attach export recording and ``_ExportCache`` dispatch to ``op``.

        Args:
            op: ``JaxCallable`` instance being configured.
        """
        exported = op.exported

        def recording(*args: object, **kwargs: object) -> _LoweredExportProtocol:
            """Record the lowered export object on ``op`` and return it.

            Args:
                *args: Positional arguments forwarded to ``exported``.
                **kwargs: Keyword arguments forwarded to ``exported``.

            Returns:
                Lowered export object returned by ``exported``.
            """
            lowered = exported(*args, **kwargs)
            op._recorded_export = lowered
            return lowered

        op.exported = recording
        op._export_cache = _ExportCache(root, export_key)
        op.__class__ = _cached_class(type(op))

    return configure
