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
import hashlib
import importlib
import importlib.metadata
import operator
import os
import pathlib
import pickle
import time
from typing import TYPE_CHECKING

import jax
import torch
from torch_tpu._internal.pallas import pallas as pallas_impl

import vllm_torchtpu
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from jax.sharding import Mesh, PartitionSpec

logger = init_logger(__name__)

_EXPORT_ENV_PREFIXES = ("JAX_",)
_SOURCE_FINGERPRINT: str | None = None
_PICKLE_LOADS: Callable[[bytes], dict[str, object]] = vars(pickle)["loads"]
_SHAPE_AND_DTYPE: Callable[[object], tuple[Iterable[int], object]] = (
    operator.attrgetter("shape", "dtype")
)
_EXPORTED_GETTER: Callable[[object], Callable[..., object]] = operator.attrgetter(
    "exported"
)


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
    pallas_attrs = vars(pallas_impl)
    dtype_map: dict[object, torch.dtype] = pallas_attrs["JAX_TO_TORCH_DTYPE_MAP"]
    torch_dtype = dtype_map.get(aval.dtype)
    if torch_dtype is None:
        msg = f"Unsupported dtype for pallas kernels: {aval.dtype}"
        raise NotImplementedError(msg)
    local_shape_fn = pallas_attrs["get_local_shape"]
    shape = tuple(int(dim) for dim in local_shape_fn(aval.shape, mesh, spec))
    fp4_x2_dtype = vars(torch).get("float4_e2m1fn_x2")
    if fp4_x2_dtype is not None and torch_dtype == fp4_x2_dtype and shape:
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


def _source_fingerprint() -> str:
    """Compute and memoize a Blake2b digest over ``vllm_torchtpu`` sources and versions.

    Returns:
        32-hex-character Blake2b digest string.
    """
    cached = _SOURCE_FINGERPRINT
    if cached is None:
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
        cached = digest.hexdigest()
        globals()["_SOURCE_FINGERPRINT"] = cached
    return cached


def _process_rank() -> int:
    """Return the active distributed worker rank, falling back to ``jax.process_index``.

    Returns:
        Zero-based worker rank integer.
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except RuntimeError as exc:
        logger.debug("export cache: rank lookup fallback to jax.process_index: %s", exc)
    return int(jax.process_index())


class _ExportCache:
    """Persistent on-disk MLIR export cache for one registered custom op."""

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
        op: object,
        args: tuple[object, ...],
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
        op_attrs = vars(op)
        static_argnums = op_attrs.get("static_argnums", ())
        signature: list[object] = []
        for index, arg in enumerate(args):
            if index in static_argnums:
                signature.append(("static", repr(arg)))
            elif arg is None:
                signature.append(None)
            else:
                raw_shape, raw_dtype = _SHAPE_AND_DTYPE(arg)
                signature.append((tuple(raw_shape), str(raw_dtype)))
        env = sorted(
            (k, v) for k, v in os.environ.items() if k.startswith(_EXPORT_ENV_PREFIXES)
        )
        op_mesh = op_attrs.get("mesh")
        mesh = (
            (
                tuple(int(dev.id) for dev in op_mesh.devices.flat),
                tuple(op_mesh.axis_names),
            )
            if op_mesh
            else None
        )
        op_name = str(op_attrs["name"])
        key = repr(
            (
                op_name,
                self.export_key,
                signature,
                sorted((k, repr(v)) for k, v in kwargs.items()),
                repr(op_attrs.get("input_partition_specs")),
                repr(op_attrs.get("output_partition_specs")),
                tuple(op_attrs.get("donate_argnums", ())),
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
        op: object,
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
        op_attrs = vars(op)
        get_key = vars(pallas_impl)["_get_kernel_invocation_key"]
        kernel_key = get_key(
            op_attrs["trace_key"],
            args,
            kwargs,
            op_attrs["static_argnums"],
        )
        output_shapes: dict[object, object] = op_attrs["output_shapes"]
        if kernel_key in output_shapes:
            return call(op, *args, **kwargs)
        cache_path = self.path(op, args, kwargs)
        if self._load(op, kernel_key, cache_path):
            return call(op, *args, **kwargs)
        op_attrs["_recorded_export"] = None
        result = call(op, *args, **kwargs)
        lowered = op_attrs.get("_recorded_export")
        fingerprints: dict[object, str] = op_attrs["kernel_key_to_mlir_fingerprint"]
        fingerprint = fingerprints.get(kernel_key)
        if lowered is not None and fingerprint is not None:
            try:
                self._store(cache_path, fingerprint, lowered, op=op)
            except OSError as exc:
                logger.warning("export cache: could not store %s: %s", cache_path, exc)
        return result

    def _load(self, op: object, kernel_key: object, path: pathlib.Path) -> bool:
        """Load a cached MLIR module and output placeholders from ``path``.

        Args:
            op: Registered ``JaxCallable`` instance to populate.
            kernel_key: Invocation key for the current call signature.
            path: Path to the ``.pkl`` metadata file.

        Returns:
            ``True`` when the cached export was loaded, ``False`` on cache miss.
        """
        try:
            entry = _PICKLE_LOADS(path.read_bytes())
            fingerprint = str(entry["fingerprint"])
            module = (self.root / f"module-{fingerprint}.mlirbc").read_bytes()
            raw_outputs = entry["outputs"]
            out_tree = entry["out_tree"]
        except (OSError, EOFError, pickle.UnpicklingError, KeyError, TypeError):
            return False
        if not isinstance(raw_outputs, list):
            return False
        started = time.perf_counter()
        op_attrs = vars(op)
        op_name = str(op_attrs["name"])
        op_mesh = op_attrs.get("mesh")
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
        fingerprints: dict[object, str] = op_attrs["kernel_key_to_mlir_fingerprint"]
        output_shapes: dict[object, object] = op_attrs["output_shapes"]
        fingerprints[kernel_key] = fingerprint
        output_shapes[kernel_key] = (outputs, out_tree)
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
        lowered: object,
        *,
        op: object = None,
    ) -> None:
        """Persist lowered MLIR bytecode and output shape metadata to ``self.root``.

        Args:
            path: Target ``.pkl`` metadata path.
            fingerprint: MLIR module fingerprint string.
            lowered: Lowered JAX export object carrying ``out_avals`` and MLIR bytes.
            op: Optional ``JaxCallable`` providing fallback ``output_partition_specs``.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        lowered_attrs = vars(lowered)
        out_avals: Sequence[jax.core.ShapedArray] = lowered_attrs["out_avals"]
        shardings = lowered_attrs.get("_out_named_shardings")
        if shardings is None or len(shardings) != len(out_avals):
            shardings = (None,) * len(out_avals)
        op_specs = vars(op).get("output_partition_specs") if op is not None else None
        if op_specs is None or len(op_specs) != len(out_avals):
            op_specs = (None,) * len(out_avals)
        outputs = []
        for aval, sharding, fallback_spec in zip(
            out_avals, shardings, op_specs, strict=True
        ):
            spec = vars(sharding).get("spec") if sharding is not None else None
            if spec is None:
                spec = fallback_spec
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
            mlir_bytes = bytes(lowered_attrs["mlir_module_serialized"])
            _atomic_write(module_path, mlir_bytes)
        _atomic_write(
            path,
            pickle.dumps(
                {
                    "fingerprint": fingerprint,
                    "outputs": outputs,
                    "out_tree": lowered_attrs["out_tree"],
                }
            ),
        )


def prepare_distributed(calls: Sequence[object]) -> None:
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
    ops: list[object] = [vars(vars(call)["op"]).get("_init_fn") for call in calls]
    eligible = all(
        op is not None
        and vars(op).get("_export_cache") is not None
        and vars(op).get("mesh") is not None
        and vars(op)["mesh"].size == world
        for op in ops
    )
    signatures = (
        [
            (
                vars(op)["_export_cache"]
                .path(op, vars(call)["operands"], {}, rank=index % world)
                .name,
                sorted(vars(call)["options"].items()),
            )
            for index, (op, call) in enumerate(zip(ops, calls, strict=True))
        ]
        if eligible
        else None
    )
    first_cache: _ExportCache | None = (
        vars(ops[0]).get("_export_cache") if eligible and ops and ops[0] else None
    )
    if first_cache is not None and first_cache.root is not None:
        with contextlib.suppress(OSError):
            first_cache.root.mkdir(parents=True, exist_ok=True)
            (first_cache.root / f".peer-{rank}").write_bytes(b"")
    gathered: list[object] = [None] * world
    torch.distributed.all_gather_object(gathered, signatures)
    if any(signature is None for signature in gathered):
        return
    if any(signature != signatures for signature in gathered):
        msg = "distributed export requires identical program signatures"
        raise ValueError(msg)

    cache_root = first_cache.root if first_cache is not None else None
    peers = (
        [peer for peer in range(world) if (cache_root / f".peer-{peer}").exists()]
        if cache_root is not None
        else []
    ) or list(range(world))
    for index, op in enumerate(ops):
        cache_obj: _ExportCache = vars(op)["_export_cache"]
        cache_obj.shared_rank = peers[index % len(peers)]
    eager_members = execution_mode.EagerMode.__members__
    eager_mode = (
        eager_members.get("INTERNAL_COMPILE_FX_GRAPH")
        or eager_members["_INTERNAL_COMPILE_FX_GRAPH"]
    )
    failure: Exception | None = None
    try:
        for index, call in enumerate(calls):
            if peers[index % len(peers)] != rank:
                continue
            call_attrs = vars(call)
            operands = call_attrs["operands"]
            call_op = call_attrs["op"]
            inputs = tuple(
                tpu_torch_compile.placeholder_like(tensor) for tensor in operands
            )
            with execution_mode.set_eager_mode(eager_mode):
                call_op(*inputs)
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


def _cached_class(cls: type) -> type:
    """Return a subclass of ``cls`` whose ``__call__`` routes through ``_export_cache``.

    Args:
        cls: Original ``JaxCallable`` class.

    Returns:
        Subclass of ``cls`` dispatching ``__call__`` through ``self._export_cache``.
    """
    cached = _CACHED_CLASSES.get(cls)
    if cached is None:
        call = cls.__call__

        def _invoke_with_cache(
            self: object,
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
            cache_obj: _ExportCache = vars(self)["_export_cache"]
            return cache_obj.call(self, call, args, kwargs)

        cached = type(
            f"{cls.__name__}WithExportCache",
            (cls,),
            {"__call__": _invoke_with_cache},
        )
        _CACHED_CLASSES[cls] = cached
    return cached


def hook(export_key: str | None) -> Callable[[object], None] | None:
    """Build the ``sharded_jax_op(configure_callable=...)`` hook for ``export_key``.

    Returns ``None`` when ``export_key`` is ``None`` or ``VLLM_XLA_CACHE_PATH`` is
    not explicitly set to a non-empty path.

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

    def configure(op: object) -> None:
        """Attach export recording and ``_ExportCache`` dispatch to ``op``.

        Args:
            op: ``JaxCallable`` instance being configured.
        """
        op_attrs = vars(op)
        exported = _EXPORTED_GETTER(op)

        def recording(*args: object, **kwargs: object) -> object:
            """Record the lowered export object on ``op`` and return it.

            Args:
                *args: Positional arguments forwarded to ``exported``.
                **kwargs: Keyword arguments forwarded to ``exported``.

            Returns:
                Lowered export object returned by ``exported``.
            """
            lowered = exported(*args, **kwargs)
            op_attrs["_recorded_export"] = lowered
            return lowered

        op_attrs["exported"] = recording
        op_attrs["_export_cache"] = _ExportCache(root, export_key)
        op.__class__ = _cached_class(type(op))

    return configure
