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
# src/vllm_torchtpu/megakernel/bridge.py
"""Running JAX/Pallas megakernels as torch ops inside vLLM's workers.

A megakernel is a pure JAX/Pallas SPMD program over every device; each worker
drives its own device through torch custom ops:

* :func:`kernel_mesh` is the mesh the ops run on, over every device in
  ascending device id (the order Mosaic resolves ``DeviceIdType.MESH`` peers
  in; see ``distributed/ep_mesh.py``), and this worker's position in it. A
  kernel's rank is that position, not vLLM's TP rank.
* :func:`register_op` wraps a ``shard_map`` body as a torch custom op (outer
  ``shard_map`` bodies call shared ``mesh_axis_index`` directly instead of
  ``jax.lax.axis_index``); :func:`register_global_op` does the same for a
  global-view function.
* :func:`upload` / :func:`stream_to_device` turn per-rank host arrays into
  ``[1, ...]`` device tensors (the rank axis the op bodies strip).

No vLLM dependencies: the standalone tools use this module as serving does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import time
import types
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol, cast

import jax
import numpy as np
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.logger import init_logger

from ..common.constants import GIB_BYTES
from . import export_cache
from .config import get_active_kernel_config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "OpRegistrationSpec",
    "RegisteredOp",
    "hbm",
    "kernel_mesh",
    "materialize",
    "op_key_suffix",
    "register_global_op",
    "register_op",
    "report_load_stats",
    "stream_to_device",
    "synchronize_tpu",
    "upload",
]

type RegisteredOp = Callable[..., tuple[torch.Tensor, ...]]
"""Torch custom op returned by :func:`register_op`; returns one tensor per output."""

type _PallasSpec = tuple[str, ...]
"""Positional axis-name tuple accepted by ``torch_tpu`` Pallas type stubs."""


logger = init_logger(__name__)

_MESH: dict[str, tuple[Mesh, int]] = {}
_OP_KEY_SUFFIX_LEN = 8


def kernel_mesh(axis: str, world_size: int | None = None) -> tuple[Mesh, int]:
    """Return the global device mesh on ``axis`` and this worker's kernel rank.

    The returned position is this worker's kernel rank, which need not be its
    vLLM TP rank.

    Args:
        axis: Mesh axis name (for example ``"tp"``).
        world_size: Expected total TPU device count, if enforced.

    Returns:
        Pair of ``(mesh, position)`` for the calling worker.

    Raises:
        RuntimeError: If visible devices do not match ``world_size`` or fail to
            include the worker's global device ID.
    """
    if axis in _MESH:
        cached_mesh, cached_position = _MESH[axis]
        if world_size is not None and cached_mesh.size != world_size:
            msg = (
                f"the megakernel needs all {world_size} TPU devices visible to "
                f"every worker; cached mesh '{axis}' has {cached_mesh.size}"
            )
            raise RuntimeError(msg)
        return cached_mesh, cached_position
    devices = sorted(jax.devices(), key=lambda dev: int(dev.id))
    if world_size is not None and len(devices) != world_size:
        msg = (
            f"the megakernel needs all {world_size} TPU devices visible to "
            f"every worker; this worker sees {len(devices)}"
        )
        raise RuntimeError(msg)

    device_id = int(tpu_distributed.global_device_id())
    ids = [int(dev.id) for dev in devices]
    if device_id not in ids:
        msg = f"worker device {device_id} is not among the JAX devices {ids}"
        raise RuntimeError(msg)
    position = ids.index(device_id)
    _MESH[axis] = (Mesh(np.asarray(devices), axis_names=(axis,)), position)
    if position == 0:
        logger.info("megakernel: kernel mesh '%s' over devices %s", axis, ids)
    return _MESH[axis]


def op_key_suffix(key: object) -> str:
    """Return a stable 8-character hex suffix derived from ``repr(key)``.

    Disambiguates process-global torch custom op names whose traced bodies vary
    with static configuration, pool layout, or device mapping. Assumes one
    active ``KernelConfig`` per process; export keys additionally append the
    active kernel config digest via :func:`_keyed_on_kernel_config`.

    Args:
        key: Canonical configuration or layout descriptor.

    Returns:
        Eight lowercase hexadecimal characters of the SHA-256 digest of
        ``repr(key)``.
    """
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:_OP_KEY_SUFFIX_LEN]


def _keyed_on_kernel_config(export_key: str | None) -> str | None:
    """Append the active kernel config digest to ``export_key``.

    Its settings are baked into every trace.

    Args:
        export_key: Base export cache key, or ``None`` if caching is disabled.

    Returns:
        Composite export cache key including the active kernel config digest.
    """
    config = get_active_kernel_config()
    if export_key is None or config is None:
        return export_key
    return f"{export_key}|config={config.key()}"


def _validate_partition_specs(in_specs: object, out_specs: object) -> None:
    """Validate that ``in_specs`` and ``out_specs`` hold ``PartitionSpec`` entries.

    Args:
        in_specs: Positional input sharding sequence.
        out_specs: Output sharding sequence.

    Raises:
        TypeError: If ``in_specs`` or ``out_specs`` is a bare ``PartitionSpec``, is
            not a ``Sequence``, or contains a non-``PartitionSpec`` entry.
        ValueError: If any entry in ``in_specs`` or ``out_specs`` is ``None``.
    """
    if isinstance(in_specs, PartitionSpec) or isinstance(out_specs, PartitionSpec):
        msg = "partition specs must be a sequence, not a bare PartitionSpec."
        raise TypeError(msg)
    if not isinstance(in_specs, Sequence) or not isinstance(out_specs, Sequence):
        msg = "partition specs must be a sequence of PartitionSpec instances."
        raise TypeError(msg)
    entries = (*in_specs, *out_specs)
    if any(entry is None for entry in entries):
        msg = (
            "partition spec entries cannot be None; "
            "pass PartitionSpec() for replicated tensors"
        )
        raise ValueError(msg)
    if not all(isinstance(entry, PartitionSpec) for entry in entries):
        msg = "partition spec entries must be PartitionSpec instances."
        raise TypeError(msg)


@dataclasses.dataclass(frozen=True, slots=True)
class OpRegistrationSpec:
    """Partitioning, buffer donation, and export-cache contract for a custom op.

    Attributes:
        in_specs: Input ``PartitionSpec`` tuple (one per positional array input).
        out_specs: Output ``PartitionSpec`` tuple (one per returned array shard).
        donate_argnums: Zero-based input indices donated in-place to XLA.
        export_key: Static export-cache key prefix, or ``None`` to bypass cache.
    """

    in_specs: tuple[PartitionSpec, ...]
    out_specs: tuple[PartitionSpec, ...]
    donate_argnums: tuple[int, ...] = ()
    export_key: str | None = None

    def __post_init__(self) -> None:
        """Validate input and output partition spec sequences on construction."""
        _validate_partition_specs(self.in_specs, self.out_specs)


def _register_sharded_op(
    name: str,
    fn: Callable[..., object],
    mesh: Mesh,
    spec: OpRegistrationSpec,
    configure_callable: Callable[[export_cache.JaxCallableProtocol], None]
    | None = None,
) -> RegisteredOp:
    """Register a Kimi-K3 SPMD JAX function as a torch custom op over ``mesh``.

    Attaches ``out_shardings`` to ``jax.jit`` so exported shardings survive on
    ``lowered._out_named_shardings`` for ``export_cache``, passes one
    ``PartitionSpec`` per output to ``pallas_impl.JaxCallable``, and applies an
    optional K3-local ``configure_callable`` hook before custom-op registration.

    Args:
        name: Qualified torch custom op name ``"namespace::op"``.
        fn: Annotated JAX callable taking positional ``jax.Array`` inputs.
        mesh: Global JAX device mesh over which ``fn`` executes.
        spec: Partitioning, buffer-donation, and export-key contract.
        configure_callable: Optional callback mutating ``JaxCallable`` prior to
            ``torch.library.custom_op`` registration.

    Returns:
        Registered torch custom op callable.

    Raises:
        TypeError: If ``spec.in_specs`` or ``spec.out_specs`` is not a sequence
            of ``PartitionSpec`` instances.
        ValueError: If ``name`` lacks ``"namespace::op"`` format or a spec
            entry is ``None``.
    """
    _validate_partition_specs(spec.in_specs, spec.out_specs)
    namespace_part, sep, op_part = name.partition("::")
    if not sep or not namespace_part or not op_part or "::" in op_part:
        msg = f"Op name {name} must be 'namespace::op', with one '::'."
        raise ValueError(msg)

    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    in_specs = tuple(spec.in_specs)
    out_specs = tuple(spec.out_specs)
    donate_argnums = spec.donate_argnums or None
    pallas_in_specs = cast("tuple[_PallasSpec, ...]", in_specs)
    pallas_out_specs = cast("tuple[_PallasSpec, ...]", out_specs)
    out_shardings = tuple(NamedSharding(mesh, item) for item in out_specs)

    jit_fn = jax.jit(
        fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        out_shardings=out_shardings,
        keep_unused=True,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums,
            "output_partition_specs": str(out_specs),
        },
    )
    wrapped_fn = pallas_impl.JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=pallas_in_specs,
        output_partition_specs=pallas_out_specs,
        static_argnums=static_argnums,
        donate_argnums=list(donate_argnums) if donate_argnums is not None else None,
    )
    if configure_callable is not None:
        configure_callable(cast("export_cache.JaxCallableProtocol", wrapped_fn))

    op = torch.library.custom_op(name, wrapped_fn, mutates_args=())

    def fake_fn(*args: object, **kwargs: object) -> object:
        """Trace the exported JAX function to size sharded torch placeholders.

        Args:
            *args: Positional FakeTensor or placeholder inputs to ``op``.
            **kwargs: Keyword arguments forwarded to ``wrapped_fn.exported``.

        Returns:
            Unflattened PyTree of per-rank ``torch.Tensor`` placeholders.
        """
        jax_args = pallas_impl.jax_placeholders(
            cast("Sequence[torch.Tensor]", args),
            mesh=mesh,
            partition_specs=pallas_in_specs,
        )
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = wrapped_fn.exported(*jax_args, **kwargs)
        return lowered.out_tree.unflatten(
            pallas_impl.torch_placeholder(aval, mesh=mesh, partition_spec=out_spec)
            for aval, out_spec in zip(lowered.out_avals, pallas_out_specs, strict=True)
        )

    op.register_fake(fake_fn)
    return cast("RegisteredOp", op)


class _AnnotatedCallable(Protocol):
    """Protocol for a Python function carrying dynamic annotations and signature."""

    __annotations__: dict[str, object]
    __signature__: inspect.Signature

    def __call__(self, *args: jax.Array) -> object:
        """Execute the underlying wrapped callable."""
        ...


_ITER_DONE = object()


def _build_typed_op(
    traced: Callable[..., object],
    *,
    in_count: int,
    out_count: int,
) -> Callable[..., object]:
    """Attach an explicit positional ``jax.Array`` signature to ``traced``.

    ``torch_tpu`` derives the custom op schema from annotated positional
    parameters; constructing ``inspect.Signature`` and ``__annotations__``
    directly avoids dynamic ``exec()`` code generation.

    Args:
        traced: Underlying traced callable accepting positional JAX arrays.
        in_count: Number of input ``jax.Array`` parameters.
        out_count: Number of output ``jax.Array`` elements in the return tuple.

    Returns:
        Wrapper callable carrying ``__signature__`` and ``__annotations__``.
    """

    def op(*args: jax.Array) -> object:
        """Execute the traced JAX callable over positional array inputs.

        Args:
            *args: Positional ``jax.Array`` inputs matching ``in_specs``.

        Returns:
            Tuple of output ``jax.Array`` shards.
        """
        return traced(*args)

    param_names = [f"arg_{index}" for index in range(in_count)]
    parameters = [
        inspect.Parameter(
            param_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=jax.Array,
        )
        for param_name in param_names
    ]
    return_type = types.GenericAlias(tuple, (jax.Array,) * out_count)
    annotations: dict[str, object] = dict.fromkeys(param_names, jax.Array)
    annotations["return"] = return_type
    annotated_op = cast("_AnnotatedCallable", op)
    annotated_op.__annotations__ = annotations
    annotated_op.__signature__ = inspect.Signature(
        parameters,
        return_annotation=return_type,
    )
    return op


def register_op(
    name: str,
    mesh: Mesh,
    body: Callable[..., object],
    spec: OpRegistrationSpec,
) -> RegisteredOp:
    """Register a torch custom op running ``shard_map(body)`` over ``mesh``.

    ``torch_tpu`` derives the op schema from annotated positional parameters, so
    the wrapper attaches one ``jax.Array`` parameter per input. Passing an
    :class:`OpRegistrationSpec` with ``export_key`` opts the op into the
    persistent export cache.

    Args:
        name: Custom op registration name.
        mesh: Global JAX device mesh.
        body: Per-shard function wrapped with ``jax.shard_map``.
        spec: Partitioning, buffer-donation, and export-key contract.

    Returns:
        Registered torch custom op callable.

    Raises:
        TypeError: If ``spec`` is not an :class:`OpRegistrationSpec`.
    """
    if not isinstance(spec, OpRegistrationSpec):
        msg = f"spec must be an OpRegistrationSpec, got {type(spec).__name__}"
        raise TypeError(msg)
    mapped = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=spec.in_specs,
        out_specs=spec.out_specs,
        check_vma=False,
    )
    typed_op = _build_typed_op(
        mapped,
        in_count=len(spec.in_specs),
        out_count=len(spec.out_specs),
    )
    return _register_sharded_op(
        name,
        typed_op,
        mesh,
        spec,
        export_cache.hook(_keyed_on_kernel_config(spec.export_key)),
    )


def register_global_op(
    name: str,
    mesh: Mesh,
    fn: Callable[..., object],
    spec: OpRegistrationSpec,
) -> RegisteredOp:
    """Register a torch custom op for a global-view JAX function over ``mesh``.

    ``fn`` sees global arrays (``spec.in_specs`` describe how each worker's
    tensor is its shard) and may call the kernels' own ``jit(shard_map(...))``
    programs.

    Args:
        name: Custom op registration name.
        mesh: Global JAX device mesh.
        fn: Global-view JAX function.
        spec: Partitioning, buffer-donation, and export-key contract.

    Returns:
        Registered torch custom op callable.

    Raises:
        TypeError: If ``spec`` is not an :class:`OpRegistrationSpec`.
    """
    if not isinstance(spec, OpRegistrationSpec):
        msg = f"spec must be an OpRegistrationSpec, got {type(spec).__name__}"
        raise TypeError(msg)
    typed_op = _build_typed_op(
        fn,
        in_count=len(spec.in_specs),
        out_count=len(spec.out_specs),
    )
    return _register_sharded_op(
        name,
        typed_op,
        mesh,
        spec,
        export_cache.hook(_keyed_on_kernel_config(spec.export_key)),
    )


def upload(value: np.ndarray) -> torch.Tensor:
    """Upload a host NumPy array as a ``[1, ...]`` TPU tensor.

    Args:
        value: Host NumPy shard array (bfloat16 arrays are reinterpreted via uint16).

    Returns:
        TPU ``torch.Tensor`` with a leading rank axis of size ``1``.
    """
    value = np.ascontiguousarray(value)
    if value.dtype.name == "bfloat16":
        tensor = torch.from_numpy(value.view(np.uint16)).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(value)
    return tensor.unsqueeze(0).to("tpu")


UPLOAD_THREADS = 4
UPLOAD_INFLIGHT_BYTES = 16 * GIB_BYTES


def _part_of(name: str) -> tuple[str, int, int] | None:
    """Parse a multi-part weight name ``family@index/count`` if present.

    Args:
        name: Weight tensor identifier string.

    Returns:
        Tuple of ``(family, index, count)`` for split expert tensors, else ``None``.
    """
    if "@" not in name:
        return None
    family, part = name.split("@")
    index, count = part.split("/")
    return family, int(index), int(count)


def stream_to_device(
    iterator: Iterable[tuple[str, str, np.ndarray]],
    store: Callable[[str, str, torch.Tensor], None],
) -> dict[str, float]:
    """Upload ``(group, name, array)`` tuples concurrently and pass them to ``store``.

    Uploads run concurrently on ``UPLOAD_THREADS`` threads with at most that
    many arrays and ``UPLOAD_INFLIGHT_BYTES`` of host buffers pending. Arrays
    named ``family@index/count`` (``rank_loader.iter_*_rank``'s expert files)
    are joined on the device along the layer axis once all parts are up.

    Args:
        iterator: Iterable yielding ``(group, name, host_array)`` triples.
        store: Callback ``store(group, name, device_tensor)`` receiving uploaded
            tensors.

    Returns:
        Dictionary of ``wall``, ``wait``, ``upload`` seconds and ``gib``
        transferred.

    Raises:
        RuntimeError: If any split ``family@index/count`` tensor is incomplete
            at EOF.
    """
    started = time.perf_counter()
    wait = upload_wait = 0.0
    total = 0
    pending: list[tuple[Future[tuple[str, str, torch.Tensor]], int]] = []
    parts: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def finish(future: Future[tuple[str, str, torch.Tensor]]) -> None:
        """Commit one completed upload future to ``store`` or the split-part table.

        Args:
            future: Completed upload future returning ``(group, name, tensor)``.
        """
        group, name, tensor = future.result()
        part = _part_of(name)
        if part is None:
            store(group, name, tensor)
            return
        family, index, count = part
        collected = parts.setdefault((group, family), {})
        collected[index] = tensor
        if len(collected) == count:
            joined = torch.cat([collected[idx] for idx in range(count)], dim=1)
            del parts[(group, family)]
            collected.clear()
            store(group, family, joined)

    def drain(limit_count: int, limit_bytes: int) -> float:
        """Drain oldest pending uploads until count and byte limits are satisfied.

        Args:
            limit_count: Maximum number of pending futures to leave in flight.
            limit_bytes: Maximum total pending bytes to leave in flight.

        Returns:
            Wall-clock seconds spent waiting for drained uploads.
        """
        before = time.perf_counter()
        while pending and (
            len(pending) > limit_count or sum(n for _, n in pending) > limit_bytes
        ):
            future, _ = pending.pop(0)
            finish(future)
        return time.perf_counter() - before

    with ThreadPoolExecutor(UPLOAD_THREADS) as pool:
        it = iter(iterator)
        while True:
            before = time.perf_counter()
            item = next(it, _ITER_DONE)
            wait += time.perf_counter() - before
            if item is _ITER_DONE:
                break
            group, name, value = cast("tuple[str, str, np.ndarray]", item)
            nbytes = int(value.nbytes)
            total += nbytes
            upload_wait += drain(
                UPLOAD_THREADS - 1, max(0, UPLOAD_INFLIGHT_BYTES - nbytes)
            )
            pending.append(
                (
                    pool.submit(
                        lambda grp=group, nm=name, arr=value: (grp, nm, upload(arr))
                    ),
                    nbytes,
                )
            )
            del item, value
        upload_wait += drain(0, 0)
    if parts:
        msg = f"megakernel weights: incomplete parts {sorted(parts)}"
        raise RuntimeError(msg)
    return {
        "wall": time.perf_counter() - started,
        "wait": wait,
        "upload": upload_wait,
        "gib": total / GIB_BYTES,
    }


def report_load_stats(label: str, position: int, stats: dict[str, float]) -> None:
    """Gather all ranks' weight-load statistics and log min/median/max on rank 0.

    Args:
        label: Role or weight-group label for the log line.
        position: Calling worker's position in the kernel mesh.
        stats: Timing and byte statistics returned by :func:`stream_to_device`.
    """
    gathered: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, (position, stats))
    if position != 0:
        return
    typed_rows: list[tuple[int, dict[str, float]]] = [
        item for item in gathered if isinstance(item, tuple)
    ]
    rows = sorted(typed_rows, key=lambda item: item[1]["wall"])

    def spread(key: str) -> str:
        """Format min/median/max across ranks for metric ``key``.

        Args:
            key: Metric key in ``stats`` (``"wall"``, ``"wait"``, or ``"upload"``).

        Returns:
            Slash-separated ``min/median/max`` string in seconds.
        """
        values = sorted(item[1][key] for item in rows)
        return f"{values[0]:.1f}/{values[len(values) // 2]:.1f}/{values[-1]:.1f}"

    logger.info(
        "K3 %s weights on %d ranks, min/median/max s: wall %s, "
        "read-wait %s, upload-wait %s; "
        "slowest rank at mesh position %d (%.1f GiB)",
        label,
        len(rows),
        spread("wall"),
        spread("wait"),
        spread("upload"),
        rows[-1][0],
        rows[-1][1]["gib"],
    )


def materialize(tensor: torch.Tensor) -> None:
    """Force ``torch_tpu`` lazy execution up to ``tensor`` (a one-element read).

    Args:
        tensor: Device tensor whose pending lazy graph work should be flushed.
    """
    tensor.reshape(-1)[:1].cpu()


def synchronize_tpu(
    tensors: Sequence[torch.Tensor],
    *,
    wait: bool = False,
) -> None:
    """Synchronize ``tensors`` via ``torch_tpu._internal.sync``.

    Kept local to ``runtime_shim.bridge`` instead of importing
    ``vllm_torchtpu.utils`` so standalone host tools can import this module
    without pulling ``vllm.utils``.

    Args:
        tensors: Device tensors to flush through the ``torch_tpu`` barrier.
        wait: Whether to block the host thread until device execution completes.
    """
    sync_mod = importlib.import_module("torch_tpu._internal.sync")
    sync_mod.synchronize(list(tensors), wait=wait)


def hbm(position: int) -> str:
    """Format current and peak HBM usage on mesh ``position`` for diagnostics.

    Args:
        position: Device position in the sorted JAX device list.

    Returns:
        Formatted GiB usage string, or ``"unavailable (...)"`` if unsupported.
    """
    try:
        device = sorted(jax.devices(), key=lambda dev: int(dev.id))[position]
        stats = device.memory_stats() or {}
        in_use = stats.get("bytes_in_use", 0) / GIB_BYTES
        peak = stats.get("peak_bytes_in_use", 0) / GIB_BYTES
        limit = stats.get("bytes_limit", 0) / GIB_BYTES
    except (
        ImportError,
        AttributeError,
        RuntimeError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        return f"unavailable ({exc})"
    return f"{in_use:.1f} GiB in use, peak {peak:.1f} of {limit:.1f}"
