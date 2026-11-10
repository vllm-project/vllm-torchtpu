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
* :func:`register_op` wraps a ``shard_map`` body as a ``sharded_jax_op``,
  traced under ``spmd_safe_axis_index`` so the kernels' XLA-level
  ``lax.axis_index`` survives torch_tpu's SPMD compile;
  :func:`register_global_op` does the same for a global-view function.
* :func:`upload` / :func:`stream_to_device` turn per-rank host arrays into
  ``[1, ...]`` device tensors (the rank axis the op bodies strip).

No vLLM dependencies: the standalone tools use this module as serving does.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import inspect
import threading
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax._src.pallas import core as pallas_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger

from ..common.constants import GIB_BYTES
from . import export_cache
from .config import get_active_kernel_config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

logger = init_logger(__name__)

_MESH: dict[str, tuple[Mesh, int]] = {}
_STOCK_AXIS_INDEX = jax.lax.axis_index


@dataclasses.dataclass
class _AxisIndexState:
    """Tracks nesting depth for :func:`spmd_safe_axis_index`."""

    depth: int = 0


_AXIS_INDEX_LOCK = threading.Lock()
_AXIS_INDEX_STATE = _AxisIndexState()


def kernel_mesh(axis: str, world_size: int | None = None) -> tuple[Mesh, int]:
    """Return the global device mesh on ``axis`` and this worker's position in it.

    The mesh the ops run on (all devices in ascending id, on one axis
    named ``axis``) and this worker's position in it: its kernel rank, which
    need not be its vLLM TP rank. ``world_size``: the device count the
    program needs.

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
        return _MESH[axis]
    devices = sorted(jax.devices(), key=lambda dev: int(dev.id))
    if world_size is not None and len(devices) != world_size:
        msg = (
            f"the megakernel needs all {world_size} TPU devices visible to "
            f"every worker; this worker sees {len(devices)}"
        )
        raise RuntimeError(msg)
    tpu_distributed = importlib.import_module(
        "torch_tpu._internal.distributed.tpu_distributed"
    )

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


class _MutableRef(Protocol):
    """Protocol for a single-element SMEM output reference."""

    def __setitem__(self, index: int, value: object) -> None:
        """Store ``value`` at ``index``."""
        ...


@contextlib.contextmanager
def spmd_safe_axis_index() -> Iterator[None]:
    """Route XLA-level ``jax.lax.axis_index`` through Pallas while active.

    Outside a Pallas body ``axis_index`` lowers to an HLO ``partition-id``,
    which torch_tpu's SPMD partitioner rejects; such calls become a one-scalar
    Pallas kernel reading the device's position on the axis. Calls inside a
    Pallas kernel keep the standard primitive.

    Yields:
        Control to the caller with ``jax.lax.axis_index`` patched.
    """

    def axis_index(axis_name: object) -> jax.Array:
        """Read ``axis_name`` coordinate via SMEM Pallas call when outside a grid.

        Args:
            axis_name: Mesh axis name passed to ``jax.lax.axis_index``.

        Returns:
            Scalar int32 ``jax.Array`` holding this device's coordinate on
            ``axis_name``.
        """
        tracing_env = pallas_core._pallas_tracing_env
        # A Pallas kernel body is being traced (``tracing_grid_env`` sets
        # ``grid_context``; ``core_map`` kernels too) or lowered/interpreted
        # (``grid_env`` pushes the stack).
        if (
            tracing_env is not None and tracing_env.grid_context is not None
        ) or pallas_core.current_grid_env() is not None:
            return _STOCK_AXIS_INDEX(axis_name)

        def kernel(out_ref: _MutableRef) -> None:
            """Write the current axis coordinate into ``out_ref[0]``.

            Args:
                out_ref: Single-element SMEM output reference.
            """
            out_ref[0] = _STOCK_AXIS_INDEX(axis_name).astype(jnp.int32)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
            name="spmd_safe_axis_index",
        )()[0]

    with _AXIS_INDEX_LOCK:
        if _AXIS_INDEX_STATE.depth == 0:
            jax.lax.axis_index = axis_index
        _AXIS_INDEX_STATE.depth += 1
    try:
        yield
    finally:
        with _AXIS_INDEX_LOCK:
            _AXIS_INDEX_STATE.depth -= 1
            if _AXIS_INDEX_STATE.depth == 0:
                jax.lax.axis_index = _STOCK_AXIS_INDEX


def _keyed_on_kernel_config(export_key: str | None) -> str | None:
    """Append the active kernel config digest to ``export_key``.

    ``export_key`` with the active kernel config's key: its settings are
    baked into every trace.

    Args:
        export_key: Base export cache key, or ``None`` if caching is disabled.

    Returns:
        Composite export cache key including the active kernel config digest.
    """
    config = get_active_kernel_config()
    if export_key is None or config is None:
        return export_key
    return f"{export_key}|config={config.key()}"


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


type RegisteredOp = Callable[..., tuple[torch.Tensor, ...]]
"""Torch custom op returned by :func:`register_op`; returns one tensor per output."""


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


def _resolve_op_spec(
    in_specs: OpRegistrationSpec | tuple[PartitionSpec, ...],
    out_specs: tuple[PartitionSpec, ...] | None,
) -> OpRegistrationSpec:
    """Normalize an :class:`OpRegistrationSpec` or ``(in_specs, out_specs)`` pair.

    Args:
        in_specs: Full :class:`OpRegistrationSpec` or input ``PartitionSpec``
            tuple.
        out_specs: Output ``PartitionSpec`` tuple when ``in_specs`` is a tuple,
            or ``None`` when ``in_specs`` is an :class:`OpRegistrationSpec`.

    Returns:
        Resolved :class:`OpRegistrationSpec` instance.

    Raises:
        TypeError: If ``out_specs`` is omitted for a tuple ``in_specs`` or
            supplied alongside an :class:`OpRegistrationSpec`.
    """
    if isinstance(in_specs, OpRegistrationSpec):
        if out_specs is not None:
            msg = "out_specs must be omitted when passing OpRegistrationSpec"
            raise TypeError(msg)
        return in_specs
    if out_specs is None:
        msg = "out_specs is required when in_specs is a PartitionSpec tuple"
        raise TypeError(msg)
    return OpRegistrationSpec(in_specs=in_specs, out_specs=out_specs)


def register_op(
    name: str,
    mesh: Mesh,
    body: Callable[..., object],
    in_specs: OpRegistrationSpec | tuple[PartitionSpec, ...],
    out_specs: tuple[PartitionSpec, ...] | None = None,
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
        in_specs: Full :class:`OpRegistrationSpec` or input ``PartitionSpec``
            tuple.
        out_specs: Output ``PartitionSpec`` tuple when ``in_specs`` is a tuple,
            or ``None`` when ``in_specs`` is an :class:`OpRegistrationSpec`.

    Returns:
        Registered torch custom op callable.
    """
    spec = _resolve_op_spec(in_specs, out_specs)
    mapped = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=spec.in_specs,
        out_specs=spec.out_specs,
        check_vma=False,
    )

    def traced(*args: jax.Array) -> object:
        """Trace ``mapped`` under ``spmd_safe_axis_index``.

        Args:
            *args: Positional ``jax.Array`` inputs.

        Returns:
            Result of ``mapped(*args)``.
        """
        with spmd_safe_axis_index():
            return mapped(*args)

    typed_op = _build_typed_op(
        traced,
        in_count=len(spec.in_specs),
        out_count=len(spec.out_specs),
    )
    return cast(
        "RegisteredOp",
        sharded_jax_op(
            name,
            typed_op,
            mesh=mesh,
            input_partition_specs=spec.in_specs,
            output_partition_specs=spec.out_specs,
            donate_argnums=spec.donate_argnums or None,
            configure_callable=export_cache.hook(
                _keyed_on_kernel_config(spec.export_key)
            ),
        ),
    )


def register_global_op(
    name: str,
    mesh: Mesh,
    fn: Callable[..., object],
    in_specs: OpRegistrationSpec | tuple[PartitionSpec, ...],
    out_specs: tuple[PartitionSpec, ...] | None = None,
) -> RegisteredOp:
    """Register a torch custom op for a global-view JAX function over ``mesh``.

    ``fn`` sees global arrays (``in_specs`` describe how each worker's tensor is
    its shard) and may call the kernels' own ``jit(shard_map(...))`` programs;
    it is traced under ``spmd_safe_axis_index`` like :func:`register_op`.

    Args:
        name: Custom op registration name.
        mesh: Global JAX device mesh.
        fn: Global-view JAX function.
        in_specs: Full :class:`OpRegistrationSpec` or input ``PartitionSpec``
            tuple.
        out_specs: Output ``PartitionSpec`` tuple when ``in_specs`` is a tuple,
            or ``None`` when ``in_specs`` is an :class:`OpRegistrationSpec`.

    Returns:
        Registered torch custom op callable.
    """
    spec = _resolve_op_spec(in_specs, out_specs)

    def traced(*args: jax.Array) -> object:
        """Trace ``fn`` under ``spmd_safe_axis_index``.

        Args:
            *args: Positional ``jax.Array`` inputs.

        Returns:
            Result of ``fn(*args)``.
        """
        with spmd_safe_axis_index():
            return fn(*args)

    typed_op = _build_typed_op(
        traced,
        in_count=len(spec.in_specs),
        out_count=len(spec.out_specs),
    )
    return cast(
        "RegisteredOp",
        sharded_jax_op(
            name,
            typed_op,
            mesh=mesh,
            input_partition_specs=spec.in_specs,
            output_partition_specs=spec.out_specs,
            donate_argnums=spec.donate_argnums or None,
            configure_callable=export_cache.hook(
                _keyed_on_kernel_config(spec.export_key)
            ),
        ),
    )


def upload(value: np.ndarray) -> torch.Tensor:
    """Upload a host NumPy array as a ``[1, ...]`` TPU tensor.

    Host array -> ``[1, ...]`` device tensor (bf16 via its uint16 view).

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
UPLOAD_INFLIGHT_BYTES = 16 * 2**30


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
    # Note Shape (future, nbytes)
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

    Gather every rank's weight-load stats; mesh position 0 logs
    min/median/max (the default process group: one per device).

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
    """Force ``torch_tpu`` lazy execution up to ``tensor`` via a 1-element host read.

    Args:
        tensor: Device tensor whose pending lazy graph work should be flushed.
    """
    tensor.reshape(-1)[:1].cpu()


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
