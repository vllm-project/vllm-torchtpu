# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
import threading
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax._src.pallas import core as pallas_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import export_cache
from vllm_torchtpu.megakernel.config import get_active_kernel_config

logger = init_logger(__name__)

_MESH: dict[str, tuple[Mesh, int]] = {}
_STOCK_AXIS_INDEX = jax.lax.axis_index
_AXIS_INDEX_LOCK = threading.Lock()
_AXIS_INDEX_DEPTH = 0


def kernel_mesh(axis: str, world_size: int | None = None) -> tuple[Mesh, int]:
    """The mesh the ops run on (all devices in ascending id, on one axis
    named ``axis``) and this worker's position in it: its kernel rank, which
    need not be its vLLM TP rank. ``world_size``: the device count the
    program needs."""
    if axis in _MESH:
        return _MESH[axis]
    devices = sorted(jax.devices(), key=lambda d: int(d.id))
    if world_size is not None and len(devices) != world_size:
        raise RuntimeError(
            f"the megakernel needs all {world_size} TPU devices visible to "
            f"every worker; this worker sees {len(devices)}"
        )
    from torch_tpu._internal.distributed import tpu_distributed

    device_id = int(tpu_distributed.global_device_id())
    ids = [int(d.id) for d in devices]
    if device_id not in ids:
        raise RuntimeError(
            f"worker device {device_id} is not among the JAX devices {ids}"
        )
    position = ids.index(device_id)
    _MESH[axis] = (Mesh(np.asarray(devices), axis_names=(axis,)), position)
    if position == 0:
        logger.info("megakernel: kernel mesh '%s' over devices %s", axis, ids)
    return _MESH[axis]


@contextlib.contextmanager
def spmd_safe_axis_index():
    """Route XLA-level ``jax.lax.axis_index`` through Pallas while active.

    Outside a Pallas body ``axis_index`` lowers to an HLO ``partition-id``,
    which torch_tpu's SPMD partitioner rejects; such calls become a one-scalar
    Pallas kernel reading the device's position on the axis. Calls inside a
    Pallas kernel keep the stock primitive.
    """

    def axis_index(axis_name):
        # A Pallas kernel body is being traced (``tracing_grid_env`` sets
        # ``grid_context``; ``core_map`` kernels too) or lowered/interpreted
        # (``grid_env`` pushes the stack).
        if (
            pallas_core._pallas_tracing_env.grid_context is not None
            or pallas_core.current_grid_env() is not None
        ):
            return _STOCK_AXIS_INDEX(axis_name)

        def kernel(out_ref):
            out_ref[0] = _STOCK_AXIS_INDEX(axis_name).astype(jnp.int32)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
            name="spmd_safe_axis_index",
        )()[0]

    global _AXIS_INDEX_DEPTH
    with _AXIS_INDEX_LOCK:
        if _AXIS_INDEX_DEPTH == 0:
            jax.lax.axis_index = axis_index
        _AXIS_INDEX_DEPTH += 1
    try:
        yield
    finally:
        with _AXIS_INDEX_LOCK:
            _AXIS_INDEX_DEPTH -= 1
            if _AXIS_INDEX_DEPTH == 0:
                jax.lax.axis_index = _STOCK_AXIS_INDEX


def _keyed_on_kernel_config(export_key: str | None) -> str | None:
    """``export_key`` with the active kernel config's key: its settings are
    baked into every trace."""
    config = get_active_kernel_config()
    if export_key is None or config is None:
        return export_key
    return f"{export_key}|config={config.key()}"


def register_op(
    name: str,
    mesh: Mesh,
    body: Callable[..., Any],
    in_specs: tuple,
    out_specs: tuple,
    donate_argnums: tuple[int, ...] = (),
    export_key: str | None = None,
) -> Any:
    """A torch custom op running ``shard_map(body)`` over ``mesh``.

    torch_tpu derives the op schema from annotated positional parameters, so
    the wrapper spells out one ``jax.Array`` parameter per input.
    ``export_key``: the static configuration ``body`` bakes into its trace,
    opting the op into the persistent export cache (``export_cache``).
    """
    mapped = jax.shard_map(
        body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
    count = len(in_specs)

    def traced(*args):
        with spmd_safe_axis_index():
            return mapped(*args)

    params = ", ".join(f"a{i}: jax.Array" for i in range(count))
    call = ", ".join(f"a{i}" for i in range(count))
    returns = ", ".join("jax.Array" for _ in out_specs)
    namespace: dict[str, Any] = {"jax": jax, "_traced": traced}
    exec(  # noqa: S102 -- fixed template, no external input
        f"def op({params}) -> tuple[{returns}]:\n    return _traced({call})\n",
        namespace,
    )
    return sharded_jax_op(
        name,
        namespace["op"],
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
        donate_argnums=donate_argnums or None,
        configure_callable=export_cache.hook(_keyed_on_kernel_config(export_key)),
    )


def register_global_op(
    name: str,
    mesh: Mesh,
    fn: Callable[..., Any],
    in_specs: tuple,
    out_specs: tuple,
    donate_argnums: tuple[int, ...] = (),
    export_key: str | None = None,
) -> Any:
    """A torch custom op for a *global-view* JAX function over ``mesh``.

    ``fn`` sees global arrays (``in_specs`` say how each worker's tensor is
    its shard) and may call the kernels' own ``jit(shard_map(...))``
    programs; it is traced under ``spmd_safe_axis_index`` like :func:`register_op`.
    """
    count = len(in_specs)

    def traced(*args):
        with spmd_safe_axis_index():
            return fn(*args)

    params = ", ".join(f"a{i}: jax.Array" for i in range(count))
    call = ", ".join(f"a{i}" for i in range(count))
    returns = ", ".join("jax.Array" for _ in out_specs)
    namespace: dict[str, Any] = {"jax": jax, "_traced": traced}
    exec(  # noqa: S102 -- fixed template, no external input
        f"def op({params}) -> tuple[{returns}]:\n    return _traced({call})\n",
        namespace,
    )
    return sharded_jax_op(
        name,
        namespace["op"],
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
        donate_argnums=donate_argnums or None,
        configure_callable=export_cache.hook(_keyed_on_kernel_config(export_key)),
    )


def upload(value: np.ndarray) -> torch.Tensor:
    """Host array -> ``[1, ...]`` device tensor (bf16 via its uint16 view)."""
    value = np.ascontiguousarray(value)
    if value.dtype.name == "bfloat16":
        tensor = torch.from_numpy(value.view(np.uint16)).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(value)
    return tensor.unsqueeze(0).to("tpu")


UPLOAD_THREADS = 4
UPLOAD_INFLIGHT_BYTES = 16 * 2**30


def _part_of(name: str) -> tuple[str, int, int] | None:
    if "@" not in name:
        return None
    family, part = name.split("@")
    index, count = part.split("/")
    return family, int(index), int(count)


def stream_to_device(iterator, store) -> dict[str, float]:
    """Upload every ``(group, name, array)`` the kernel's rank iterator yields
    and hand each device tensor to ``store(group, name, tensor)``.

    Uploads run concurrently on ``UPLOAD_THREADS`` threads with at most that
    many arrays and ``UPLOAD_INFLIGHT_BYTES`` of host buffers pending. Arrays
    named ``family@index/count`` (``rank_loader.iter_*_rank(split_experts=)``)
    are joined on the device along the layer axis once all parts are up.
    Returns wall time, time blocked on the reader, time blocked on uploads
    and the GiB moved.
    """
    from concurrent.futures import ThreadPoolExecutor

    started = time.perf_counter()
    wait = upload_wait = 0.0
    total = 0
    pending: list[Any] = []  # (future, nbytes)
    parts: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def finish(future) -> None:
        group, name, tensor = future.result()
        part = _part_of(name)
        if part is None:
            store(group, name, tensor)
            return
        family, index, count = part
        collected = parts.setdefault((group, family), {})
        collected[index] = tensor
        if len(collected) == count:
            joined = torch.cat([collected[i] for i in range(count)], dim=1)
            del parts[(group, family)]
            collected.clear()
            store(group, family, joined)

    def drain(limit_count: int, limit_bytes: int) -> float:
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
            try:
                group, name, value = next(it)
            except StopIteration:
                wait += time.perf_counter() - before
                break
            wait += time.perf_counter() - before
            nbytes = int(value.nbytes)
            total += nbytes
            upload_wait += drain(
                UPLOAD_THREADS - 1, max(0, UPLOAD_INFLIGHT_BYTES - nbytes)
            )
            pending.append(
                (
                    pool.submit(lambda g=group, n=name, v=value: (g, n, upload(v))),
                    nbytes,
                )
            )
            del value
        upload_wait += drain(0, 0)
    if parts:
        raise RuntimeError(f"megakernel weights: incomplete parts {sorted(parts)}")
    return {
        "wall": time.perf_counter() - started,
        "wait": wait,
        "upload": upload_wait,
        "gib": total / 2**30,
    }


def report_load_stats(label: str, position: int, stats: dict[str, float]) -> None:
    """Gather every rank's weight-load stats; mesh position 0 logs
    min/median/max (the default process group: one per device)."""
    gathered: list[Any] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, (position, stats))
    if position != 0:
        return
    rows = sorted(gathered, key=lambda item: item[1]["wall"])

    def spread(key):
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
    """Force torch_tpu's lazy work up to ``tensor`` (a one-element read)."""
    tensor.reshape(-1)[:1].cpu()


def hbm(position: int) -> str:
    try:
        device = sorted(jax.devices(), key=lambda d: int(d.id))[position]
        stats = device.memory_stats() or {}
        return "{:.1f} GiB in use, peak {:.1f} of {:.1f}".format(
            stats.get("bytes_in_use", 0) / 2**30,
            stats.get("peak_bytes_in_use", 0) / 2**30,
            stats.get("bytes_limit", 0) / 2**30,
        )
    except Exception as exc:  # noqa: BLE001 -- diagnostics only
        return f"unavailable ({exc})"
