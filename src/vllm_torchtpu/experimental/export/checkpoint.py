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
"""Saves the weights vLLM serves with as an Orbax checkpoint of an artifact.

An artifact's weights are the ones vLLM serves with, including any it computes
while loading (FP8 requantization, dtype casts). The `checkpoint` and `all`
export modes save every `abstract_params` input `i` as array `array_{i}` of an
Orbax checkpoint in `serving_checkpoint/data`, from the weights of the loaded
model. The arrays are Zarr v2 arrays in a TensorStore OCDBT database, the Orbax
default (`CHECKPOINT_FORMAT`): their chunks are packed into a few large data
files, so a runtime that reads the checkpoint in place over a network file
system pays a few file opens instead of one per chunk. The artifact declares the
format in `serving_config.checkpoint_config`.

One process per tensor-parallel rank (a CPU process for `checkpoint`, the TPU
export worker for `all`) saves its rank's shards. The processes join a JAX
distributed runtime before any JAX use, and the arrays are laid out on a mesh of
their CPU devices, so Orbax writes each array's chunks from the processes that
hold them without copying weights to TPU memory. Orbax writes one chunk per shard,
and splits a replicated array among the processes holding it, so a small array (a
per-layer norm or bias) would become one tiny chunk per rank; an array of at most
`SMALL_ARRAY_BYTES` is saved as one chunk instead. The chunk layout is recorded
in each array's `.zarray`, so readers need no change.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import portpicker

from vllm_torchtpu.experimental.export import weight_mapper
from vllm_torchtpu.experimental.export.trace import trace

# vLLM picks its platform when imported, so CPU workers start with this env.
CPU_ENV = {"VLLM_TARGET_DEVICE": "cpu", "JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "CPU"}
# Worker-only flags the launchers add: the `host:port` of the JAX distributed
# coordinator, and where `all`-mode export workers, each exporting to its own
# directory, save the checkpoint (the final artifact's `CHECKPOINT_DATA_DIR`).
JAX_COORDINATOR_FLAG = "--jax-coordinator"
DATA_DIR_FLAG = "--checkpoint-data-dir"
_AXIS_NAMES = ("data", "model")
# Checkpoint arrays `array_{i}` live here, relative to the artifact directory.
CHECKPOINT_DATA_DIR = os.path.join("serving_checkpoint", "data")
# How the checkpoint is stored, as `serving_config.checkpoint_config.format`
# declares it: Zarr v2 arrays in a TensorStore OCDBT database (Orbax `use_ocdbt`).
CHECKPOINT_FORMAT = "zarr2_ocdbt"
# A sharded array of at most this many bytes (all shards together) is saved as one
# chunk rather than one chunk per shard.
SMALL_ARRAY_BYTES = 4 << 20


def abstract_params_inputs(artifact_dir: str) -> list[dict[str, Any]]:
    """The `abstract_params` inputs of the artifact, in argument order."""
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        meta = json.load(f)
    inputs = meta["functions"]["abstract_params"]["default"]["inputs"]
    return sorted(inputs, key=lambda t: t["arg_index"])


def mesh_axis_sizes(artifact_dir: str) -> dict[str, int]:
    """Size of each mesh axis of the artifact."""
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        mesh = json.load(f)["mesh"]
    return dict(zip(mesh["axis_names"], mesh["mesh_shape"], strict=True))


def shard_shape(spec: Mapping[str, Any], axis_sizes: Mapping[str, int]) -> list[int]:
    """Shape of one device's shard of the global tensor `spec`."""
    shape = []
    for dim, axis in zip(spec["shape"], spec["sharding"], strict=True):
        axes = [] if axis is None else [axis] if isinstance(axis, str) else list(axis)
        div = 1
        for a in axes:
            div *= axis_sizes[a]
        shape.append(int(dim) // div)
    return shape


def local_tensors(model: Any) -> dict[str, Any]:
    """The model's parameters and buffers by name, its drafts' own included
    under their `mtp.` (MTP draft) or `draft.` (separate draft model) names.

    A tied LM head (see `weight_mapper.tied_lm_head_alias`) is the embedding tensor.
    """
    from vllm_torchtpu.experimental.export import universal_adapter

    tensors = dict(model.named_buffers())
    tensors.update(model.named_parameters())
    tied = weight_mapper.tied_lm_head_alias(tensors)
    if tied:
        tensors[tied[0]] = tensors[tied[1]]
    tensors.update(universal_adapter.mtp_draft_tensors(model))
    tensors.update(universal_adapter.draft_model_tensors(model))
    return tensors


def host_tensors(model: Any) -> dict[str, Any]:
    """CPU copies of `local_tensors(model)`; tied names keep sharing one copy.

    Taken before lowering: tracing on TPU can leave placeholder tensors in the
    model (e.g. a MoE router weight), which cannot be copied to host afterwards.
    Tensors left on the meta device hold no data and no step reads them (Muse
    Spark's encoders, when every multimodal prompt limit is 0), so they are
    skipped.
    """
    copies: dict[int, Any] = {}
    return {
        name: copies.setdefault(id(t), t.detach().cpu())
        for name, t in local_tensors(model).items()
        if not t.is_meta
    }


def to_numpy(tensor: Any) -> Any:
    """A CPU numpy view of `tensor`, with `ml_dtypes` types for bf16 and FP8."""
    import ml_dtypes  # pyrefly: ignore[missing-import]
    import torch

    views = {
        torch.bfloat16: (torch.int16, ml_dtypes.bfloat16),
        torch.float8_e4m3fn: (torch.uint8, ml_dtypes.float8_e4m3fn),
        torch.float8_e5m2: (torch.uint8, ml_dtypes.float8_e5m2),
    }
    t = tensor.detach().cpu().contiguous()
    if t.dtype in views:
        int_dtype, np_dtype = views[t.dtype]
        return t.view(int_dtype).numpy().view(np_dtype)
    return t.numpy()


def iter_checkpoint_shards(
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    axis_sizes: Mapping[str, int],
) -> Iterator[Any]:
    """Yields this rank's shard of every `abstract_params` input as a numpy array.

    A tensor's shard is the rank's own tensor. A per-device constant, declared as
    a global `[tp, ...]` array, has a `[1, ...]` shard holding the rank's value.
    """
    from vllm_torchtpu.experimental.export import tpu_lowering

    for spec in inputs:
        name = spec["name"]
        if name not in tensors:
            raise KeyError(f"abstract_params input {name} is not in the loaded model")
        tensor = tensors[name]
        dtype = tpu_lowering.dtype_name(tensor.dtype)
        if dtype != spec["dtype"]:
            raise ValueError(
                f"{name}: loaded dtype {dtype}, artifact declares {spec['dtype']}"
            )
        expected = shard_shape(spec, axis_sizes)
        try:
            array = to_numpy(tensor)
        except RuntimeError as e:
            raise RuntimeError(f"{name}: cannot copy the loaded tensor to host") from e
        if list(array.shape) != expected:
            if [1, *array.shape] != expected:
                raise ValueError(
                    f"{name}: loaded shape {list(array.shape)}, "
                    f"artifact shard shape {expected}"
                )
            array = array.reshape(expected)
        yield array


def checkpoint_shards(
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    axis_sizes: Mapping[str, int],
) -> list[Any]:
    """This rank's shard of every `abstract_params` input (`iter_checkpoint_shards`)."""
    return list(iter_checkpoint_shards(inputs, tensors, axis_sizes))


def save_checkpoint(
    inputs: Sequence[Mapping[str, Any]],
    shards: Iterable[Any],
    axis_sizes: Mapping[str, int],
    data_dir: str,
    small_array_bytes: int = SMALL_ARRAY_BYTES,
) -> None:
    """Saves the global arrays assembled from every process's `shards` with Orbax.

    Each process contributes one CPU device, the mesh position of its rank; Orbax
    writes each array's chunks from the processes that hold them. A sharded array
    of at most `small_array_bytes` is gathered first and saved as one chunk. 64-bit
    types are enabled only while saving, so i64 constants keep their width without
    changing how the process's other JAX code (e.g. Pallas kernels) traces.
    """
    import jax

    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        _save_checkpoint(inputs, shards, axis_sizes, data_dir, small_array_bytes)
    finally:
        jax.config.update("jax_enable_x64", x64)


def _save_checkpoint(
    inputs: Sequence[Mapping[str, Any]],
    shards: Iterable[Any],
    axis_sizes: Mapping[str, int],
    data_dir: str,
    small_array_bytes: int,
) -> None:
    import jax
    import numpy as np
    import orbax.checkpoint as ocp  # pyrefly: ignore[missing-import]
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = sorted(jax.devices("cpu"), key=lambda d: (d.process_index, d.id))
    mesh = Mesh(
        np.asarray(devices).reshape([axis_sizes[a] for a in _AXIS_NAMES]), _AXIS_NAMES
    )
    (local,) = jax.local_devices(backend="cpu")
    tree = {}
    digests = []
    small = []
    for i, (spec, shard) in enumerate(zip(inputs, shards, strict=True)):
        sharding = NamedSharding(mesh, PartitionSpec(*spec["sharding"]))
        tree[f"array_{i}"] = jax.make_array_from_single_device_arrays(
            tuple(spec["shape"]), sharding, [jax.device_put(shard, local)]
        )
        shard_count = int(np.prod([axis_sizes[a] for a in _sharded_axes(spec)]))
        if shard_count > 1 and shard.nbytes * shard_count <= small_array_bytes:
            small.append(f"array_{i}")
        digests.append(
            zlib.crc32(shard.tobytes()) if _has_replicas(spec, axis_sizes) else 0
        )
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        check_replicas(
            inputs,
            np.asarray(multihost_utils.process_allgather(np.asarray(digests))),
            axis_sizes,
        )
    if small:
        # One chunk instead of one per shard: gather the array and save it
        # replicated (Orbax then writes it once, from one process).
        from jax.experimental import multihost_utils

        gathered = multihost_utils.process_allgather(
            {k: tree[k] for k in small}, tiled=True
        )
        replicated = NamedSharding(mesh, PartitionSpec())
        for k in small:
            tree[k] = jax.make_array_from_single_device_arrays(
                tree[k].shape, replicated, [jax.device_put(gathered[k], local)]
            )
    # Orbax also splits a replicated array among the processes holding it (one
    # chunk each); keep that for large arrays only.
    registry = ocp.type_handlers.create_type_handler_registry(
        (
            jax.Array,
            ocp.type_handlers.ArrayHandler(
                min_slice_bytes_for_replica_parallel=small_array_bytes
            ),
        )
    )
    checkpointer = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(
            use_ocdbt=CHECKPOINT_FORMAT == "zarr2_ocdbt",
            use_zarr3=False,
            type_handler_registry=registry,
        )
    )
    checkpointer.save(
        os.path.abspath(data_dir), args=ocp.args.PyTreeSave(tree), force=True
    )


def _sharded_axes(spec: Mapping[str, Any]) -> set[str]:
    axes: set[str] = set()
    for entry in spec["sharding"]:
        if entry is not None:
            axes.update([entry] if isinstance(entry, str) else entry)
    return axes


def _has_replicas(spec: Mapping[str, Any], axis_sizes: Mapping[str, int]) -> bool:
    """Whether more than one process holds a copy of the same shard of `spec`."""
    sharded = _sharded_axes(spec)
    return any(axis_sizes[a] > 1 for a in _AXIS_NAMES if a not in sharded)


def check_replicas(
    inputs: Sequence[Mapping[str, Any]],
    digests: Any,
    axis_sizes: Mapping[str, int],
) -> None:
    """Fails if processes holding copies of one shard saved different values.

    `digests[p][i]` is process `p`'s digest of its shard of input `i`; process `p`
    is the `p`-th device of the mesh. Orbax writes a replicated array's chunks
    from several processes, so a tensor that really differs per rank but is
    declared replicated would be saved as a mix of the ranks' values.
    """
    import numpy as np

    sizes = [axis_sizes[a] for a in _AXIS_NAMES]
    coords = [
        dict(zip(_AXIS_NAMES, (int(c) for c in coord), strict=True))
        for coord in zip(*np.unravel_index(range(len(digests)), sizes), strict=True)
    ]
    mixed = []
    for i, spec in enumerate(inputs):
        if not _has_replicas(spec, axis_sizes):
            continue
        sharded = sorted(_sharded_axes(spec))
        groups: dict[tuple[int, ...], set[int]] = {}
        for p, coord in enumerate(coords):
            groups.setdefault(tuple(coord[a] for a in sharded), set()).add(
                int(digests[p][i])
            )
        if any(len(g) > 1 for g in groups.values()):
            mixed.append(f"{spec['name']} {spec['sharding']}")
    if mixed:
        raise ValueError(
            f"{len(mixed)} tensors are declared replicated along an axis whose ranks "
            "hold different values; they need a sharded axis: " + ", ".join(mixed)
        )


def init_jax(rank: int, world_size: int, coordinator: str | None) -> None:
    """Joins this rank to the launcher's JAX distributed runtime at `coordinator`.

    Must run before any JAX use in the process. A single process needs none.
    """
    import jax

    if world_size > 1:
        if not coordinator:
            raise ValueError(
                f"{world_size} export workers need {JAX_COORDINATOR_FLAG} host:port"
            )
        jax.distributed.initialize(
            coordinator, num_processes=world_size, process_id=rank
        )


@trace("export.save_checkpoint")
def save_model_checkpoint(
    tensors: Mapping[str, Any], artifact_dir: str, data_dir: str
) -> None:
    """Saves this rank's `tensors`, by name, for the artifact in `artifact_dir`."""
    inputs = abstract_params_inputs(artifact_dir)
    sizes = mesh_axis_sizes(artifact_dir)
    shards = iter_checkpoint_shards(inputs, tensors, sizes)
    save_checkpoint(inputs, shards, sizes, data_dir)


def run_cpu_load_worker(args: argparse.Namespace) -> None:
    """`checkpoint` mode: loads this rank's weights on CPU and saves its shards."""
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment

    from vllm_torchtpu import patch_registry
    from vllm_torchtpu.experimental.export import universal_adapter

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    init_jax(rank, world_size, args.jax_coordinator)

    vllm_config = universal_adapter.create_export_vllm_config(args)
    # vLLM runs on its CPU platform here, which skips the TPU platform's patches;
    # apply them so the model has the parameters and buffers it has on TPU.
    patch_registry.apply("platform_activation", model_config=vllm_config.model_config)
    universal_adapter.check_supported_parallelism(vllm_config.parallel_config)
    with set_current_vllm_config(vllm_config):
        if not dist.is_initialized():
            # TPU collectives are unavailable on CPU; loading only needs gloo.
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                local_rank=rank,
                distributed_init_method="env://",
                backend="gloo",
            )
        model = universal_adapter.load_vllm_model(vllm_config, load_format="auto")

    save_model_checkpoint(
        local_tensors(model),
        args.output_dir,
        os.path.join(args.output_dir, CHECKPOINT_DATA_DIR),
    )


def launch_cpu_workers(
    argv: Sequence[str],
    tp_size: int,
    worker_flag: str,
) -> None:
    """Runs `python -m vllm_torchtpu.experimental.export` once per rank on CPU."""
    env = dict(
        os.environ,
        **CPU_ENV,
        WORLD_SIZE=str(tp_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(portpicker.pick_unused_port()),
    )
    coordinator = f"127.0.0.1:{portpicker.pick_unused_port()}"
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "vllm_torchtpu.experimental.export"]
            + [*argv, JAX_COORDINATOR_FLAG, coordinator, worker_flag],
            env=dict(env, RANK=str(rank), LOCAL_RANK=str(rank)),
        )
        for rank in range(tp_size)
    ]
    try:
        failed = [r for r, proc in enumerate(procs) if proc.wait() != 0]
        if failed:
            raise RuntimeError(f"checkpoint workers for ranks {failed} failed")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()


def declare_checkpoint_format(artifact_dir: str) -> None:
    """Records `CHECKPOINT_FORMAT` in the artifact's `serving_config.checkpoint_config`.

    `checkpoint` mode fills an artifact exported earlier, possibly before it declared
    a format, so the artifact is updated to declare what was written.
    """
    path = os.path.join(artifact_dir, "metadata.json")
    with open(path) as f:
        meta = json.load(f)
    meta["serving_config"]["checkpoint_config"] = {"format": CHECKPOINT_FORMAT}
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def export_checkpoint(argv: Sequence[str], tp_size: int, output_dir: str) -> None:
    """`checkpoint` mode: fills the existing artifact in `output_dir` on CPU."""
    if not os.path.isfile(os.path.join(output_dir, "metadata.json")):
        raise FileNotFoundError(
            f"--export-mode checkpoint needs an exported artifact; {output_dir} has "
            "no metadata.json"
        )
    shutil.rmtree(os.path.join(output_dir, CHECKPOINT_DATA_DIR), ignore_errors=True)
    launch_cpu_workers(argv, tp_size, "--export-worker")
    declare_checkpoint_format(output_dir)
