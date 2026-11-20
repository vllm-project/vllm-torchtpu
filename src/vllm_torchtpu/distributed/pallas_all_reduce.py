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
"""TP all-reduce of small bf16 activations with a Pallas kernel
(TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES; kernels/collectives/tp_all_reduce.py).

`arm` runs once at the end of TPUModelRunner.load_model. It is collective over
the TP group: it builds the op's mesh (the EP mesh, which must span the same
ranks as the TP group and the whole world, so every Pallas kernel in a module
shares one device assignment), checks that every rank armed with a distinct
mesh index, and runs the kernel once on inputs whose sums are exact. After
that, TpuDeviceCommunicator.all_reduce of the TP group sends every 2-D bf16
tensor through one torch op. Inside it the token count is static (torch_tpu
exports one program per count): the kernel serves views up to the byte cap,
and a larger tensor (a prefill chunk) takes a psum over the same mesh, which
XLA lowers as it lowers the usual all-reduce. Views above
kernel.TWO_LEVEL_MIN_VIEW_BYTES take the kernel's two-level variant when every
chip of the mesh holds two of its devices. The kernel's output bits can
differ from XLA's all-reduce: it sums in f32 in rank order and rounds once
(the two-level variant rounds the on-chip pair sums too); every rank gets the
same bits.
"""

import dataclasses
import faulthandler
from typing import Any

import jax
import numpy as np
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.chip_topology import get_chip_topology
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.collectives import tp_all_reduce as kernel
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# A wrong rank or mesh hangs the self-test's barrier: dump every stack and exit.
SELFTEST_TIMEOUT_S = 300
# At hidden 6144 over 8 ranks: 8 and 24 rows take the flat kernel, 32 and 48
# the two-level one (kernel.TWO_LEVEL_MIN_VIEW_BYTES); rows whose view the cap
# refuses are skipped.
_SELFTEST_ROWS = (8, 24, 32, 48)


@dataclasses.dataclass(frozen=True)
class _Armed:
    ranks: tuple[int, ...]
    op: Any
    rank: torch.Tensor  # this rank's mesh index, int32 [1, 1]


_ARMED: _Armed | None = None


def _body(n: int, max_bytes: int, axis: str, table: tuple[int, ...] | None):
    def body(x, rank):
        rows, hidden = x.shape
        view = kernel.view_bytes(rows, hidden, n)
        if not 0 < view <= max_bytes:
            return lax.psum(x, axis)
        if table is not None and view > kernel.TWO_LEVEL_MIN_VIEW_BYTES:
            return kernel.all_reduce_two_level(x, rank, table, n)
        return kernel.all_reduce(x, rank, n)

    return body


def _two_level_table(mesh) -> tuple[int, ...] | None:
    """kernel.two_level_table of the mesh's chips (chip_topology's grouping of
    device ids); None, the flat kernel only, when the grouping is unknown."""
    topology = get_chip_topology()
    if topology is None:
        return None
    index = {int(d.id): i for i, d in enumerate(mesh.devices.flat)}
    if sorted(index) != sorted(d for chip in topology.chips for d in chip):
        logger.warning(
            "tp_all_reduce: mesh devices %s are not the chips' %s",
            sorted(index),
            topology.chips,
        )
        return None
    return kernel.two_level_table([[index[d] for d in chip] for chip in topology.chips])


def _make_op(mesh, max_bytes: int, table: tuple[int, ...] | None):
    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        _body(int(mesh.size), max_bytes, axis, table),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(axis),
        check_vma=False,
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        args = (x, rank)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # PartitionSpecs, as other jax_op call sites pass them (torch_tpu's
    # annotation is narrower).
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::tp_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, rank: torch.empty_like(x))
    return op


def _refusal() -> str | None:
    tp = parallel_state.get_tp_group()
    ep = get_ep_group()
    if ep is None:
        return "no EP group"
    tp_ranks = tuple(int(r) for r in tp.ranks)
    if tp_ranks != tuple(int(r) for r in ep.ranks):
        return f"TP ranks {tp_ranks} != EP ranks {tuple(ep.ranks)}"
    world = torch.distributed.get_world_size()
    if len(tp_ranks) != world:
        return f"the TP group has {len(tp_ranks)} of {world} ranks"
    return None


def _selftest_inputs(n: int, rows: int, hidden: int) -> np.ndarray:
    """[n, rows, hidden] partials whose sums are small integers, exact in bf16,
    and differ between lane blocks and rows, so a chunk in the wrong place shows."""
    d = np.arange(n)[:, None, None]
    r = np.arange(rows)[None, :, None]
    c = np.arange(hidden)[None, None, :] // 128
    return (((d + 1) * 3 + r * 5 + c * 7) % 11 - 5).astype(np.float32)


def _selftest(
    op, rank: torch.Tensor, mesh_index: int, n: int, hidden: int, max_bytes: int, device
) -> list:
    failures = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for rows in _SELFTEST_ROWS:
            if kernel.view_bytes(rows, hidden, n) > max_bytes:
                continue
            x = _selftest_inputs(n, rows, hidden)
            mine = torch.from_numpy(x[mesh_index]).to(torch.bfloat16).to(device)
            got = op(mine, rank).to(torch.float32).cpu().numpy()
            if not np.array_equal(got, x.sum(0)):
                failures.append(rows)
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def _gather(obj) -> list:
    tp = parallel_state.get_tp_group()
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, obj, group=tp.cpu_group)
    return seen


def arm(hidden: int, device: torch.device) -> bool:
    """Route the TP group's small bf16 all-reduces through the kernel; returns
    whether it armed. Collective over the TP group."""
    global _ARMED
    max_bytes = min(envs.TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES, kernel.MAX_VIEW_BYTES)
    tp = parallel_state.get_tp_group()
    if max_bytes <= 0 or tp.world_size == 1:
        return False
    refusal = _refusal()
    mesh = None if refusal else build_ep_mesh()  # collective
    mesh_index = -1 if mesh is None else ep_mesh_index()
    table = None if mesh is None else _two_level_table(mesh)
    seen = _gather(dict(refusal=refusal, mesh_index=mesh_index, table=table))
    if any(s["refusal"] for s in seen):
        logger.warning("tp_all_reduce: not armed: %s", [s["refusal"] for s in seen])
        return False
    if sorted(s["mesh_index"] for s in seen) != list(range(tp.world_size)):
        raise RuntimeError(f"tp_all_reduce: mesh indices are not 0..n-1: {seen}")
    if any(s["table"] != table for s in seen):
        # Every rank must run the same kernel: the two differ in their DMAs.
        logger.warning("tp_all_reduce: ranks disagree on the chip table; flat only")
        table = None
    n = int(mesh.size)
    op = _make_op(mesh, max_bytes, table)
    rank = torch.tensor([[mesh_index]], dtype=torch.int32, device=device)
    failures = _gather(_selftest(op, rank, mesh_index, n, hidden, max_bytes, device))
    if any(failures):
        raise RuntimeError(
            f"tp_all_reduce: self-test failed (rows per rank): {failures}"
        )
    _ARMED = _Armed(ranks=tuple(int(r) for r in tp.ranks), op=op, rank=rank)
    logger.info(
        "tp_all_reduce: armed over %d ranks (mesh index %d), Pallas up to %d B "
        "views, two-level above %s B",
        n,
        mesh_index,
        max_bytes,
        "no" if table is None else kernel.TWO_LEVEL_MIN_VIEW_BYTES,
    )
    return True


def all_reduce(ranks, x: torch.Tensor) -> torch.Tensor | None:
    """The armed op's all-reduce of x over the group of `ranks`, or None when
    this group or tensor is not armed (the caller takes its usual path)."""
    armed = _ARMED
    if armed is None or tuple(ranks) != armed.ranks:
        return None
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        return None
    # Every size goes to the op, which decides on the static size inside: a
    # branch here would give the model's Dynamo traces different graphs (and
    # inputs) per bucket, which shape_variants refuses to hand buckets between.
    return armed.op(x, armed.rank)
