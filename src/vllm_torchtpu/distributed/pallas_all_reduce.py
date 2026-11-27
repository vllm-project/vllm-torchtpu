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
XLA lowers as it lowers the usual all-reduce, except tensors up to
TPU_TP_PAIR_ALL_REDUCE_MAX_BYTES, which take the chip-pairs-first kernel
(kernels/collectives/pair_all_reduce.py). The kernels' output bits can differ
from XLA's all-reduce: they sum in f32 (the pair kernel after rounding each
chip's partial to bf16) and round once; every rank gets the same bits.

With TPU_TP_PAIR_REDUCE_SCATTER_MIN_BYTES, the TP group's dim-0 reduce-scatters
of 2-D bf16 tensors take a second op the same way: tensors of at least that
many bytes whose shape the chip-pairs-first reduce-scatter takes
(kernels/collectives/pair_reduce_scatter.py) run it, the rest a psum_scatter
over the mesh.
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
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.collectives import pair_all_reduce, pair_reduce_scatter
from vllm_torchtpu.kernels.collectives import tp_all_reduce as kernel
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# A wrong rank or mesh hangs the self-test's barrier: dump every stack and exit.
SELFTEST_TIMEOUT_S = 300
_SELFTEST_ROWS = (8, 24)
# Row counts the pair kernel's self-test may take (4, 2 and 1 row blocks).
_PAIR_SELFTEST_ROWS = (64, 96, 80, 112, 128, 192, 256)


@dataclasses.dataclass(frozen=True)
class _Armed:
    ranks: tuple[int, ...]
    op: Any
    gather_op: Any
    rank: torch.Tensor  # this rank's mesh index, int32 [1, 1]
    # Every rank's mesh index equals its group rank (decided once, from all
    # ranks' indices, so the answer is the same on every rank).
    rank_ordered: bool = False
    mesh: Any = None
    max_bytes: int = 0
    # The chip-pairs-first kernel: its byte cap (0: off) and topology table.
    pair_max_bytes: int = 0
    pair_table: Any = None
    # The chip-pairs-first reduce-scatter op (None: off).
    rs_op: Any = None


_ARMED: _Armed | None = None


def reduce_body(n: int, max_bytes: int, axis: str, pair_max_bytes=0, pair_table=None):
    """The op's per-device all-reduce of (x, rank), for use under shard_map
    over the op's mesh (axis `axis`, n devices): the direct kernel up to
    max_bytes of view, the pair kernel up to pair_max_bytes (with its
    topology table), else psum."""

    def body(x, rank):
        rows, hidden = x.shape
        if 0 < kernel.view_bytes(rows, hidden, n) <= max_bytes:
            return kernel.all_reduce(x, rank, n)
        if x.size * x.dtype.itemsize <= pair_max_bytes and pair_all_reduce.supported(
            rows, hidden, n
        ):
            return pair_all_reduce.all_reduce(x, rank, pair_table)
        return lax.psum(x, axis)

    return body


def _make_op(mesh, max_bytes: int, pair_max_bytes=0, pair_table=None):
    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        reduce_body(int(mesh.size), max_bytes, axis, pair_max_bytes, pair_table),
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


def reduce_scatter_body(n: int, min_bytes: int, axis: str, table, wire: str):
    """The reduce-scatter op's per-device body over (x, rank), under shard_map
    over the op's mesh (axis `axis`, n devices): the chip-pairs-first kernel
    (its `wire`) from min_bytes of x where it takes the shape, else
    psum_scatter."""

    def body(x, rank):
        rows, hidden = x.shape
        chunk = pair_reduce_scatter.chunk_rows(rows, hidden, n)
        if chunk is not None and x.size * x.dtype.itemsize >= min_bytes:
            return pair_reduce_scatter.reduce_scatter(x, rank, table, chunk, wire)
        return lax.psum_scatter(x, axis, scatter_dimension=0, tiled=True)

    return body


def _make_rs_op(mesh, min_bytes: int, table, wire: str):
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        reduce_scatter_body(n, min_bytes, axis, table, wire),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(axis),
        check_vma=False,
    )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        args = (x, rank)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::tp_reduce_scatter",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, rank: x.new_empty((x.shape[0] // n, *x.shape[1:])))
    return op


def _rs_selftest(
    op,
    rank: torch.Tensor,
    mesh_index: int,
    n: int,
    rows: int,
    hidden: int,
    device,
    tol: float,
) -> bool:
    """Whether the reduce-scatter op gives this rank the sum of its rows block
    within tol (the sums are integers: a misplaced chunk is off by more)."""
    x = _selftest_inputs(n, rows, hidden)
    block = rows // n
    want = x.sum(0)[mesh_index * block : (mesh_index + 1) * block]
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        mine = torch.from_numpy(x[mesh_index]).to(torch.bfloat16).to(device)
        got = op(mine, rank).to(torch.float32).cpu().numpy()
    finally:
        faulthandler.cancel_dump_traceback_later()
    return bool(np.max(np.abs(got - want)) <= tol)


def _make_gather_op(mesh):
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        lambda x, rank: kernel.all_gather_small(x, rank, n),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(axis),
        check_vma=False,
    )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        args = (x, rank)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::tp_all_gather_small",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, rank: x.new_empty((n * x.shape[0], *x.shape[1:])))
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
    op, rank: torch.Tensor, mesh_index: int, n: int, hidden: int, device, rows_list
) -> list:
    failures = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for rows in rows_list:
            x = _selftest_inputs(n, rows, hidden)
            mine = torch.from_numpy(x[mesh_index]).to(torch.bfloat16).to(device)
            got = op(mine, rank).to(torch.float32).cpu().numpy()
            if not np.array_equal(got, x.sum(0)):
                failures.append(rows)
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def _gather_selftest(op, rank: torch.Tensor, mesh_index: int, n: int, device) -> bool:
    """Whether the gather op returns every rank's distinct rows in rank order."""
    rows = 512
    want = np.arange(n * rows, dtype=np.float32).reshape(n * rows, 1)
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        mine = torch.from_numpy(want[mesh_index * rows : (mesh_index + 1) * rows])
        got = op(mine.to(device), rank).cpu().numpy()
    finally:
        faulthandler.cancel_dump_traceback_later()
    return bool(np.array_equal(got, want))


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
    seen = _gather(dict(refusal=refusal, mesh_index=mesh_index))
    if any(s["refusal"] for s in seen):
        logger.warning("tp_all_reduce: not armed: %s", [s["refusal"] for s in seen])
        return False
    if sorted(s["mesh_index"] for s in seen) != list(range(tp.world_size)):
        raise RuntimeError(f"tp_all_reduce: mesh indices are not 0..n-1: {seen}")
    n = int(mesh.size)
    pair_max_bytes = envs.TPU_TP_PAIR_ALL_REDUCE_MAX_BYTES
    pair_table = None
    rows_list = _SELFTEST_ROWS
    if pair_max_bytes > 0:
        # Raises on a mesh that is not a tpu7x 2x2x1 (the flag asks for it).
        pair_table = pair_all_reduce.topology_table(d.id for d in mesh.devices.flat)
        pair_max_bytes = min(pair_max_bytes, pair_all_reduce.MAX_BYTES)
        # The rows that reach the pair kernel under these caps, up to three.
        pair_rows = [
            r
            for r in _PAIR_SELFTEST_ROWS
            if kernel.view_bytes(r, hidden, n) > max_bytes
            and r * hidden * 2 <= pair_max_bytes
            and pair_all_reduce.supported(r, hidden, n)
        ][:3]
        if not pair_rows:
            raise RuntimeError(
                "tp_all_reduce: TPU_TP_PAIR_ALL_REDUCE_MAX_BYTES admits no "
                f"self-test rows above the {max_bytes} B direct cap"
            )
        rows_list = _SELFTEST_ROWS + tuple(pair_rows)
    op = _make_op(mesh, max_bytes, pair_max_bytes, pair_table)
    rank = torch.tensor([[mesh_index]], dtype=torch.int32, device=device)
    failures = _gather(_selftest(op, rank, mesh_index, n, hidden, device, rows_list))
    if any(failures):
        raise RuntimeError(
            f"tp_all_reduce: self-test failed (rows per rank): {failures}"
        )
    gather_op = None
    if envs.TPU_TP_PALLAS_SMALL_ALL_GATHER:
        gather_op = _make_gather_op(mesh)
        if not all(_gather(_gather_selftest(gather_op, rank, mesh_index, n, device))):
            raise RuntimeError("tp_all_gather_small: self-test failed")
    rank_ordered = all(s["mesh_index"] == i for i, s in enumerate(seen))
    rs_op = None
    rs_min_bytes = envs.TPU_TP_PAIR_REDUCE_SCATTER_MIN_BYTES
    if rs_min_bytes > 0:
        # The kernel gives mesh index i rows block i; the group's reduce-scatter
        # gives it to group rank i.
        if not rank_ordered:
            raise RuntimeError(
                "TPU_TP_PAIR_REDUCE_SCATTER_MIN_BYTES needs a rank-ordered mesh: "
                f"{seen}"
            )
        rs_table = pair_reduce_scatter.topology_table(d.id for d in mesh.devices.flat)
        wire = envs.TPU_TP_PAIR_REDUCE_SCATTER_WIRE
        rs_op = _make_rs_op(mesh, rs_min_bytes, rs_table, wire)
        # The smallest kernel shape at the byte floor: whole 64-row chunks.
        unit = n * pair_reduce_scatter.CHUNK_ROWS
        rows = unit * max(1, -(-rs_min_bytes // (unit * hidden * 2)))
        # int8 chip sums (|s| <= 10) are off by up to 10 / 254 each, and the
        # bf16 output by an eighth near 40.
        tol = 0.0 if wire == "bf16" else 0.5
        ok = _rs_selftest(rs_op, rank, mesh_index, n, rows, hidden, device, tol)
        if not all(_gather(ok)):
            raise RuntimeError(f"tp_reduce_scatter: self-test failed at {rows} rows")
    _ARMED = _Armed(
        ranks=tuple(int(r) for r in tp.ranks),
        op=op,
        gather_op=gather_op,
        rank=rank,
        mesh=mesh,
        max_bytes=max_bytes,
        rank_ordered=rank_ordered,
        pair_max_bytes=pair_max_bytes,
        pair_table=pair_table,
        rs_op=rs_op,
    )
    logger.info(
        "tp_all_reduce: armed over %d ranks (mesh index %d), Pallas up to %d B "
        "views, chip-pairs-first up to %d B, reduce-scatter from %d B",
        n,
        mesh_index,
        max_bytes,
        pair_max_bytes,
        rs_min_bytes if rs_op is not None else 0,
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


def reduce_scatter(ranks, x: torch.Tensor) -> torch.Tensor | None:
    """The armed reduce-scatter op's dim-0 reduce-scatter of x over the group
    of `ranks`, or None when it is not armed for this group or tensor (the
    caller takes its usual path)."""
    armed = _ARMED
    if armed is None or armed.rs_op is None or tuple(ranks) != armed.ranks:
        return None
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        return None
    # The op picks the kernel per static size inside (see `all_reduce`).
    return armed.rs_op(x, armed.rank)


def armed_rank(ranks) -> torch.Tensor | None:
    """This rank's index in the group of `ranks` as an int32 [1, 1] tensor:
    the armed op's mesh index, when the op is armed for that group and every
    rank's mesh index is its group rank (the same answer on every rank), so
    callers can index rank-ordered blocks by it."""
    armed = _ARMED
    if armed is None or tuple(ranks) != armed.ranks or not armed.rank_ordered:
        return None
    return armed.rank


def armed_for(ranks) -> _Armed | None:
    """The armed op's record (mesh, mesh index, byte cap) when it serves the
    group of `ranks`, else None: for kernels that fold this all-reduce in."""
    armed = _ARMED
    if armed is None or tuple(ranks) != armed.ranks:
        return None
    return armed


# Bytes one rank sends through `all_gather_small`: the gathered tensor sits in
# VMEM (n x this).
_SMALL_GATHER_MAX_BYTES = 256 << 10


def all_gather_small(ranks, x: torch.Tensor) -> torch.Tensor | None:
    """The armed Pallas all-gather (rows, rank order) of a small 32-bit x
    over the group of `ranks`, or None when it is not armed for this group or
    tensor (the caller takes its usual all-gather)."""
    armed = _ARMED
    if armed is None or armed.gather_op is None or tuple(ranks) != armed.ranks:
        return None
    # Blocks land at their mesh index; callers pair them with group-rank-ordered
    # gathers.
    if not armed.rank_ordered:
        return None
    # The kernel holds the gathered tensor (n x x) in VMEM.
    if (
        kernel.gather_view(x.numel(), x.element_size()) is None
        or x.numel() * x.element_size() > _SMALL_GATHER_MAX_BYTES
    ):
        return None
    return armed.gather_op(x.contiguous(), armed.rank)


def tp_all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """`tensor_model_parallel_all_gather(x, dim)`, by the armed Pallas
    all-gather (`all_gather_small`, `TPU_TP_PALLAS_SMALL_ALL_GATHER`) when x
    is a small 32- or 16-bit tensor (16-bit pairs of the last dim as words),
    else vLLM's. The same concatenation in rank order. The size test picks
    the path per traced shape: call it from programs compiled per bucket
    (dynamic=False), not from the backbone's shared trace (shape_variants)."""
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather

    gathered = None
    words = x
    if x.element_size() == 2 and x.shape[-1] % 2 == 0:
        words = x.contiguous().view(torch.int32)
    if _ARMED is not None and words.element_size() == 4:
        gathered = all_gather_small(get_tp_group().ranks, words)
    if gathered is None:
        return tensor_model_parallel_all_gather(x, dim)
    blocks = gathered.view(x.dtype).view(-1, *x.shape)  # [n, *x.shape]
    dim = dim % x.dim()
    return torch.cat(blocks.unbind(0), dim=dim) if dim else blocks.flatten(0, 1)
