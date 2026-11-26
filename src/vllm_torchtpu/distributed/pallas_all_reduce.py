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
XLA lowers as it lowers the usual all-reduce. The kernel's output bits can
differ from XLA's all-reduce: it sums in f32 in rank order and rounds once;
every rank gets the same bits.

With TPU_MINIMAX_M3_SP_QRS (and a rank-ordered mesh), `arm` also builds a row
reduce-scatter op for the sequence-parallel steps (`reduce_scatter`): steps of
>= TPU_MINIMAX_M3_SP_QRS_MIN_TOKENS rows run qar_reduce_scatter (int8 ICI
payload), smaller ones XLA's reduce-scatter (psum_scatter) over the same mesh.
"""

import dataclasses
import faulthandler
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.collectives import qar_all_reduce as qar
from vllm_torchtpu.kernels.collectives import qar_reduce_scatter as qrs
from vllm_torchtpu.kernels.collectives import tp_all_reduce as kernel
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# A wrong rank or mesh hangs the self-test's barrier: dump every stack and exit.
SELFTEST_TIMEOUT_S = 300
_SELFTEST_ROWS = (8, 24, 32)


@dataclasses.dataclass(frozen=True)
class _Armed:
    ranks: tuple[int, ...]
    op: Any
    rank: torch.Tensor  # this rank's mesh index, int32 [1, 1]
    # Every rank's mesh index equals its group rank (decided once, from all
    # ranks' indices, so the answer is the same on every rank).
    rank_ordered: bool = False
    # Row reduce-scatter op (TPU_MINIMAX_M3_SP_QRS), or None.
    rs_op: Any = None
    # Small 32-bit all-gather op (TPU_TP_PALLAS_SMALL_ALL_GATHER), or None.
    gather_op: Any = None
    # The op's mesh and byte cap, for kernels that run over the same mesh
    # (`armed_for`).
    mesh: Any = None
    max_bytes: int = 0


_ARMED: _Armed | None = None
_QAR_TRACED: set[int] = set()
_QRS_TRACED: set[int] = set()


def _qdq(x: jax.Array) -> jax.Array:
    """bf16 [rows, hidden] -> int8 with one absmax scale per (32 rows, lane) ->
    bf16: the payload rounding of kernels/collectives/qar_all_reduce.py."""
    rows = x.shape[0]
    t = qar.padded_rows(rows)
    xf = jnp.pad(x.astype(jnp.float32), ((0, t - rows), (0, 0)))
    return qar.dequantize(*qar.quantize(xf))[:rows].astype(x.dtype)


def _body(n: int, max_bytes: int, axis: str, direct_rows: int = 0,
          qdq_min_rows: int = 0, qar_min_rows: int = 0,
          qar_tab: np.ndarray | None = None):
    def body(x, rank):
        rows, hidden = x.shape
        # Static per bucket: decided at trace time inside the op.
        if qar_tab is not None and 0 < qar_min_rows <= rows:
            if rows not in _QAR_TRACED:
                # JAX trace time (once per bucket), not the torch forward.
                _QAR_TRACED.add(rows)
                logger.warning("tp_all_reduce: qar_all_reduce lowered for [%d, %d]", rows, hidden)
            return qar.all_reduce_v2(x, rank, qar_tab)
        qdq = 0 < qdq_min_rows <= rows
        if qdq:
            x = _qdq(x)
        if 0 < kernel.view_bytes(rows, hidden, n) <= max_bytes:
            out = kernel.all_reduce(x, rank, n, direct_rows)
        else:
            out = lax.psum(x, axis)
        return _qdq(out) if qdq else out

    return body


def _qdq_min_rows() -> int:
    """TPU_TP_AR_QDQ_ROWS > 0 emulates the int8 all-reduce payload (accuracy
    only, no speedup) on all-reduces of >= max(1, TPU_TP_AR_QDQ_MIN_TOKENS)
    rows; 0 = off."""
    if envs.TPU_TP_AR_QDQ_ROWS <= 0:
        return 0
    return max(1, envs.TPU_TP_AR_QDQ_MIN_TOKENS)


@dataclasses.dataclass(frozen=True)
class _ChipCore:
    id: int
    coords: tuple[int, int, int]
    core_on_chip: int


def _qar_devices(mesh) -> list[_ChipCore]:
    """The mesh's devices with chip coordinates and core index, in mesh order.

    The serving mesh holds torch_tpu's compile-only abstract devices, which
    carry only an id; the coordinates come from the PJRT attributes every rank
    reported at init. Without them, the tpu7x id layout is assumed (id = 2 *
    chip + core, chips in x-major order over the 2x2 grid)."""
    ids = [int(d.id) for d in mesh.devices.flat]
    if all(hasattr(d, "coords") and hasattr(d, "core_on_chip") for d in mesh.devices.flat):
        return [
            _ChipCore(int(d.id), tuple(int(c) for c in d.coords[:3]), int(d.core_on_chip))
            for d in mesh.devices.flat
        ]
    by_id: dict[int, _ChipCore] = {}
    try:
        from torch_tpu._internal.distributed import tpu_distributed

        for a in tpu_distributed.global_device_attributes():
            c = tuple(int(x) for x in (a.coords or ())) + (0, 0, 0)
            by_id[int(a.id)] = _ChipCore(int(a.id), c[:3], int(a.core_on_chip or 0))
    except Exception as e:  # noqa: BLE001 - fall back to the id layout below
        logger.warning("qar_all_reduce: no PJRT device attributes (%s)", e)
    if all(i in by_id for i in ids):
        source = "pjrt"
        out = [by_id[i] for i in ids]
    else:
        source = "id-layout"
        out = [_ChipCore(i, ((i // 2) % 2, (i // 2) // 2, 0), i % 2) for i in ids]
    logger.warning(
        "qar_all_reduce: device layout from %s: %s",
        source,
        [(d.id, d.coords[:2], d.core_on_chip) for d in out],
    )
    return out


def _make_op(mesh, max_bytes: int, direct_rows: int = 0,
             qdq_min_rows: int = 0, qar_min_rows: int = 0):
    axis = mesh.axis_names[0]
    qar_tab = qar.layout_table(_qar_devices(mesh)) if qar_min_rows else None
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        _body(int(mesh.size), max_bytes, axis, direct_rows, qdq_min_rows,
              qar_min_rows, qar_tab),
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
        "pallas::tp_all_reduce"
        + ("_qdq" if qdq_min_rows else "")
        + ("_qar" if qar_min_rows else ""),
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, rank: torch.empty_like(x))
    return op


def qrs_applies(rows: int, hidden: int, min_rows: int) -> bool:
    """Whether a [rows, hidden] row reduce-scatter runs qar_reduce_scatter."""
    return 0 < min_rows <= rows and rows % 8 == 0 and hidden % 256 == 0


def _rs_body(axis: str, min_rows: int, tab: np.ndarray):
    def body(x, rank):
        rows, hidden = x.shape
        # Static per bucket: decided at trace time inside the op.
        if qrs_applies(rows, hidden, min_rows):
            if rows not in _QRS_TRACED:
                _QRS_TRACED.add(rows)
                logger.warning("tp_reduce_scatter: qar_reduce_scatter lowered for [%d, %d]",
                               rows, hidden)
            return qrs.reduce_scatter(x, rank, tab)
        return lax.psum_scatter(x, axis, scatter_dimension=0, tiled=True)

    return body


def hier_applies(rows: int, hidden: int, min_rows: int) -> bool:
    """Whether a [rows, hidden] bf16 row reduce-scatter over 8 ranks runs
    hierrs_tc (row-streaming kernel from 24 MiB, micro-batched below)."""
    from vllm_torchtpu.kernels.collectives.hierrs_tc import dispatch, wrapper

    if not (0 < min_rows <= rows and wrapper.is_supported_by_hierrs_tc(8, rows, hidden)):
        return False
    return not (rows * hidden * 2 >= dispatch.STREAM_MIN_BYTES) or dispatch.supports_stream(
        rows, hidden, 2
    )


def _chip_perm(devices) -> tuple[int, ...] | None:
    """hierrs_tc's chip_perm for a mesh in ``devices`` order (``_ChipCore``):
    ``perm[k]`` is the mesh chip (mesh indices 2m, 2m + 1) on physical chip k,
    chips sorted by ICI coordinates; None when that is the identity. Raises if
    mesh indices 2m, 2m + 1 are not one chip's two cores."""
    chips = sorted({d.coords for d in devices})
    mesh_chips = []
    for m in range(len(devices) // 2):
        a, b = devices[2 * m], devices[2 * m + 1]
        if a.coords != b.coords or a.core_on_chip == b.core_on_chip:
            raise ValueError(f"mesh indices {2 * m}, {2 * m + 1} are not one chip: {devices}")
        mesh_chips.append(chips.index(a.coords))
    perm = tuple(mesh_chips.index(k) for k in range(len(mesh_chips)))
    return None if perm == tuple(range(len(perm))) else perm


def _hier_rs_body(axis: str, min_rows: int, chip_perm):
    from vllm_torchtpu.kernels.collectives.hierrs_tc import dispatch

    local = dispatch.make_reduce_scatter_fn(
        None, tuple(range(8)), axis_name=axis, chip_perm=chip_perm
    ).local

    def body(x, rank):
        del rank
        rows, hidden = x.shape
        if hier_applies(rows, hidden, min_rows):
            if rows not in _QRS_TRACED:
                _QRS_TRACED.add(rows)
                logger.warning("tp_reduce_scatter: hierrs_tc lowered for [%d, %d]", rows, hidden)
            return local(x)
        return lax.psum_scatter(x, axis, scatter_dimension=0, tiled=True)

    return body


def _make_rs_op(mesh, min_rows: int, hier: bool = False):
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    devices = _qar_devices(mesh)
    if hier:
        body = _hier_rs_body(axis, min_rows, _chip_perm(devices))
    else:
        body = _rs_body(axis, min_rows, qrs.layout_table(devices))
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        body,
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
        "pallas::tp_hier_reduce_scatter" if hier else "pallas::tp_qar_reduce_scatter",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, rank: x.new_empty((x.shape[0] // n, x.shape[1])))
    return op


def _qrs_selftest(op, rank, n: int, hidden: int, device, rows: int) -> dict:
    """One row reduce-scatter of known partials: rel L2 of this rank's chunk
    against the exact sum of its rows (a chunk in the wrong place shows)."""
    gen = torch.Generator().manual_seed(4321)
    parts = torch.randn((n, rows, hidden), generator=gen).to(torch.bfloat16)
    me = int(rank.reshape(-1)[0].item())
    out = op(parts[me].to(device), rank).float().cpu()
    c = rows // n
    exact = parts.double().sum(0)[me * c : (me + 1) * c]
    rel = float((out.double() - exact).norm() / exact.norm())
    return dict(rank=me, rel=rel)


def _hier_selftest(op, rank, n: int, hidden: int, device, rows: int) -> dict:
    """Integer partials (exact in bf16 in any order: chunk placement must be
    bitwise) and random partials (rel L2 of this rank's chunk)."""
    me = int(rank.reshape(-1)[0].item())
    c = rows // n
    ints = _selftest_inputs(n, rows, hidden)
    got = op(torch.from_numpy(ints[me]).to(torch.bfloat16).to(device), rank)
    exact_ok = bool(np.array_equal(got.float().cpu().numpy(), ints.sum(0)[me * c : (me + 1) * c]))
    out = _qrs_selftest(op, rank, n, hidden, device, rows)
    return dict(rank=me, rel=out["rel"], exact=exact_ok)


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
    op, rank: torch.Tensor, mesh_index: int, n: int, hidden: int, device
) -> list:
    failures = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for rows in _SELFTEST_ROWS:
            x = _selftest_inputs(n, rows, hidden)
            mine = torch.from_numpy(x[mesh_index]).to(torch.bfloat16).to(device)
            got = op(mine, rank).to(torch.float32).cpu().numpy()
            if not np.array_equal(got, x.sum(0)):
                failures.append(rows)
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def _qar_selftest(op, rank, n: int, hidden: int, device, rows: int) -> dict:
    """One qar all-reduce of known partials: rel L2 against the exact sum and a
    digest of the bits (the caller requires every rank's digest to agree)."""
    # Host-side math in numpy: ATen's CPU thread pool is invalid in some
    # worker processes (forward_run twins: "Invalid thread pool").
    rng = np.random.default_rng(1234)
    u = rng.standard_normal((n, rows, hidden), dtype=np.float32).view(np.uint32)
    bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)  # bf16, RNE
    parts = (bits.astype(np.uint32) << 16).view(np.float32)
    me = int(rank.reshape(-1)[0].item())
    mine = torch.from_numpy(bits[me].view(np.int16)).view(torch.bfloat16)
    out = op(mine.to(device), rank).cpu().view(torch.int16).numpy().view(np.uint16)
    out = (out.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    exact = parts.astype(np.float64).sum(0)
    rel = float(np.linalg.norm(out - exact) / np.linalg.norm(exact))
    digest = float((out.reshape(-1) * np.arange(1, out.size + 1, dtype=np.float64)).sum())
    return dict(rank=me, rel=rel, digest=digest)


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


# libtpu flags that decide which all-reduces the kernel competes with: below
# the SparseCore offload minimum, XLA's all-reduce runs on the TensorCore.
_COLLECTIVE_FLAGS = (
    "xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes",
    "xla_tpu_enable_sparse_core_collective_offload_reduce_scatter",
    "xla_tpu_enable_sparse_core_hierarchical_all_reduce",
)


def collective_flags(libtpu_args: str) -> dict[str, str]:
    """The last value of each of _COLLECTIVE_FLAGS in LIBTPU_INIT_ARGS (the
    one libtpu uses), "default" when absent."""
    found = dict.fromkeys(_COLLECTIVE_FLAGS, "default")
    for arg in libtpu_args.split():
        name, _, value = arg.lstrip("-").partition("=")
        if name in found:
            found[name] = value
    return found


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
    # Decided from every rank's index, so the answer is the same on every rank.
    rank_ordered = all(s["mesh_index"] == i for i, s in enumerate(seen))
    n = int(mesh.size)
    direct_rows = max(envs.TPU_TP_PALLAS_ALL_REDUCE_DIRECT_ROWS, 0)
    op = _make_op(mesh, max_bytes, direct_rows)
    rank = torch.tensor([[mesh_index]], dtype=torch.int32, device=device)
    failures = _gather(_selftest(op, rank, mesh_index, n, hidden, device))
    if any(failures):
        raise RuntimeError(
            f"tp_all_reduce: self-test failed (rows per rank): {failures}"
        )
    qar_min_rows = envs.TPU_TP_QAR_MIN_TOKENS
    if qar_min_rows > 0:
        op = _make_op(mesh, max_bytes, direct_rows, _qdq_min_rows(), qar_min_rows)
        seen = _gather(_qar_selftest(op, rank, n, hidden, device, qar_min_rows))
        if len({r["digest"] for r in seen}) != 1 or max(r["rel"] for r in seen) >= 0.03:
            raise RuntimeError(f"qar_all_reduce: self-test failed: {seen}")
        logger.info(
            "tp_all_reduce: qar_all_reduce (int8 ICI payload) on all-reduces of >= %d"
            " rows (TPU_TP_QAR_MIN_TOKENS); self-test passed",
            qar_min_rows,
        )
    qdq_min_rows = _qdq_min_rows()
    if qdq_min_rows and qar_min_rows <= 0:
        # The self-test checks exact sums; the emulation rounds, so it runs on
        # the plain op above and serving gets the rounding one.
        op = _make_op(mesh, max_bytes, direct_rows, qdq_min_rows)
        logger.info(
            "tp_all_reduce: int8 payload emulation ON (TPU_TP_AR_QDQ_ROWS=%d) on"
            " all-reduces of >= %d rows",
            envs.TPU_TP_AR_QDQ_ROWS,
            qdq_min_rows,
        )
    rs_op = None
    if envs.TPU_MINIMAX_M3_SP_QRS:
        rs_min = envs.TPU_MINIMAX_M3_SP_QRS_MIN_TOKENS
        test_rows = max(-(-rs_min // 8) * 8, 8 * qrs.TILE)
        if not rank_ordered:
            logger.warning("tp_reduce_scatter: qar_reduce_scatter not armed: mesh index"
                           " != TP rank on some rank")
        elif n != 8 or not qrs_applies(test_rows, hidden, rs_min):
            logger.warning("tp_reduce_scatter: qar_reduce_scatter not armed: %d ranks,"
                           " hidden %d, min rows %d", n, hidden, rs_min)
        else:
            rs_op = _make_rs_op(mesh, rs_min)
            seen = _gather(_qrs_selftest(rs_op, rank, n, hidden, device, test_rows))
            if max(r["rel"] for r in seen) >= 0.03:
                raise RuntimeError(f"qar_reduce_scatter: self-test failed: {seen}")
            logger.info(
                "tp_reduce_scatter: qar_reduce_scatter (int8 ICI payload) on SP row"
                " reduce-scatters of >= %d rows (TPU_MINIMAX_M3_SP_QRS_MIN_TOKENS);"
                " self-test passed: %s", rs_min, [round(r["rel"], 4) for r in seen],
            )
    if envs.TPU_MINIMAX_M3_SP_HIER_RS:
        rs_min = envs.TPU_MINIMAX_M3_SP_HIER_RS_MIN_TOKENS
        test_rows = max(-(-rs_min // 1024) * 1024, 1024)
        if rs_op is not None:
            raise ValueError("TPU_MINIMAX_M3_SP_HIER_RS and TPU_MINIMAX_M3_SP_QRS are exclusive")
        if not rank_ordered:
            logger.warning("tp_reduce_scatter: hierrs_tc not armed: mesh index != TP rank"
                           " on some rank")
        elif n != 8 or not hier_applies(test_rows, hidden, rs_min):
            logger.warning("tp_reduce_scatter: hierrs_tc not armed: %d ranks, hidden %d,"
                           " min rows %d", n, hidden, rs_min)
        else:
            rs_op = _make_rs_op(mesh, rs_min, hier=True)
            seen = _gather(_hier_selftest(rs_op, rank, n, hidden, device, test_rows))
            if not all(r["exact"] for r in seen) or max(r["rel"] for r in seen) >= 0.01:
                raise RuntimeError(f"hierrs_tc reduce-scatter: self-test failed: {seen}")
            logger.info(
                "tp_reduce_scatter: hierrs_tc (hierarchical TensorCore, bf16) on SP row"
                " reduce-scatters of >= %d rows (TPU_MINIMAX_M3_SP_HIER_RS_MIN_TOKENS);"
                " self-test passed: %s", rs_min, [round(r["rel"], 5) for r in seen],
            )
    gather_op = None
    if envs.TPU_TP_PALLAS_SMALL_ALL_GATHER:
        if not rank_ordered:
            # Blocks land at their mesh index; callers pair them with
            # group-rank-ordered gathers.
            logger.warning("tp_all_gather_small: not armed: mesh index != TP rank"
                           " on some rank")
        else:
            gather_op = _make_gather_op(mesh)
            if not all(_gather(_gather_selftest(gather_op, rank, mesh_index, n, device))):
                raise RuntimeError("tp_all_gather_small: self-test failed")
            logger.info("tp_all_gather_small: armed (TPU_TP_PALLAS_SMALL_ALL_GATHER);"
                        " self-test passed")
    _ARMED = _Armed(
        ranks=tuple(int(r) for r in tp.ranks),
        op=op,
        rank=rank,
        rank_ordered=rank_ordered,
        rs_op=rs_op,
        gather_op=gather_op,
        mesh=mesh,
        max_bytes=max_bytes,
    )
    logger.info(
        "tp_all_reduce: armed over %d ranks (mesh index %d), Pallas up to %d B views"
        " (TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES=%d, kernel cap %d); libtpu %s",
        n,
        mesh_index,
        max_bytes,
        envs.TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES,
        kernel.MAX_VIEW_BYTES,
        collective_flags(os.environ.get("LIBTPU_INIT_ARGS", "")),
    )
    if direct_rows:
        logger.info(
            "tp_all_reduce: TPU_TP_PALLAS_ALL_REDUCE_DIRECT_ROWS=%d: rows that are a"
            " multiple of it all-reduce on their own [rows, %d] layout (no view copy)",
            direct_rows,
            hidden,
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
    """Row reduce-scatter of x over the group of `ranks` with the armed
    qar_reduce_scatter op (rank r gets rows [r T/n, (r + 1) T/n) of the sum),
    or None when it is not armed for this group or tensor (the caller takes
    its usual path). Sizes below the op's minimum take a psum_scatter inside
    the op, so every bucket traces the same graph."""
    armed = _ARMED
    if armed is None or armed.rs_op is None or tuple(ranks) != armed.ranks:
        return None
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        return None
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
def armed_for(ranks) -> _Armed | None:
    """The armed op's record (mesh, mesh index, byte cap) when it serves the
    group of `ranks`, else None: for kernels that run over its mesh."""
    armed = _ARMED
    if armed is None or tuple(ranks) != armed.ranks:
        return None
    return armed
