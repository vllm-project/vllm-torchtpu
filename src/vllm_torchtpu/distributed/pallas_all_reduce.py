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
from vllm_torchtpu.kernels.collectives import tp_all_reduce as kernel
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# A wrong rank or mesh hangs the self-test's barrier: dump every stack and exit.
SELFTEST_TIMEOUT_S = 300
_SELFTEST_ROWS = (8, 24)


@dataclasses.dataclass(frozen=True)
class _Armed:
    ranks: tuple[int, ...]
    op: Any
    rank: torch.Tensor  # this rank's mesh index, int32 [1, 1]


_ARMED: _Armed | None = None


def _qdq(x: jax.Array) -> jax.Array:
    """bf16 [rows, hidden] -> int8 with one absmax scale per (32 rows, lane) ->
    bf16: the payload rounding of kernels/collectives/qar_all_reduce.py."""
    rows = x.shape[0]
    t = qar.padded_rows(rows)
    xf = jnp.pad(x.astype(jnp.float32), ((0, t - rows), (0, 0)))
    return qar.dequantize(*qar.quantize(xf))[:rows].astype(x.dtype)


def _body(n: int, max_bytes: int, axis: str, qdq_min_rows: int = 0,
          qar_min_rows: int = 0, qar_tab: np.ndarray | None = None,
          qar_v3_min_rows: int = 0):
    def body(x, rank):
        rows, hidden = x.shape
        # Static per bucket: decided at trace time inside the op.
        if qar_tab is not None and 0 < qar_min_rows <= rows:
            return qar.all_reduce_v2(x, rank, qar_tab)
        if (qar_tab is not None and 0 < qar_v3_min_rows <= rows
                and qar.padded_rows(rows) <= 256):
            return qar.all_reduce_v3(x, rank, qar_tab)
        qdq = 0 < qdq_min_rows <= rows
        if qdq:
            x = _qdq(x)
        if 0 < kernel.view_bytes(rows, hidden, n) <= max_bytes:
            out = kernel.all_reduce(x, rank, n)
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


def _make_op(mesh, max_bytes: int, qdq_min_rows: int = 0, qar_min_rows: int = 0,
             qar_v3_min_rows: int = 0):
    axis = mesh.axis_names[0]
    qar_tab = (qar.layout_table(list(mesh.devices.flat))
               if (qar_min_rows or qar_v3_min_rows) else None)
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        _body(int(mesh.size), max_bytes, axis, qdq_min_rows, qar_min_rows, qar_tab,
              qar_v3_min_rows),
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
        + ("_qar" if qar_min_rows else "")
        + ("_qar3" if qar_v3_min_rows else ""),
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
    gen = torch.Generator().manual_seed(1234)
    parts = torch.randn((n, rows, hidden), generator=gen).to(torch.bfloat16)
    me = int(rank.reshape(-1)[0].item())
    out = op(parts[me].to(device), rank).float().cpu()
    exact = parts.double().sum(0)
    rel = float((out.double() - exact).norm() / exact.norm())
    digest = float((out.double() * torch.arange(1, out.numel() + 1).view(out.shape)).sum())
    return dict(rank=me, rel=rel, digest=digest)


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
    n = int(mesh.size)
    op = _make_op(mesh, max_bytes)
    rank = torch.tensor([[mesh_index]], dtype=torch.int32, device=device)
    failures = _gather(_selftest(op, rank, mesh_index, n, hidden, device))
    if any(failures):
        raise RuntimeError(
            f"tp_all_reduce: self-test failed (rows per rank): {failures}"
        )
    qar_min_rows = envs.TPU_TP_QAR_MIN_TOKENS
    qar_v3_min_rows = envs.TPU_TP_QAR_V3_MIN_TOKENS
    if qar_min_rows > 0 or qar_v3_min_rows > 0:
        op = _make_op(mesh, max_bytes, _qdq_min_rows(), qar_min_rows, qar_v3_min_rows)
        for test_rows in sorted({r for r in (qar_min_rows, qar_v3_min_rows) if r > 0}):
            seen = _gather(_qar_selftest(op, rank, n, hidden, device, test_rows))
            if len({r["digest"] for r in seen}) != 1 or max(r["rel"] for r in seen) >= 0.03:
                raise RuntimeError(f"qar_all_reduce: self-test failed at {test_rows} rows: {seen}")
        logger.info(
            "tp_all_reduce: qar_all_reduce (int8 ICI payload) on all-reduces of >= %d"
            " rows (TPU_TP_QAR_MIN_TOKENS), v3 on %d..256 rows (TPU_TP_QAR_V3_MIN_TOKENS);"
            " self-test passed",
            qar_min_rows,
            qar_v3_min_rows,
        )
    qdq_min_rows = _qdq_min_rows()
    if qdq_min_rows and qar_min_rows <= 0:
        # The self-test checks exact sums; the emulation rounds, so it runs on
        # the plain op above and serving gets the rounding one.
        op = _make_op(mesh, max_bytes, qdq_min_rows)
        logger.info(
            "tp_all_reduce: int8 payload emulation ON (TPU_TP_AR_QDQ_ROWS=%d) on"
            " all-reduces of >= %d rows",
            envs.TPU_TP_AR_QDQ_ROWS,
            qdq_min_rows,
        )
    _ARMED = _Armed(ranks=tuple(int(r) for r in tp.ranks), op=op, rank=rank)
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
