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
_SELFTEST_ROWS = (8, 24, 32)


@dataclasses.dataclass(frozen=True)
class _Armed:
    ranks: tuple[int, ...]
    op: Any
    rank: torch.Tensor  # this rank's mesh index, int32 [1, 1]


_ARMED: _Armed | None = None
_QAR_TRACED: set[int] = set()
# Fused all-reduce + residual add + RMSNorm (TPU_TP_QAR_FUSE_NORM): eps values
# models asked for at construction, and the ops arm() built for them.
_NORM_EPS: set[float] = set()
_NORM_OPS: dict[float, Any] = {}


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


def _norm_body(n: int, max_bytes: int, axis: str, direct_rows: int, qar_min_rows: int,
               qar_tab: np.ndarray, eps: float):
    """(x, res, w, rank) -> (normed, new_res) of the all-reduce of x: qar sizes
    in one fused kernel call, smaller ones all-reduce as `_body` then add +
    norm in XLA (same rounding points)."""
    ar = _body(n, max_bytes, axis, direct_rows, 0, 0, None)
    in_kernel = envs.TPU_TP_QAR_FUSE_NORM != "xla"

    def body(x, res, w, rank):
        rows, hidden = x.shape
        if 0 < qar_min_rows <= rows:
            if -rows not in _QAR_TRACED:
                _QAR_TRACED.add(-rows)
                logger.warning(
                    "tp_all_reduce: qar_all_reduce+add_norm (%s) lowered for [%d, %d]",
                    "kernel" if in_kernel else "xla", rows, hidden,
                )
            if in_kernel:
                return qar.all_reduce_v2_add_norm(x, res, w[0], eps, rank, qar_tab)
            return qar.add_norm_reference(qar.all_reduce_v2(x, rank, qar_tab), res, w[0], eps)
        return qar.add_norm_reference(ar(x, rank), res, w[0], eps)

    return body


def _make_norm_op(mesh, max_bytes: int, direct_rows: int, qar_min_rows: int, eps: float):
    axis = mesh.axis_names[0]
    qar_tab = qar.layout_table(_qar_devices(mesh))
    specs = (P(axis), P(axis), P(axis), P(axis))
    sharded = jax.shard_map(
        _norm_body(int(mesh.size), max_bytes, axis, direct_rows, qar_min_rows, qar_tab, eps),
        mesh=mesh,
        in_specs=specs,
        out_specs=(P(axis), P(axis)),
        check_vma=False,
    )

    def fn(
        x: jax.Array, res: jax.Array, w: jax.Array, rank: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        args = (x, res, w, rank)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::tp_all_reduce_qar_add_norm_{len(_NORM_OPS)}",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=(P(axis), P(axis)),
    )
    op.register_fake(lambda x, res, w, rank: (torch.empty_like(x), torch.empty_like(x)))
    return op


def request_add_norm(eps: float) -> bool:
    """Called at model construction: whether the model should leave this
    all-reduce un-reduced and call `all_reduce_add_norm` (arm() then builds
    the op for eps; if arming fails the call returns None and the model
    falls back to all-reduce + norm)."""
    if envs.TPU_TP_QAR_FUSE_NORM in ("", "0") or envs.TPU_TP_QAR_MIN_TOKENS <= 0:
        return False
    _NORM_EPS.add(float(eps))
    return True


def all_reduce_add_norm(ranks, x: torch.Tensor, res: torch.Tensor, w: torch.Tensor,
                        eps: float):
    """(normed, new_res) of fused_add_rms_norm(all_reduce(x), res, w, eps) (w
    f32 [hidden], the norm's effective weight), or None when not armed."""
    armed = _ARMED
    op = _NORM_OPS.get(float(eps))
    if armed is None or op is None or tuple(ranks) != armed.ranks:
        return None
    if x.dtype != torch.bfloat16 or x.dim() != 2 or res.dtype != torch.bfloat16:
        return None
    return op(x, res, w.reshape(1, -1).to(torch.float32), armed.rank)


def _norm_selftest(op, rank, n: int, hidden: int, device, rows: int, eps: float) -> dict:
    """Exact integer partials (the all-reduce is exact) -> normed / new_res
    against a numpy f64 reference: max abs error in bf16 ulps of the result."""
    x = _selftest_inputs(n, rows, hidden)
    me = int(rank.reshape(-1)[0].item())
    rng = np.random.default_rng(7)
    res = np.round(rng.standard_normal((rows, hidden)) * 4).astype(np.float32)
    w = (1.0 + rng.standard_normal(hidden) * 0.1).astype(np.float32)
    to_dev = lambda a, dt: torch.from_numpy(a).to(device).to(dt)
    normed, new_res = op(to_dev(x[me], torch.bfloat16), to_dev(res, torch.bfloat16),
                         to_dev(w.reshape(1, -1), torch.float32), rank)
    nr = (x.sum(0) + res).astype(np.float64)  # small integers: exact in bf16
    ref = nr / np.sqrt((nr * nr).mean(-1, keepdims=True) + eps) * w
    got = normed.to(torch.float32).cpu().numpy().astype(np.float64)
    ulp = np.maximum(np.abs(ref), 1e-30) * 2.0**-8
    return dict(rank=me, res_ok=bool(np.array_equal(new_res.to(torch.float32).cpu().numpy(), nr)),
                ulps=float(np.max(np.abs(got - ref) / ulp)),
                digest=float((got.reshape(-1) * np.arange(1, got.size + 1)).sum()))


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
        for eps in sorted(_NORM_EPS):
            nop = _make_norm_op(mesh, max_bytes, direct_rows, qar_min_rows, eps)
            seen = []
            for rows in (qar_min_rows, 24):  # the fused kernel and the XLA form
                seen += _gather(_norm_selftest(nop, rank, n, hidden, device, rows, eps))
            if (not all(r["res_ok"] for r in seen) or max(r["ulps"] for r in seen) > 2.0
                    or len({r["digest"] for r in seen[:n]}) != 1):
                raise RuntimeError(f"qar add_norm: self-test failed: {seen}")
            _NORM_OPS[eps] = nop
            logger.warning(
                "tp_all_reduce: TPU_TP_QAR_FUSE_NORM=%s: all-reduce + residual add + RMSNorm"
                " fused (eps %g); self-test passed (max %.3f bf16 ulp)",
                envs.TPU_TP_QAR_FUSE_NORM, eps, max(r["ulps"] for r in seen),
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
