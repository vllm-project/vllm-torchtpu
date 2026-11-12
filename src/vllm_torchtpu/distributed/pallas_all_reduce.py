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
    # op(x, rank, prev): the same all-reduce, started only once `prev` exists.
    ordered_op: Any = None


_ARMED: _Armed | None = None
# While a chain is open (`begin_chain`), every armed all-reduce starts after
# the previous one in the chain: XLA keeps them in program order on the
# SparseCore and cannot combine two of them into one that both consumers
# would wait for.
_CHAIN: list | None = None


def _body(n: int, max_bytes: int, axis: str):
    def body(x, rank):
        rows, hidden = x.shape
        if 0 < kernel.view_bytes(rows, hidden, n) <= max_bytes:
            return kernel.all_reduce(x, rank, n)
        return lax.psum(x, axis)

    return body


def _ordered_body(n: int, max_bytes: int, axis: str):
    body = _body(n, max_bytes, axis)

    def ordered(x, rank, prev):
        # An exact zero read from `prev` (NaN/inf mapped to 0 first) added to
        # one element: a data dependency that changes no value.
        zero = jnp.nan_to_num(prev[0, 0], nan=0.0, posinf=0.0, neginf=0.0) * 0
        return body(x.at[0, 0].add(zero.astype(x.dtype)), rank)

    return ordered


def _make_op(mesh, max_bytes: int, ordered: bool = False):
    axis = mesh.axis_names[0]
    specs = (P(axis),) * (3 if ordered else 2)
    make_body = _ordered_body if ordered else _body
    sharded = jax.shard_map(
        make_body(int(mesh.size), max_bytes, axis),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(axis),
        check_vma=False,
    )

    def put(args):
        return (jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))

    # torch_tpu reads the operand types from these annotations.
    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return sharded(*put((x, rank)))

    def fn_ordered(x: jax.Array, rank: jax.Array, prev: jax.Array) -> jax.Array:
        return sharded(*put((x, rank, prev)))

    # PartitionSpecs, as other jax_op call sites pass them (torch_tpu's
    # annotation is narrower).
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::tp_all_reduce_ordered" if ordered else "pallas::tp_all_reduce",
        fn_ordered if ordered else fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *_: torch.empty_like(x))
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
    _ARMED = _Armed(
        ranks=tuple(int(r) for r in tp.ranks),
        op=op,
        rank=rank,
        ordered_op=_make_op(mesh, max_bytes, ordered=True),
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
    chain = _CHAIN
    if chain is None or armed.ordered_op is None:
        return armed.op(x, armed.rank)
    out = armed.op(x, armed.rank) if not chain else armed.ordered_op(
        x, armed.rank, chain[-1]
    )
    chain.append(out)
    return out


def armed() -> bool:
    """Whether the op is armed (so `begin_chain` orders the all-reduces)."""
    return _ARMED is not None and _ARMED.ordered_op is not None


def begin_chain() -> None:
    """Order every armed all-reduce from here to `end_chain` in program
    order (each starts once the previous one's output exists)."""
    global _CHAIN
    _CHAIN = []


def end_chain() -> None:
    global _CHAIN
    _CHAIN = None
