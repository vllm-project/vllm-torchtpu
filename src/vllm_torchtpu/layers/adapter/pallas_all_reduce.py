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
"""Decode-sized TP all-reduce as a one-hop Pallas kernel
(TPU_PALLAS_SMALL_ALL_REDUCE_MAX_ROWS).

XLA lowers a small bf16 all-reduce over the 16 ranks of a tpu7x 2x2x2 slice
with a multi-phase emitter: one dependent exchange per torus dimension plus
one between the two cores of a chip. For a [16, 6144] partial that is about
15 us of fixed cost, while a small 16-rank all-to-all in the same program
takes about 6 us. The all-reduce of kernels/experimental/moe_tp_decode/
all_reduce.py exchanges once in each direction (a reduce-scatter of column
chunks straight to their owner, an f32 sum, an all-gather), so its fixed cost
is close to that of the all-to-all.

`arm` runs once at the end of TPUModelRunner.load_model. It builds one torch
op over a 1-D mesh of the TP group and checks it on every row count it will
serve. TpuDeviceCommunicator.all_reduce then hands every bf16 [rows, hidden]
all-reduce over the TP group's ranks to `maybe_all_reduce`. This covers the
attention exit and the dense FFN exit (RowParallelLinear), the MoE FFN exit
(the runner's tensor_model_parallel_all_reduce) and the embedding, because
all of them call the same GroupCoordinator.all_reduce.

`maybe_all_reduce` routes an all-reduce to the op only when its row count is
one of the armed row counts. Every other all-reduce, for example a prefill
step of 4096 tokens, keeps the communicator's own XLA all-reduce, as it would
with the op unarmed. The row count is a concrete int at that point: vLLM
compiles the graph once per token count (the TPU platform disables dynamic
compile ranges), and torch_tpu runs vllm.all_reduce's implementation with
inputs of that one shape. A symbolic row count (torch.SymInt) is not routed,
because reading its value would add a shape guard. Every rank compiles the
same shapes, so every rank takes the same path.

All rows travel: the kernel's n_valid is the padded row count. The true token
count exists on device only as token_padding's mask, which is set only with
TPU_MOE_SKIP_PADDED_TOKENS and only for the forwards that call mark_padding,
so it is not a value every caller of this all-reduce can rely on.
"""

import dataclasses
import faulthandler
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.layers.adapter.moe_tp_decode import (
    SELFTEST_TIMEOUT_S,
    _gather,
    _rank_tensor,
    _topology,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class _Armed:
    """The armed op and the all-reduces it serves."""

    op: Any  # torch op (x, rank) -> reduced x
    rank: torch.Tensor  # this rank's mesh index, int32 [1, 1]
    ranks: tuple[int, ...]  # global ranks of the TP group
    hidden: int
    buckets: tuple[int, ...]  # row counts the kernel serves


_ARMED: _Armed | None = None


def row_buckets(n_ranks: int, hidden: int, max_rows: int, info) -> tuple[int, ...]:
    """Row counts the kernel serves for a [rows, hidden] bf16 partial over
    n_ranks: every multiple of the bf16 sublane tiling up to max_rows that
    the kernel supports.

    Example: tpu7x (16 bf16 rows per tile), n_ranks 16, hidden 6144 and
    max_rows 40 give (16, 32); max_rows 8 gives ().
    """
    tile_rows = info.get_sublane_tiling(jnp.bfloat16)
    return tuple(
        m
        for m in range(tile_rows, max_rows + 1, tile_rows)
        if all_reduce.supported(n_ranks, m, hidden, info)
    )


def _op_body(n_ranks: int, buckets: tuple[int, ...]):
    """The op's per-rank body under shard_map: x is this rank's bf16
    [rows, hidden] partial and rank its int32 [1, 1] mesh index. Returns the
    sum over the n_ranks ranks from the Pallas kernel.

    Raises:
        ValueError: At trace time, when rows is not in `buckets`.
            `maybe_all_reduce` routes only the armed row counts here.
    """

    def body(x, rank):
        rows, hidden = x.shape  # static: one exported program per token count
        if rows not in buckets:
            raise ValueError(
                f"pallas_all_reduce: {rows} rows reached the op; it serves "
                f"only rows {list(buckets)}"
            )
        spec = all_reduce.make_spec(n_ranks, rows, hidden)
        # n_valid = rows: every row is reduced (see the module docstring).
        meta = jnp.stack(
            [jnp.asarray(rows, jnp.int32), rank.reshape(()).astype(jnp.int32)]
        )
        return all_reduce.standalone(spec)(meta, x)

    return body


def _build_op(mesh, buckets: tuple[int, ...]):
    """torch op (x [rows, hidden] bf16, rank int32 [1, 1]) -> reduced x."""
    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        _op_body(int(mesh.size), buckets),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(axis),
        check_vma=False,
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return sharded(
            *(
                jax.device_put(a, NamedSharding(mesh, s))
                for a, s in zip((x, rank), specs)
            )
        )

    # PartitionSpecs, as moe_tp_decode passes them (torch_tpu's annotation is
    # narrower).
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::small_tp_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    return op


def _selftest(op, rank, mesh_index, n_ranks, hidden, rows, device) -> list[int]:
    """Run the op on partials with exact sums at every row count in `rows`;
    returns the row counts whose result on this rank is wrong. Collective:
    every rank makes the same calls."""
    failures = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for m in rows:
            spec = all_reduce.make_spec(n_ranks, m, hidden)
            x = all_reduce.selftest_inputs(spec)
            mine = torch.from_numpy(x[mesh_index]).to(torch.bfloat16).to(device)
            out = op(mine, rank).to(torch.float32).cpu().numpy()
            if not all_reduce.selftest_ok(spec, x, out, m):
                failures.append(m)
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def arm(hidden: int, device: torch.device) -> bool:
    """Route the TP group's bf16 [rows, hidden] all-reduces with rows up to
    TPU_PALLAS_SMALL_ALL_REDUCE_MAX_ROWS through the Pallas all-reduce.

    Collective over the TP group: every rank must call it. The ranks first
    agree on the plan, then run the op on known inputs at every served row
    count.

    Args:
        hidden: The model's hidden size, the last dimension of the TP
            all-reduces to serve.
        device: This rank's TPU device.

    Returns:
        Whether the op is armed. False when the topology or the shape cannot
        be served; the reason is logged and the XLA all-reduce stays.

    Raises:
        RuntimeError: The ranks disagree on the plan or the self-test failed.
    """
    global _ARMED
    max_rows = envs.TPU_PALLAS_SMALL_ALL_REDUCE_MAX_ROWS
    topo = _topology()  # collective: first, on every rank
    info = pltpu.get_tpu_info()
    if isinstance(topo, str):
        reason, buckets, mesh_index = topo, (), -1
    else:
        mesh_index = topo.mesh_index
        buckets = row_buckets(int(topo.mesh.size), hidden, max_rows, info)
        reason = (
            None
            if buckets
            else f"no row count up to {max_rows} is served at hidden {hidden}"
        )
    plan = dict(reason=reason, buckets=buckets, hidden=hidden)
    seen = _gather(dict(plan=plan, mesh_index=mesh_index))
    indices = sorted(s["mesh_index"] for s in seen)
    if any(s["plan"] != plan for s in seen) or (
        reason is None and indices != list(range(len(seen)))
    ):
        raise RuntimeError(
            "pallas_all_reduce: TP ranks disagree on the plan or the mesh "
            f"indices; their programs and the kernel's barrier would not "
            f"match. Per rank: {seen}"
        )
    if reason is not None:
        logger.warning("pallas_all_reduce: not armed: %s", reason)
        return False
    assert not isinstance(topo, str)
    n_ranks = int(topo.mesh.size)
    op = _build_op(topo.mesh, buckets)
    rank = _rank_tensor(device, mesh_index)
    failures = _selftest(op, rank, mesh_index, n_ranks, hidden, buckets, device)
    seen = _gather(dict(mesh_index=mesh_index, failures=failures))
    if any(s["failures"] for s in seen):
        raise RuntimeError(
            "pallas_all_reduce: the self-test failed (per rank: mesh index, "
            f"failing row counts): {seen}"
        )
    tp_ranks = tuple(int(r) for r in parallel_state.get_tp_group().ranks)
    _ARMED = _Armed(op=op, rank=rank, ranks=tp_ranks, hidden=hidden, buckets=buckets)
    logger.info(
        "pallas_all_reduce: armed for bf16 [rows, %d] over %d ranks, rows %s "
        "(mesh index %d)",
        hidden,
        n_ranks,
        list(buckets),
        mesh_index,
    )
    return True


def maybe_all_reduce(x: torch.Tensor, ranks) -> torch.Tensor | None:
    """The sum of x over the group `ranks` through the armed op, or None when
    the op does not serve this all-reduce. The caller then runs its own XLA
    all-reduce.

    The op serves x only when it is armed for the group `ranks`, x is bf16
    [rows, hidden] at the armed hidden size, and rows is a concrete int in the
    armed row counts. Example: armed with rows (16, 32), a [32, 6144] decode
    partial goes to the op, while a [4096, 6144] prefill partial and a
    [24, 6144] partial return None.

    Args:
        x: This rank's partial.
        ranks: Global ranks of the group to reduce over.

    Returns:
        The reduced tensor, or None.
    """
    armed = _ARMED
    if (
        armed is None
        or tuple(ranks) != armed.ranks
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.shape[1] != armed.hidden
    ):
        return None
    rows = x.shape[0]
    # A symbolic row count is a torch.SymInt, not an int. The isinstance check
    # reads no value from it, so it adds no guard and no recompile, and the
    # XLA all-reduce is kept. Dynamo traces this function (the embedding's
    # all-reduce reaches it inside a compiled graph), so the check must also
    # be traceable: torch.fx's is_concrete_int is on Dynamo's skip list.
    if not isinstance(rows, int) or rows not in armed.buckets:
        return None
    return armed.op(x, armed.rank)
