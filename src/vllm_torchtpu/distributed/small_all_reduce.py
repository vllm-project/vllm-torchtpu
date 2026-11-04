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
"""Torch bridge for the Pallas small all-reduce (TPU_SMALL_ALL_REDUCE_KERNEL).

A tensor-parallel all-reduce of a decode-size bf16 activation (for GLM-5.2,
bf16[16, 6144] at every attention and FFN exit) costs about 14 us as XLA's
TensorCore all-reduce. `kernels/collectives/small_all_reduce.py` does the
same sum in one Pallas call with one barrier and two exchange rounds
(reduce-scatter, then all-gather): 10.5 us against 15.2 us for XLA's
all-reduce in the same torch_tpu program on 8 tpu7x devices (device time per
call). Above about 48 tokens XLA's all-reduce is faster, so larger tensors
keep it (TPU_SMALL_ALL_REDUCE_MAX_BYTES).

`TpuDeviceCommunicator.all_reduce` calls `maybe_all_reduce`, which returns
None (use the usual all-reduce) unless:

- the flag is on and `arm()` has prepared this communicator's group: the
  group spans the whole torch.distributed world (TP = world; torch_tpu
  meshes must span the world) and has 2 to TPU_SMALL_ALL_REDUCE_MAX_RANKS
  ranks (default 8: the size whose speed was measured);
- the tensor is bf16 with a last dimension the kernel can split into n
  lane-aligned chunks (a multiple of n x 128).

The torch op it calls takes the tensor alone and is one exported JAX program
per shape. The size choice happens inside it, where the row count is static:
at most TPU_SMALL_ALL_REDUCE_MAX_BYTES bytes runs the kernel, a larger tensor
runs `lax.psum` (XLA's all-reduce) in the same program. The torch graph
therefore has the same structure and inputs for every token bucket and for
every traced model (target, MTP drafter), which vllm-torchtpu's per-bucket
compile (compilation/shape_variants.py) requires.

`arm()` runs once after model load, on every rank (collective over the TP
group). It builds the mesh, runs the kernel once on inputs whose sum is
exact, compares with the expected sum on every rank, and turns the path off
on every rank if any rank fails.
"""

import dataclasses
import faulthandler
import traceback
from typing import Any

import jax
import numpy as np
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.collectives import small_all_reduce as kernel
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# A wrong mesh hangs the self-test: dump every stack and exit instead.
SELFTEST_TIMEOUT_S = 300


@dataclasses.dataclass(frozen=True)
class _Context:
    mesh: Any
    index: int  # this rank's position in the mesh
    n: int
    algo: str
    max_bytes: int
    device: torch.device  # this rank's TPU device (self-test inputs)
    op: Any


# Communicator id -> context, filled by arm().
_CONTEXTS: dict[int, _Context] = {}
_OPS: dict[tuple, Any] = {}
# The per-rank results of the last arm(), kept for diagnostics.
LAST_ARM: list | None = None


def _mesh_for_world() -> tuple[Any, int] | str:
    """The 1-D mesh over the world and this rank's index in it.

    Reuses the EP mesh when the EP group is the whole world, so every Pallas
    kernel in one compiled module shares one device assignment (torch_tpu
    rejects two orders in one module); otherwise torch_tpu's rank-ordered
    pallas mesh, which is also the order of the DCP and PCP meshes.
    """
    import torch.distributed as dist

    world = int(dist.get_world_size())
    rank = int(dist.get_rank())
    try:
        from vllm_torchtpu.distributed.ep_mesh import (
            build_ep_mesh,
            ep_mesh_index,
            get_ep_group,
        )

        ep = get_ep_group()
        if ep is not None and tuple(int(r) for r in ep.ranks) == tuple(range(world)):
            mesh, index = build_ep_mesh(), ep_mesh_index()
            if mesh is not None and index is not None and len(mesh.axis_names[0]) == 1:
                return mesh, int(index)
    except Exception as exc:  # noqa: BLE001 - fall back to the rank-ordered mesh
        logger.info("small all-reduce: EP mesh unavailable (%s)", exc)
    from torch_tpu._internal.pallas import get_pallas_mesh

    # One-letter axis name: see _get_op.
    mesh = get_pallas_mesh(axis_names=("r",))
    if int(mesh.size) != world:
        return f"pallas mesh has {mesh.size} devices, world is {world}"
    return mesh, rank


def _choose_algo() -> str:
    """flat2 unless TPU_SMALL_ALL_REDUCE_ALGO names another algorithm. The
    algorithms are correct on any mesh order; hier1/hier2 are faster only when
    mesh positions (2c, 2c + 1) share a chip, and measured no faster than
    flat2 at the decode size on tpu7x."""
    algo = envs.TPU_SMALL_ALL_REDUCE_ALGO
    return "flat2" if algo in (None, "auto") else algo


def _eligible(n: int, rows: int, h: int, algo: str, max_bytes: int) -> bool:
    return rows * h * 2 <= max_bytes and kernel.supported(n, rows, h, algo)


def _get_op(mesh, n: int, algo: str, max_bytes: int):
    key = (tuple(int(d.id) for d in mesh.devices.flat), algo, max_bytes)
    op = _OPS.get(key)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    # torch_tpu's jax_op reads each partition-spec entry as a sequence of axis
    # names, and JAX hands a one-axis entry back as a plain string, so an axis
    # name longer than one letter is split into letters and fails the lookup.
    if len(axis) != 1:
        raise ValueError(
            f"mesh axis {axis!r}: torch_tpu jax_op needs a one-letter name"
        )
    spec = P(axis)

    # The kernel reads its rank from the mesh (lax.axis_index inside the
    # kernel), so the op's only operand is the tensor: a rank tensor would be
    # an extra graph input that every trace of every model must agree on.
    # The size choice is made here, where the row count is static per
    # exported program, and never in the torch graph: a branch on the token
    # count while tracing would leave a shape guard that splits the compile
    # buckets between traces.
    def body(x):
        rows, h = x.shape  # static: torch_tpu exports one program per shape
        if _eligible(n, rows, h, algo, max_bytes):
            return kernel.build(kernel.Plan(n, rows, h, algo, axis=axis))(x)
        return lax.psum(x, axis)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,), out_specs=spec, check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(x: jax.Array) -> jax.Array:
        return sharded(jax.device_put(x, NamedSharding(mesh, spec)))

    from torch_tpu._internal import pallas

    name = "pallas::small_all_reduce" + (f"_{len(_OPS)}" if _OPS else "")
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name, fn, mesh=mesh, input_partition_specs=(spec,), output_partition_specs=spec
    )
    op.register_fake(lambda x: torch.empty_like(x))
    _OPS[key] = op
    return op


def maybe_all_reduce(communicator: Any, x: torch.Tensor) -> torch.Tensor | None:
    """The kernel's sum of `x` over the communicator's group, or None when
    the usual all-reduce must run. Reads only tensor metadata and cached
    state, so a compiled graph carries no per-call host work for it."""
    ctx = _CONTEXTS.get(id(communicator))
    if ctx is None or not envs.TPU_SMALL_ALL_REDUCE_KERNEL:
        return None
    if x.dtype != torch.bfloat16 or x.dim() < 1:
        return None
    h = x.shape[-1]
    if not isinstance(h, int) or not kernel.supported(ctx.n, 16, h, ctx.algo):
        return None
    # No branch on the row count here (see _get_op): the op takes every row
    # count and runs XLA's all-reduce inside for the large ones.
    y = ctx.op(x.reshape(-1, h))
    return y.reshape(x.shape)


def _selftest(ctx: _Context, device) -> list[str]:
    """Run the op at the decode bucket and the largest eligible bucket on
    integer inputs whose sum is exact in bf16 in any order; returns this
    rank's failures."""
    failures = []
    h = 6144 if kernel.supported(ctx.n, 16, 6144, ctx.algo) else 256 * ctx.n
    for rows in sorted({16, max(16, (ctx.max_bytes // (2 * h)) // 16 * 16)}):
        if not _eligible(ctx.n, rows, h, ctx.algo, ctx.max_bytes):
            continue
        d = np.arange(ctx.n)[:, None, None]
        r = np.arange(rows)[None, :, None]
        c = np.arange(h)[None, None, :]
        x = (((d + 1) * 3 + r * 5 + (c // 128) * 7 + d * c) % 11 - 5).astype(np.float32)
        want = x.sum(0)
        mine = torch.from_numpy(x[ctx.index]).to(torch.bfloat16).to(device)
        got = ctx.op(mine).float().cpu().numpy()
        if not np.array_equal(got, want):
            failures.append(
                f"rows={rows} h={h} max_abs_err={float(np.abs(got - want).max())}"
            )
    return failures


def group_refusal(tp_ranks: tuple[int, ...], world: int, max_ranks: int) -> str | None:
    """Why the kernel must not arm for this TP group, or None.

    The group must span the whole world (torch_tpu meshes span the world)
    and have 2 to `max_ranks` ranks. The rank limit exists because speed was
    measured only on one 8-device host; a larger group (for example TP16
    over two hosts) keeps XLA's all-reduce unless the limit is raised.
    """
    n = len(tp_ranks)
    if n < 2:
        return "the TP group has one rank"
    if tp_ranks != tuple(range(world)):
        return f"TP ranks {tp_ranks} do not span the world of {world}"
    if n > max_ranks:
        return f"{n} ranks > TPU_SMALL_ALL_REDUCE_MAX_RANKS={max_ranks}"
    return None


def _build_context() -> _Context | str:
    """The mesh, index and op for this rank, or a refusal reason."""
    got = _mesh_for_world()
    if isinstance(got, str):
        return got
    mesh, index = got
    n = int(mesh.size)
    algo = _choose_algo()
    device = torch.empty((), device="tpu").device
    max_bytes = int(envs.TPU_SMALL_ALL_REDUCE_MAX_BYTES)
    op = _get_op(mesh, n, algo, max_bytes)
    return _Context(mesh, index, n, algo, max_bytes, device, op)


def arm() -> bool:
    """Prepare the TP group's communicator for the kernel; collective over
    the world. Returns whether the path is on."""
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    if not envs.TPU_SMALL_ALL_REDUCE_KERNEL:
        return False
    tp = parallel_state.get_tp_group()
    comm = getattr(tp, "device_communicator", None)
    world = int(dist.get_world_size())
    reason = (
        "the TP group has no device communicator"
        if comm is None
        else group_refusal(
            tuple(int(r) for r in tp.ranks),
            world,
            int(envs.TPU_SMALL_ALL_REDUCE_MAX_RANKS),
        )
    )
    ctx = None
    if reason is None:
        # A failure while building the op turns the path off on every rank
        # (agreed below) instead of failing model load.
        try:
            ctx = _build_context()
        except Exception:  # noqa: BLE001 - reported and agreed on below
            reason = traceback.format_exc()[-1500:]
        if isinstance(ctx, str):
            reason, ctx = ctx, None
    failures = []
    if ctx is not None:
        faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
        try:
            failures = _selftest(ctx, ctx.device)
        except Exception:  # noqa: BLE001 - reported and agreed on below
            failures = [traceback.format_exc()[-1500:]]
        finally:
            faulthandler.cancel_dump_traceback_later()
    # Every rank must take the same path, or a rank inside the kernel would
    # wait forever for a peer that chose the usual all-reduce.
    mine = {
        "ok": ctx is not None and not failures,
        "reason": reason,
        "failures": failures,
    }
    seen: list = [None] * world
    dist.all_gather_object(seen, mine, group=tp.cpu_group)
    global LAST_ARM
    LAST_ARM = seen
    on = all(s["ok"] for s in seen)
    if on:
        _CONTEXTS[id(comm)] = ctx
        logger.info(
            "small all-reduce: on | n=%d algo=%s max_bytes=%d mesh=%s",
            ctx.n,
            ctx.algo,
            ctx.max_bytes,
            [int(d.id) for d in ctx.mesh.devices.flat],
        )
    else:
        _CONTEXTS.pop(id(comm), None)
        logger.warning("small all-reduce: off | per rank: %s", seen)
    return on
