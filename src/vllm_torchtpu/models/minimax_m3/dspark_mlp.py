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
"""The MiniMax DSpark draft's dense MLP as one Pallas call at decode rows.

With ``TPU_DSPARK_FUSED_MLP``, `arm_fused_mlp` (once after load, when the
linears hold their K-major e4m3 weights) hands each draft layer's MLP to one
torch op: kernels/dspark_mlp.py at up to MAX_ROWS rows, the same math in XLA
above (decided on the static row count inside the op, so every bucket traces
the same graph). With ``fold_all_reduce`` (and the Pallas all-reduce armed,
distributed/pallas_all_reduce.py) the op also does the TP all-reduce after
the MLP, in the kernel's epilogue at decode rows and through the armed op's
own reduction above; otherwise that all-reduce is unchanged.
"""

import faulthandler

import jax
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)

from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.kernels import dspark_mlp as kernel
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, e4m3_mlp_operands

logger = init_logger(__name__)

# Non-persistent buffers of an armed MLP, in kernel.fused_mlp's order.
_BUFFERS = ("_fused_w_gu", "_fused_s_gu", "_fused_w_d", "_fused_s_d")
# Rows of the all-reduce self-test: the epilogue's fold, trim and full paths.
_SELFTEST_ROWS = (2, 8, 24)
SELFTEST_TIMEOUT_S = 300


def _mlp(
    x: jax.Array, w_gu: jax.Array, s_gu: jax.Array, w_d: jax.Array, s_d: jax.Array
) -> jax.Array:
    if kernel.supported(x.shape[0], x.shape[1], w_d.shape[0]):
        return kernel.fused_mlp(x, w_gu, s_gu, w_d, s_d)
    return kernel.reference(x, w_gu, s_gu, w_d, s_d)


def _fake(x, *operands):
    return torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)


# Registered at import: Dynamo cannot trace a registration inside a forward.
_op = pallas.jax_op("pallas::dspark_fused_mlp", _mlp)
_op.register_fake(_fake)


def _make_reducing_op(armed):
    """The MLP plus the TP all-reduce over the armed all-reduce's mesh, as a
    torch op of (x, *operands, rank)."""
    mesh = armed.mesh
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    reduce = pallas_all_reduce.reduce_body(n, armed.max_bytes, axis)

    def body(x, w_gu, s_gu, w_d, s_d, rank):
        m, h = x.shape
        if kernel.supported(m, h, w_d.shape[0], n):
            return kernel.fused_mlp(x, w_gu, s_gu, w_d, s_d, rank=rank, n=n)
        return reduce(kernel.reference(x, w_gu, s_gu, w_d, s_d), rank)

    specs = (P(axis),) * 6
    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        x: jax.Array,
        w_gu: jax.Array,
        s_gu: jax.Array,
        w_d: jax.Array,
        s_d: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        args = (x, w_gu, s_gu, w_d, s_d, rank)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::dspark_fused_mlp_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(_fake)
    return op


class MiniMaxM3DSparkMLP(MiniMaxM3MLP):
    """MiniMaxM3MLP that runs as the fused op once `arm_fused_mlp` armed it."""

    # The armed op folding in the all-reduce (with a buffer `_mesh_rank`,
    # this rank's mesh index).
    _reducing_op = None

    def forward(self, x):
        if (
            getattr(self, _BUFFERS[0], None) is None
            or not isinstance(x, torch.Tensor)
            or x.dim() != 2
            or x.dtype != torch.bfloat16
        ):
            return super().forward(x)
        operands = [getattr(self, name) for name in _BUFFERS]
        if self._reducing_op is not None:
            return self._reducing_op(x, *operands, self._mesh_rank)
        out = _op(x, *operands)
        if self.down_proj.reduce_results and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out


def _selftest(mlp: MiniMaxM3DSparkMLP, op, rank: torch.Tensor) -> bool:
    """Whether the reducing op gives this rank the bits of the kernel plus the
    armed all-reduce. Collective over the TP group; a wrong rank or mesh
    hangs the kernel, so a timeout dumps every stack and exits."""
    operands = [getattr(mlp, name) for name in _BUFFERS]
    gen = torch.Generator().manual_seed(1 + get_tp_group().rank_in_group)
    hidden = operands[0].shape[0]
    ok = True
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for rows in _SELFTEST_ROWS:
            x = torch.randn(rows, hidden, generator=gen).to(torch.bfloat16)
            x = x.to(operands[0].device)
            want = tensor_model_parallel_all_reduce(_op(x, *operands))
            got = op(x, *operands, rank)
            ok &= bool(torch.equal(got.cpu(), want.cpu()))
    finally:
        faulthandler.cancel_dump_traceback_later()
    return ok


def _arm_all_reduce(mlps: list[MiniMaxM3DSparkMLP]) -> bool:
    """Fold the TP all-reduce into every armed MLP's op; collective."""
    tp = get_tp_group()
    armed = pallas_all_reduce.armed_for(tp.ranks)
    if armed is None or armed.mesh is None or tp.world_size == 1:
        logger.warning(
            "DSpark draft: the MLP keeps its TP all-reduce: the Pallas "
            "all-reduce is not armed for the TP group."
        )
        return False
    if not all(mlp.down_proj.reduce_results for mlp in mlps):
        return False
    op = _make_reducing_op(armed)
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(
        seen, _selftest(mlps[0], op, armed.rank), group=tp.cpu_group
    )
    if not all(seen):
        logger.warning(
            "DSpark draft: the MLP keeps its TP all-reduce: the fused "
            "all-reduce self-test failed (per rank: %s).",
            seen,
        )
        return False
    for mlp in mlps:
        mlp._reducing_op = op
        mlp.register_buffer("_mesh_rank", armed.rank, persistent=False)
    return True


def arm_fused_mlp(model: torch.nn.Module | None, fold_all_reduce: bool = False) -> int:
    """Arm every eligible draft MLP of `model`; returns how many were armed.
    With fold_all_reduce, collective over the TP group."""
    if model is None:
        return 0
    mlps = [m for m in model.modules() if isinstance(m, MiniMaxM3DSparkMLP)]
    armed = []
    for mlp in mlps:
        operands = e4m3_mlp_operands(mlp)
        if operands is None:
            continue
        for name, tensor in zip(_BUFFERS, operands):
            mlp.register_buffer(name, tensor, persistent=False)
        armed.append(mlp)
    if not mlps:
        return 0
    folded = bool(armed) and fold_all_reduce and _arm_all_reduce(armed)
    # [i, h]: K-major.
    fits = bool(armed) and kernel.supported(
        kernel.MAX_ROWS, *armed[0].down_proj.weight.shape[::-1]
    )
    logger.info(
        "DSpark draft: %d of %d MLPs run as one op%s; its kernel serves up to "
        "%d rows (%s at this shape).",
        len(armed),
        len(mlps),
        " with the TP all-reduce" if folded else "",
        kernel.MAX_ROWS,
        "supported" if fits else "NOT supported",
    )
    return len(armed)
