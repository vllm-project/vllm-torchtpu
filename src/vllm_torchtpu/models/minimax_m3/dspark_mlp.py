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
the same graph). The op also runs the decoder layer's post-attention
residual add + GemmaRMSNorm in front of the MLP (`forward_with_norm`).
With ``fold_all_reduce`` (and the Pallas all-reduce armed,
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
from vllm import ir
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.kernels import dspark_mlp as kernel
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, e4m3_mlp_operands
from vllm_torchtpu.utils import fresh_buffer

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


def _norm_fake(x, residual, *operands):
    return _fake(x), _fake(x)


def _mlp_body(eps: float | None, n: int = 1, reduce=None, pre_ar: bool = False):
    """Per-device body of (x, [residual, norm_w,] *operands[, rank]): the
    kernel at its rows, else XLA (plus `reduce` when n > 1: of the output, or
    with pre_ar of the input x)."""

    def body(x, *args):
        if eps is not None:
            residual, norm_w, *args = args
            norm = (residual, norm_w, eps)
        else:
            norm = None
        operands, rank = (args[:4], args[4]) if n > 1 else (args, None)
        m, h = x.shape
        if kernel.supported(m, h, operands[2].shape[0], n):
            return kernel.fused_mlp(
                x, *operands, rank=rank, n=n, norm=norm, pre_ar=pre_ar
            )
        if pre_ar:
            x = reduce(x, rank)
        out = kernel.reference(x, *operands, norm=norm)
        if norm is not None:
            out, new_residual = out
        if n > 1 and not pre_ar:
            out = reduce(out, rank)
        return (out, new_residual) if norm is not None else out

    return body


# torch_tpu reads a jax_op's operand types from its annotations: one
# signature per operand list.
A = jax.Array


def _norm_signature(body):
    def fn(
        x: A, residual: A, norm_w: A, w_gu: A, s_gu: A, w_d: A, s_d: A
    ) -> tuple[A, A]:
        return body(x, residual, norm_w, w_gu, s_gu, w_d, s_d)

    return fn


def _reducing_signature(body):
    def fn(x: A, w_gu: A, s_gu: A, w_d: A, s_d: A, rank: A) -> A:
        return body(x, w_gu, s_gu, w_d, s_d, rank)

    return fn


def _reducing_norm_signature(body):
    def fn(
        x: A, residual: A, norm_w: A, w_gu: A, s_gu: A, w_d: A, s_d: A, rank: A
    ) -> tuple[A, A]:
        return body(x, residual, norm_w, w_gu, s_gu, w_d, s_d, rank)

    return fn


def _make_norm_op(eps: float):
    """The local MLP op with the residual add + RMSNorm prologue: (x,
    residual, norm_w, *operands) -> (out, new residual)."""
    op = pallas.jax_op("pallas::dspark_fused_mlp_norm", _norm_signature(_mlp_body(eps)))
    op.register_fake(_norm_fake)
    return op


def _make_reducing_op(armed, eps: float | None = None, pre_ar: bool = False):
    """The MLP (with eps: and its norm prologue) plus the TP all-reduce over
    the armed all-reduce's mesh, of its output or (pre_ar, with eps) of its
    input x, as a torch op of (x, [residual, norm_w,] *operands, rank)."""
    mesh = armed.mesh
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    reduce = pallas_all_reduce.reduce_body(n, armed.max_bytes, axis)
    body = _mlp_body(eps, n, reduce, pre_ar)
    nargs = 6 if eps is None else 8
    specs = (P(axis),) * nargs
    out_specs = P(axis) if eps is None else (P(axis), P(axis))
    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=out_specs, check_vma=False
    )

    def put(*args):
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::dspark_fused_mlp_"
        + ("pre_all_reduce" if pre_ar else "all_reduce")
        + ("" if eps is None else "_norm"),
        (_reducing_signature if eps is None else _reducing_norm_signature)(put),
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=out_specs,
    )
    op.register_fake(_fake if eps is None else _norm_fake)
    return op


class MiniMaxM3DSparkMLP(MiniMaxM3MLP):
    """MiniMaxM3MLP that runs as the fused op once `arm_fused_mlp` armed it.

    Armed with its layer's post-attention norm (buffer `_fused_norm_w`, and
    `_norm_op` set), `forward_with_norm` also does that norm's residual add.
    With the all-reduce folded in, `_reducing_op` / `_reducing_norm_op` take
    a buffer `_mesh_rank` (this rank's mesh index) last. With
    `_pre_reducing_norm_op`, `forward_with_norm` takes the attention output
    before its TP all-reduce (o_proj's partial) and reduces it in the kernel,
    under the weight stream.
    """

    _norm_op = None
    _reducing_op = None
    _reducing_norm_op = None
    _pre_reducing_norm_op = None

    def _armed_for(self, x) -> bool:
        return (
            getattr(self, _BUFFERS[0], None) is not None
            and isinstance(x, torch.Tensor)
            and x.dim() == 2
            and x.dtype == torch.bfloat16
        )

    def _operands(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name in _BUFFERS]

    def _reduce(self, out: torch.Tensor) -> torch.Tensor:
        if self.down_proj.reduce_results and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def forward(self, x):
        if not self._armed_for(x):
            return super().forward(x)
        if self._reducing_op is not None:
            return self._reducing_op(x, *self._operands(), self._mesh_rank)
        return self._reduce(_op(x, *self._operands()))

    def forward_with_norm(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(mlp(norm(x + residual)), x + residual)`` for the post-attention
        GemmaRMSNorm this MLP was armed with, or None when not armed so (the
        caller runs the norm and `forward`)."""
        if (
            self._norm_op is None
            or not self._armed_for(x)
            or not isinstance(residual, torch.Tensor)
            or residual.shape != x.shape
            or residual.dtype != x.dtype
        ):
            return None
        args = (x, residual, self._fused_norm_w, *self._operands())
        if self._pre_reducing_norm_op is not None:
            out, residual = self._pre_reducing_norm_op(*args, self._mesh_rank)
            return self._reduce(out), residual
        if self._reducing_norm_op is not None:
            return self._reducing_norm_op(*args, self._mesh_rank)
        out, residual = self._norm_op(*args)
        return self._reduce(out), residual


class CachedGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm that, once `arm_fused_mlp` cached it, reads (1 + w) in f32
    from a buffer instead of computing it on every call: the same values,
    without a convert op per norm per step (1.1 / 1.3 us per q/k and hidden
    norm at 24 rows, c4 twin j-617cb005)."""

    def cache_weight(self) -> None:
        w = fresh_buffer(self.weight.float() + 1.0)
        self.register_buffer("_weight_plus_one", w, persistent=False)

    def forward_native(self, x, residual=None):
        w = getattr(self, "_weight_plus_one", None)
        if w is None:
            return super().forward_native(x, residual)
        if residual is None:
            return ir.ops.rms_norm(x, w, self.variance_epsilon)
        return ir.ops.fused_add_rms_norm(x, residual, w, self.variance_epsilon)


def _norm_weight(norm) -> torch.Tensor | None:
    """GemmaRMSNorm's f32 [1, h] (1 + w), or None for another norm."""
    if not isinstance(norm, GemmaRMSNorm) or norm.weight.dim() != 1:
        return None
    return fresh_buffer((norm.weight.float() + 1.0).reshape(1, -1))


def _selftest(mlp: MiniMaxM3DSparkMLP, ops, rank: torch.Tensor) -> bool:
    """Whether the reducing ops give this rank the bits of the local ops plus
    the armed all-reduce. Collective over the TP group; a wrong rank or mesh
    hangs the kernel, so a timeout dumps every stack and exits."""
    operands = mlp._operands()
    gen = torch.Generator().manual_seed(1 + get_tp_group().rank_in_group)
    hidden = operands[0].shape[0]
    device = operands[0].device

    def rows(n):
        return torch.randn(n, hidden, generator=gen).to(torch.bfloat16).to(device)

    ok = True
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for n in _SELFTEST_ROWS:
            x = rows(n)
            want = tensor_model_parallel_all_reduce(_op(x, *operands))
            got = ops["mlp"](x, *operands, rank)
            ok &= bool(torch.equal(got.cpu(), want.cpu()))
            if "norm" in ops:
                args = (x, rows(n), mlp._fused_norm_w, *operands)
                out, res = mlp._norm_op(*args)
                got, got_res = ops["norm"](*args, rank)
                want = tensor_model_parallel_all_reduce(out)
                ok &= bool(torch.equal(got.cpu(), want.cpu()))
                ok &= bool(torch.equal(got_res.cpu(), res.cpu()))
    finally:
        faulthandler.cancel_dump_traceback_later()
    return ok


def _arm_all_reduce(mlps: list[MiniMaxM3DSparkMLP], eps: float | None) -> bool:
    """Fold the TP all-reduce into every armed MLP's ops; collective."""
    tp = get_tp_group()
    armed = pallas_all_reduce.armed_for(tp.ranks)
    if armed is None or armed.mesh is None or tp.world_size == 1:
        logger.warning(
            "DSpark draft: the MLP keeps its TP all-reduce: the Pallas "
            "all-reduce is not armed for the TP group."
        )
        return False
    if not all(mlp.down_proj.reduce_results for mlp in mlps):
        logger.warning(
            "DSpark draft: the MLP's all-reduce is not folded: a down_proj "
            "does not reduce its results."
        )
        return False
    ops = {"mlp": _make_reducing_op(armed)}
    if eps is not None:
        ops["norm"] = _make_reducing_op(armed, eps)
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(
        seen, _selftest(mlps[0], ops, armed.rank), group=tp.cpu_group
    )
    if not all(seen):
        logger.warning(
            "DSpark draft: the MLP keeps its TP all-reduce: the fused "
            "all-reduce self-test failed (per rank: %s).",
            seen,
        )
        return False
    for mlp in mlps:
        mlp._reducing_op = ops["mlp"]
        mlp._reducing_norm_op = ops.get("norm")
        mlp.register_buffer("_mesh_rank", armed.rank, persistent=False)
    return True


def _arm_norm(layers: list, mlps: list[MiniMaxM3DSparkMLP]) -> float | None:
    """Hand each armed MLP its layer's post-attention norm; returns the
    shared eps, or None (no MLP takes its norm) when any layer's norm is not
    a GemmaRMSNorm or the eps differ."""
    norms = [layer.post_attention_layernorm for layer in layers]
    weights = [_norm_weight(norm) for norm in norms]
    eps = {getattr(norm, "variance_epsilon", None) for norm in norms}
    if any(w is None for w in weights) or len(eps) != 1:
        return None
    (eps,) = eps
    op = _make_norm_op(eps)
    for mlp, weight in zip(mlps, weights):
        mlp.register_buffer("_fused_norm_w", weight, persistent=False)
        mlp._norm_op = op
    return eps


def _pre_selftest(mlp: MiniMaxM3DSparkMLP, op, rank: torch.Tensor) -> bool:
    """Whether the pre-reducing op gives this rank the bits of the armed
    all-reduce followed by the local norm op. Collective, with a timeout."""
    operands = mlp._operands()
    hidden = operands[0].shape[0]
    device = operands[0].device
    mine = torch.Generator().manual_seed(1 + get_tp_group().rank_in_group)
    shared = torch.Generator().manual_seed(0)  # the residual is replicated
    ok = True
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for n in _SELFTEST_ROWS:
            x = torch.randn(n, hidden, generator=mine).to(torch.bfloat16).to(device)
            res = torch.randn(n, hidden, generator=shared).to(torch.bfloat16)
            args = (res.to(device), mlp._fused_norm_w, *operands)
            want, want_res = mlp._norm_op(tensor_model_parallel_all_reduce(x), *args)
            got, got_res = op(x, *args, rank)
            ok &= bool(torch.equal(got.cpu(), want.cpu()))
            ok &= bool(torch.equal(got_res.cpu(), want_res.cpu()))
    finally:
        faulthandler.cancel_dump_traceback_later()
    return ok


def _arm_pre_all_reduce(layers: list, eps: float | None) -> bool:
    """Hand each layer's attention-exit TP all-reduce to its MLP's op (o_proj
    stops reducing; the op reduces under the weight stream); collective."""
    tp = get_tp_group()
    armed = pallas_all_reduce.armed_for(tp.ranks)
    o_projs = [layer.self_attn.o_proj for layer in layers]
    refusal = None
    if eps is None:
        refusal = "the MLPs do not carry their post-attention norm"
    elif armed is None or armed.mesh is None or tp.world_size == 1:
        refusal = "the Pallas all-reduce is not armed for the TP group"
    elif not all(getattr(o, "reduce_results", False) for o in o_projs):
        refusal = "an o_proj does not reduce its results"
    if refusal:
        logger.warning("DSpark draft: the attention all-reduce stays: %s.", refusal)
        return False
    op = _make_reducing_op(armed, eps, pre_ar=True)
    mlps = [layer.mlp for layer in layers]
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(
        seen, _pre_selftest(mlps[0], op, armed.rank), group=tp.cpu_group
    )
    if not all(seen):
        logger.warning(
            "DSpark draft: the attention all-reduce stays: self-test failed %s.",
            seen,
        )
        return False
    for mlp, o_proj in zip(mlps, o_projs):
        mlp._pre_reducing_norm_op = op
        mlp.register_buffer("_mesh_rank", armed.rank, persistent=False)
        o_proj.reduce_results = False
    return True


def arm_fused_mlp(
    model: torch.nn.Module | None,
    fold_all_reduce: bool = False,
    pre_all_reduce: bool = False,
) -> int:
    """Arm every eligible draft MLP of `model` (with its decoder layer's
    post-attention norm when it is a GemmaRMSNorm); returns how many were
    armed. With fold_all_reduce (the MLP's own all-reduce) or pre_all_reduce
    (the attention-exit one), collective over the TP group."""
    if model is None:
        return 0
    mlps = [m for m in model.modules() if isinstance(m, MiniMaxM3DSparkMLP)]
    layers = [
        m
        for m in model.modules()
        if isinstance(getattr(m, "mlp", None), MiniMaxM3DSparkMLP)
        and hasattr(m, "post_attention_layernorm")
    ]
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
    cached = [m for m in model.modules() if isinstance(m, CachedGemmaRMSNorm)]
    for norm in cached:
        norm.cache_weight()
    eps = None
    armed_layers = [layer for layer in layers if layer.mlp in armed]
    if armed and len(armed_layers) == len(armed):
        eps = _arm_norm(armed_layers, [layer.mlp for layer in armed_layers])
    folded = bool(armed) and fold_all_reduce and _arm_all_reduce(armed, eps)
    pre = bool(armed) and pre_all_reduce and _arm_pre_all_reduce(armed_layers, eps)
    # [i, h]: K-major.
    fits = bool(armed) and kernel.supported(
        kernel.MAX_ROWS,
        *armed[0].down_proj.weight.shape[::-1],
        n=get_tp_group().world_size if folded or pre else 1,
    )
    logger.info(
        "DSpark draft: %d of %d MLPs run as one op%s%s%s; its kernel serves up to "
        "%d rows (%s at this shape); %d norms read a cached (1 + w).",
        len(armed),
        len(mlps),
        " with the post-attention norm" if eps is not None else "",
        " with the TP all-reduce" if folded else "",
        " with the attention all-reduce in front" if pre else "",
        kernel.MAX_ROWS,
        "supported" if fits else "NOT supported",
        len(cached),
    )
    return len(armed)
