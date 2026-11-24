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
"""MiniMax-M3 decoder layers with their TP all-reduces fused with the ops
next to them (TPU_MINIMAX_M3_FUSED_RESIDUAL_AR, kernels/collectives/residual_ar.py).

`arm` (once after load, after the Pallas all-reduce armed) hands every
decoder layer two torch ops over the armed all-reduce's mesh:

  op_a(x, residual, norm_w, w, s, rank) -> (qkv, r): the previous layer's
      FFN partial x all-reduced with the input layernorm's residual add +
      norm in one kernel, then qkv_proj: in the same kernel with
      TPU_MINIMAX_M3_FUSED_RESIDUAL_AR_QKV (its weight streams in under the
      all-reduce), else as an XLA op;
  op_b(a, w, s, residual, norm_w, rank) -> (normed, r): o_proj on the
      attention heads a, all-reduced, the post-attention residual add + norm,
      one kernel.

Each decides on its static row count: the kernels up to the armed row cap,
else the same math in XLA around the armed all-reduce op's reduction, so
every bucket traces the same graph. Armed layers stop reducing o_proj and
the FFN (the model owns those reductions, as in a sequence-parallel trace);
MiniMaxM3Model reduces the last layer's FFN before the final norm.
"""

import faulthandler

import jax
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_torchtpu import envs
from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.kernels.collectives import residual_ar as kernel
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3 import sp_o_proj

logger = init_logger(__name__)

SELFTEST_TIMEOUT_S = 300
# Bar for a self-test whose reference sums differently (psum past the armed
# all-reduce's byte cap, or the matmul's K order): roundings, while a wrong
# rank or slot misses by O(1).
_SELFTEST_REL_TOL = 2**-6
# ... and for an output an e4m3-rows projection fed (e4m3's 3 mantissa bits).
_SELFTEST_E4M3_REL_TOL = 2**-4

A = jax.Array


def _op_bodies(n: int, max_rows: int, reduce, eps: float, e4m3: bool, qkv: bool):
    """op_a / op_b's per-device bodies: the kernels up to max_rows, else the
    unfused math around `reduce` (the armed all-reduce op's). With `e4m3`,
    qkv_proj (and o_proj above the kernels' rows) contract e4m3 rows with
    per-row scales against the e4m3 weight (e4m3 x e4m3). With `qkv`, op_a's
    kernel also runs qkv_proj."""

    def proj(v, w, s):
        if e4m3:
            return kernel.e4m3_rows_matmul(*kernel.quantize_rows(v), w, s)
        return kernel.reference_matmul(v, w, s)

    def body_a(x, residual, norm_w, w, s, rank):
        m, h = x.shape
        if not kernel.supported(m, h, n, max_rows):
            normed, r = kernel.reference_add_norm(
                reduce(x, rank), residual, norm_w, eps
            )
            return proj(normed, w, s), r
        if qkv and kernel.qkv_supported(m, h, n, max_rows, w.shape):
            return kernel.ar_add_norm_matmul(
                x, residual, norm_w, w, s, rank, n, eps, quant=e4m3
            )
        if e4m3:
            q, scale, r = kernel.ar_add_norm(
                x, residual, norm_w, rank, n, eps, quant=True
            )
            return kernel.e4m3_rows_matmul(q, scale, w, s), r
        normed, r = kernel.ar_add_norm(x, residual, norm_w, rank, n, eps)
        return kernel.reference_matmul(normed, w, s), r

    def body_b(a, w, s, residual, norm_w, rank):
        m, h = residual.shape
        if kernel.oproj_supported(m, h, n, max_rows, w.shape):
            return kernel.matmul_ar_norm(a, w, s, residual, norm_w, rank, n, eps)
        return kernel.reference_add_norm(
            reduce(proj(a, w, s), rank), residual, norm_w, eps
        )

    return body_a, body_b


def _make_ops(armed, max_rows: int, eps: float):
    """(op_a, op_b) as torch ops over the armed all-reduce's mesh."""
    mesh = armed.mesh
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    reduce = pallas_all_reduce.reduce_body(n, armed.max_bytes, axis)
    body_a, body_b = _op_bodies(
        n,
        max_rows,
        reduce,
        eps,
        envs.TPU_MINIMAX_M3_FUSED_RESIDUAL_AR_E4M3,
        envs.TPU_MINIMAX_M3_FUSED_RESIDUAL_AR_QKV,
    )
    specs = (P(axis),) * 6
    out_specs = (P(axis), P(axis))

    def sharded(body):
        f = jax.shard_map(
            body, mesh=mesh, in_specs=specs, out_specs=out_specs, check_vma=False
        )

        def put(*args):
            return f(
                *(
                    jax.device_put(a, NamedSharding(mesh, sp))
                    for a, sp in zip(args, specs)
                )
            )

        return put

    f_a, f_b = sharded(body_a), sharded(body_b)

    # torch_tpu reads a jax_op's operand types from its annotations.
    def fn_a(x: A, residual: A, norm_w: A, w: A, s: A, rank: A) -> tuple[A, A]:
        return f_a(x, residual, norm_w, w, s, rank)

    def fn_b(a: A, w: A, s: A, residual: A, norm_w: A, rank: A) -> tuple[A, A]:
        return f_b(a, w, s, residual, norm_w, rank)

    def fake_a(x, residual, norm_w, w, s, rank):
        return x.new_empty((x.shape[0], w.shape[1])), torch.empty_like(residual)

    def fake_b(a, w, s, residual, norm_w, rank):
        return torch.empty_like(residual), torch.empty_like(residual)

    ops = []
    for name, fn, fake in (("a", fn_a, fake_a), ("b", fn_b, fake_b)):
        # pyrefly: ignore[no-matching-overload]
        op = pallas.jax_op(
            f"pallas::minimax_m3_residual_ar_{name}",
            fn,
            mesh=mesh,
            input_partition_specs=specs,
            output_partition_specs=out_specs,
        )
        op.register_fake(fake)
        ops.append(op)
    return tuple(ops)


def _gemma_weight(norm) -> torch.Tensor | None:
    """A GemmaRMSNorm's stored 1-D weight w (it scales by 1 + w), else None."""
    if not isinstance(norm, GemmaRMSNorm) or norm.weight.dim() != 1:
        return None
    return norm.weight


def _e4m3_operands(linear, weight) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(e4m3 [K, N] weight, [N] scale) of a bias-free MXFP8 linear kept e4m3
    with a per-channel scale (MiniMaxM3Mxfp8LinearMethod), else None."""
    scale = getattr(linear, "weight_scale", None)
    if (
        getattr(linear, "bias", None) is not None
        or weight.dtype != torch.float8_e4m3fn
        or weight.dim() != 2
        or scale is None
        or scale.shape != (weight.shape[1],)
    ):
        return None
    return weight, scale


def _ffn_reduces_itself(layer) -> bool:
    """Whether a decode MoE kernel armed on the layer all-reduces its output
    whatever skip_final_all_reduce says (moe_tp_decode, fused_moe_decode):
    the layer's own reduction would then sum it twice."""
    from vllm_torchtpu.layers.adapter.moe_tp_decode import moe_tp_decode_supported

    if not layer.is_moe_layer:
        return False
    for module in layer.block_sparse_moe.modules():
        if getattr(module, "_fused_moe_decode_armed", False):
            return True
        for attr in ("quant_method", "_quant_method"):
            method = getattr(module, attr, None)
            if method is not None and moe_tp_decode_supported(method):
                return True
    return False


def _ffn_reduction_owner(layer):
    """(object, attribute, value that skips the FFN's all-reduce)."""
    if layer.is_moe_layer:
        return layer.block_sparse_moe.experts.moe_config, "skip_final_all_reduce", True
    return layer.mlp.down_proj, "reduce_results", False


def _check(name, rows, got, want, exact, tol=_SELFTEST_REL_TOL):
    got, want = got.float().cpu(), want.float().cpu()
    finite = bool(torch.isfinite(got).all())
    diff = (got - want).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    scale = float(want.abs().max()) or 1.0
    passed = finite and (
        bool(torch.equal(got, want)) if exact else max_abs <= tol * scale
    )
    return passed, (
        f"{name}@{rows}: {'ok' if passed else 'FAIL'} max_abs={max_abs:.3g} "
        f"scale={scale:.3g} finite={finite}"
    )


def _pallas_sums(rows: int, hidden: int) -> bool:
    """Whether the armed all-reduce sums [rows, hidden] with its Pallas kernel
    (rank-ordered f32, as the fused kernels do) rather than psum."""
    from vllm_torchtpu.kernels.collectives import tp_all_reduce

    armed = pallas_all_reduce.armed_for(get_tp_group().ranks)
    n = get_tp_group().world_size
    return 0 < tp_all_reduce.view_bytes(rows, hidden, n) <= armed.max_bytes


def residual_ar_operands(layer) -> tuple[torch.Tensor, ...]:
    """(qkv w, qkv s, input norm w, o_proj w, o_proj s, post-attention norm w)
    of an armed layer, as op_a / op_b take them: the layer's own parameters,
    so every trace of the model reads the same inputs (a sequence-parallel
    trace and this one hand buckets between them, compilation/shape_variants.py)."""
    attn = layer.self_attn
    return (
        attn.qkv_proj.weight,
        attn.qkv_proj.weight_scale,
        layer.input_layernorm.weight,
        sp_o_proj.local_weight(attn.o_proj),
        attn.o_proj.weight_scale,
        layer.post_attention_layernorm.weight,
    )


def _selftest(layer, ops, rank, max_rows) -> bool:
    """Whether the ops give this rank the armed all-reduce plus the unfused
    math, at the kernel's rows and one count above the cap. Collective over
    the TP group; a wrong rank or mesh hangs, so a timeout dumps and exits."""
    op_a, op_b = ops
    w_a, s_a, nw_a, w_b, s_b, nw_b = residual_ar_operands(layer)
    h = w_a.shape[0]
    device = w_a.device
    mine = torch.Generator().manual_seed(1 + get_tp_group().rank_in_group)
    shared = torch.Generator().manual_seed(0)

    def rnd(gen, *shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16).to(device)

    def proj(v, w, s):
        return (torch.matmul(v, w.to(torch.bfloat16)) * s.to(torch.bfloat16)).to(
            torch.bfloat16
        )

    def add_norm(x, residual, nw):
        r = (x.float() + residual.float()).to(torch.bfloat16)
        rf = r.float()
        var = rf.pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(var + layer._residual_ar_eps)
        return (rf * scale * (nw.float() + 1.0)).to(torch.bfloat16), r

    tol = (
        _SELFTEST_E4M3_REL_TOL
        if envs.TPU_MINIMAX_M3_FUSED_RESIDUAL_AR_E4M3
        else _SELFTEST_REL_TOL
    )
    results = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        # The kernels' rows, the cap, and two counts above it (one not a
        # multiple of the 16-row tile).
        for rows in sorted({8, 24, max_rows, max_rows + 8, max_rows + 16}):
            x, res = rnd(mine, rows, h), rnd(shared, rows, h)
            got_y, got_r = op_a(x, res, nw_a, w_a, s_a, rank)
            normed, want_r = add_norm(tensor_model_parallel_all_reduce(x), res, nw_a)
            exact = rows > max_rows or _pallas_sums(rows, h)
            results.append(_check("a.r", rows, got_r, want_r, exact))
            results.append(
                _check("a.y", rows, got_y, proj(normed, w_a, s_a), False, tol)
            )
            a = rnd(mine, rows, w_b.shape[0])
            got_o, got_r = op_b(a, w_b, s_b, res, nw_b, rank)
            want_o, want_r = add_norm(
                tensor_model_parallel_all_reduce(proj(a, w_b, s_b)), res, nw_b
            )
            results.append(_check("b.r", rows, got_r, want_r, False, tol))
            results.append(_check("b.out", rows, got_o, want_o, False, tol))
    finally:
        faulthandler.cancel_dump_traceback_later()
    ok = all(p for p, _ in results)
    (logger.info if ok else logger.warning)(
        "MiniMax-M3 fused residual all-reduce self-test: %s",
        "; ".join(r for _, r in results),
    )
    return ok


def arm(model: torch.nn.Module, max_rows: int) -> int:
    """Arm every MiniMax-M3 decoder layer of `model` (all or none); returns how
    many. Collective over the TP group."""
    from vllm_torchtpu.models.minimax_m3.model import (
        MiniMaxM3DecoderLayer,
        MiniMaxM3Model,
    )

    tp = get_tp_group()
    models = [m for m in model.modules() if isinstance(m, MiniMaxM3Model)]
    armed = pallas_all_reduce.armed_for(tp.ranks)
    refusal = None
    if len(models) != 1:
        refusal = f"{len(models)} MiniMaxM3Model modules"
    elif armed is None or armed.mesh is None or tp.world_size == 1:
        refusal = "the Pallas all-reduce is not armed for the TP group"
    layers = (
        []
        if refusal
        else [m for m in models[0].layers if isinstance(m, MiniMaxM3DecoderLayer)]
    )
    operands, eps = [], set()
    for layer in layers:
        attn = layer.self_attn
        a = _e4m3_operands(attn.qkv_proj, attn.qkv_proj.weight)
        b = _e4m3_operands(attn.o_proj, sp_o_proj.local_weight(attn.o_proj))
        nw_a = _gemma_weight(layer.input_layernorm)
        nw_b = _gemma_weight(layer.post_attention_layernorm)
        if None in (a, b, nw_a, nw_b) or not attn.o_proj.reduce_results:
            refusal = f"layer {len(operands)} is not e4m3 / GemmaRMSNorm / reducing"
            break
        if _ffn_reduces_itself(layer):
            refusal = f"layer {len(operands)}'s MoE decode kernel reduces its output"
            break
        eps |= {
            layer.input_layernorm.variance_epsilon,
            layer.post_attention_layernorm.variance_epsilon,
        }
        operands.append((*a, nw_a, *b, nw_b))
    if not refusal and len(eps) != 1:
        refusal = f"norm eps differ: {eps}"
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, refusal, group=tp.cpu_group)
    if any(seen) or not layers:
        logger.warning("MiniMax-M3 fused residual all-reduce not armed: %s", seen)
        return 0
    (eps,) = eps
    ops = _make_ops(armed, max_rows, eps)
    for layer in layers:
        layer._residual_ar_eps = eps
    ok = _selftest(layers[0], ops, armed.rank, max_rows)
    torch.distributed.all_gather_object(seen, ok, group=tp.cpu_group)
    if not all(seen):
        logger.warning(
            "MiniMax-M3 fused residual all-reduce: self-test failed %s", seen
        )
        return 0
    for layer in layers:
        layer._residual_ar_ops = ops
        layer.register_buffer("_mesh_rank", armed.rank, persistent=False)
        layer.self_attn.o_proj.reduce_results = False
        owner, attr, value = _ffn_reduction_owner(layer)
        setattr(owner, attr, value)
    models[0].residual_ar_armed = True
    logger.info(
        "MiniMax-M3: %d decoder layers fuse their TP all-reduces with o_proj "
        "and the residual add + norm up to %d rows%s%s.",
        len(layers),
        max_rows,
        "; qkv_proj runs in the FFN-exit kernel"
        if envs.TPU_MINIMAX_M3_FUSED_RESIDUAL_AR_QKV
        else "",
        "; qkv_proj contracts e4m3 rows"
        if envs.TPU_MINIMAX_M3_FUSED_RESIDUAL_AR_E4M3
        else "",
    )
    return len(layers)
