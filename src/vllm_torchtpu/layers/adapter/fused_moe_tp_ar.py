# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Serve a TP-only MoE layer, its shared expert and its TP all-reduce in one kernel.

Under MoE tensor parallelism vLLM computes the shared expert separately, adds
it to the routed output and all-reduces the sum. With
``USE_MOE_FUSED_TP_AR=1`` an eligible layer instead runs
``moe_fused_tp.fused_moe_tp_ar`` under shard_map over the TP mesh: shared +
routed experts and the all-reduce in one Pallas call per rank.

Arming happens once after the model is loaded (``arm_fused_tp_ar``), when the
shared expert's linears have their final FP8 per-channel layout. For an armed
layer the runner is told it has no shared expert (the kernel owns it), and
``MoERunner._fused_output_is_reduced`` reports True (see ``VllmMoERunner`` in
``layers/adapter/custom_ops/moe_runner.py``), so vLLM issues neither the shared
MLP nor any all-reduce. Token counts above ``MAX_DECODE_TOKENS`` run the fused
TP prefill kernel (``moe_tp_prefill``) inside the same exported program (the
token count is static there), still returning the reduced sum; branching in
torch instead would give buckets different graphs.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, build_ep_mesh, ep_mesh_index
from vllm_torchtpu.kernels.experimental.moe_fused_tp import (
    fused_moe_tp_ar as _kernel_body,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FUSED_MOE_TP_AR_OP_ATTR = "_fused_moe_tp_ar_op"
FUSED_MOE_TP_SP_OP_ATTR = "_fused_moe_tp_sp_op"
_RUNNER_ARMED_ATTR = "_fused_moe_tp_ar_armed"
# The shared expert's tensors become non-persistent buffers of the routed layer,
# so they stay module state Dynamo traces like any weight (a tensor reached
# through a plain attribute makes traces of the same graph disagree on inputs).
_SHARED_BUFFERS = (
    "_ftp_ar_w1s",
    "_ftp_ar_s1s",
    "_ftp_ar_w2s",
    "_ftp_ar_s2s",
    "_ftp_ar_wg",
)

# One exported program per distinct closure; the mesh never enters a traced
# frame (ops are built at arming time and only looked up in forward).
_OPS: dict[tuple, Any] = {}
_SP_OPS: dict[tuple, Any] = {}
_RANK_BUFFER: Any = None
_CURRENT_LAYER_USE_SP: bool = False
# Largest token count served by the decode kernel; above it, the prefill kernel.
MAX_DECODE_TOKENS = 128


def fused_moe_tp_ar_supported(quant_method: Any) -> bool:
    return getattr(quant_method, FUSED_MOE_TP_AR_OP_ATTR, None) is not None


def _build_op(mesh, n_ranks: int, topk: int):
    # The token count is static inside the exported program, so the cap is
    # applied HERE, not in torch: every bucket then traces the same torch op
    # with the same inputs, which vLLM's shape-specialized warmup requires.
    key = (id(mesh), n_ranks, topk)
    op = _OPS.get(key)
    if op is not None:
        return op
    axis = EP_AXIS_NAME
    spec = PartitionSpec(axis)

    def prefill(x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti):
        """Above MAX_DECODE_TOKENS: expert kernel + combine + chunked psum."""
        from vllm_torchtpu.kernels.experimental.moe_tp_prefill import (
            default_combine_chunks,
            moe_tp_prefill,
        )

        # Trace-time, once per token bucket: the evidence that the prefill
        # kernel, not the GMM branch, serves this shape.
        logger.info("Fused TP MoE prefill kernel traced for %d tokens", x.shape[0])
        return moe_tp_prefill(
            x,
            w1,
            w2,
            s1,
            s2,
            w1s,
            w2s,
            s1s,
            s2s,
            wg.astype(x.dtype),
            tw,
            ti,
            combine_chunks=default_combine_chunks(x.shape[0]),
            reduce_fn=lambda y: jax.lax.psum(y, axis),
        )

    def local(rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti):
        s1s, s2s, wg = s1s.reshape(1, -1), s2s.reshape(1, -1), wg.reshape(-1, 1)
        if x.shape[0] > MAX_DECODE_TOKENS:
            return prefill(x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti)
        return _kernel_body(
            rank.reshape(-1),
            x,
            w1,
            w2,
            s1,
            s2,
            w1s,
            s1s,
            w2s,
            s2s,
            wg,
            tw,
            ti,
            n_ranks=n_ranks,
        )

    # torch_tpu infers static arguments from annotations: every operand is a
    # jax.Array. Per-rank operands are sharded on axis 0; the gate is replicated.
    def fn(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        s1: jax.Array,
        s2: jax.Array,
        w1s: jax.Array,
        s1s: jax.Array,
        w2s: jax.Array,
        s2s: jax.Array,
        wg: jax.Array,
        tw: jax.Array,
        ti: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        body = jax.shard_map(
            lambda *a: local(a[-1], *a[:-1]),
            mesh=mesh,
            in_specs=(spec,) * 9 + (PartitionSpec(), spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )
        return body(
            x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti.astype(jnp.int32), rank
        )

    name = "pallas::fused_moe_tp_ar" + (f"_{len(_OPS)}" if _OPS else "")
    op = jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 9 + (PartitionSpec(), spec, spec, spec),
        output_partition_specs=spec,
    )

    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def _build_sp_op(mesh, n_ranks: int, topk: int):
    """Sequence-parallel fused TP MoE op (`[T/TP, H] -> [T/TP, H]`).

    On prefill token buckets (`T > MAX_DECODE_TOKENS`), each rank quantizes its
    local `[T/TP, H]` slice to FP8 and computes the shared-expert gate on
    `[T/TP, H]`, then all-gathers the 1-byte FP8 rows and compact metadata
    (`x_s`, `gate`, `topk_weights`, `topk_ids`) before running `moe_tp_prefill`
    with chunked `psum_scatter` directly back to `[T/TP, H]`.
    """
    key = (id(mesh), n_ranks, topk)
    op = _SP_OPS.get(key)
    if op is not None:
        return op
    axis = EP_AXIS_NAME
    spec = PartitionSpec(axis)

    def local_sp(rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti):
        from vllm_torchtpu.kernels.experimental.moe_tp_prefill import (
            default_combine_chunks,
            moe_tp_prefill,
            rowquant_rows,
        )

        s1s, s2s, wg = s1s.reshape(1, -1), s2s.reshape(1, -1), wg.reshape(-1, 1)
        t_local = x.shape[0]
        if t_local * n_ranks <= MAX_DECODE_TOKENS:
            x_full = jax.lax.all_gather(x, axis, axis=0, tiled=True)
            tw_full = jax.lax.all_gather(tw, axis, axis=0, tiled=True)
            ti_full = jax.lax.all_gather(ti.astype(jnp.int32), axis, axis=0, tiled=True)
            out_full = _kernel_body(
                rank.reshape(-1),
                x_full,
                w1,
                w2,
                s1,
                s2,
                w1s,
                s1s,
                w2s,
                s2s,
                wg,
                tw_full,
                ti_full,
                n_ranks=n_ranks,
            )
            return jax.lax.dynamic_slice_in_dim(
                out_full, rank.reshape(()) * t_local, t_local, axis=0
            )

        x_q_local, x_s_local = rowquant_rows(x)
        gate_local = jax.nn.sigmoid(
            jnp.dot(x, wg.astype(x.dtype), preferred_element_type=jnp.float32)
        )
        k = tw.shape[1]
        meta_local = jnp.concatenate(
            [
                x_s_local[:, None],
                gate_local,
                tw.astype(jnp.float32),
                jax.lax.bitcast_convert_type(ti.astype(jnp.int32), jnp.float32),
            ],
            axis=1,
        )
        x_q = jax.lax.all_gather(x_q_local, axis, axis=0, tiled=True)
        meta = jax.lax.all_gather(meta_local, axis, axis=0, tiled=True)
        x_s = meta[:, 0]
        gate = meta[:, 1:2]
        tw_full = meta[:, 2 : 2 + k]
        ti_full = jax.lax.bitcast_convert_type(meta[:, 2 + k :], jnp.int32)
        n_tok = x_q.shape[0]
        logger.info("Fused TP+SP MoE prefill kernel traced for %d tokens", n_tok)
        return moe_tp_prefill(
            x_q,
            w1,
            w2,
            s1,
            s2,
            w1s,
            w2s,
            s1s,
            s2s,
            wg.astype(x.dtype),
            tw_full,
            ti_full,
            combine_chunks=default_combine_chunks(n_tok),
            reduce_fn=lambda y: jax.lax.psum_scatter(
                y, axis, scatter_dimension=0, tiled=True
            ),
            x_quantized=(x_q, x_s),
            shared_gate=gate,
            scatter_ranks=n_ranks,
        )

    def fn(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        s1: jax.Array,
        s2: jax.Array,
        w1s: jax.Array,
        s1s: jax.Array,
        w2s: jax.Array,
        s2s: jax.Array,
        wg: jax.Array,
        tw: jax.Array,
        ti: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        body = jax.shard_map(
            lambda *a: local_sp(a[-1], *a[:-1]),
            mesh=mesh,
            in_specs=(spec,) * 9 + (PartitionSpec(), spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )
        return body(
            x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti.astype(jnp.int32), rank
        )

    name = "pallas::fused_moe_tp_sp" + (f"_{len(_SP_OPS)}" if _SP_OPS else "")
    op = jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 9 + (PartitionSpec(), spec, spec, spec),
        output_partition_specs=spec,
    )

    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _SP_OPS[key] = op
    return op


def _shared_slice(runner) -> tuple | None:
    """This rank's shared-expert tensors in the routed-slice layout, or None."""
    shared = getattr(runner, "_shared_experts", None)
    mlp = getattr(shared, "_layer", None)
    if mlp is None:
        return None
    gate_up = getattr(mlp, "gate_up_proj", None)
    down = getattr(mlp, "down_proj", None)
    gate = getattr(mlp, "expert_gate", None)
    if gate_up is None or down is None or gate is None:
        return None
    w1s, s1s = getattr(gate_up, "weight", None), getattr(gate_up, "weight_scale", None)
    w2s, s2s = getattr(down, "weight", None), getattr(down, "weight_scale", None)
    wg = getattr(gate, "weight", None)
    if w1s is None or s1s is None or w2s is None or s2s is None or wg is None:
        return None
    # Per-channel FP8 linears are stored K-major ([in, out]) with a 1-D scale.
    if s1s.dim() != 1 or s2s.dim() != 1 or w1s.dtype != torch.float8_e4m3fn:
        return None
    if getattr(gate, "bias", None) is not None or wg.dim() != 2 or wg.shape[0] != 1:
        return None
    # Hand over the parameters themselves, never torch views of them: a view
    # passed to the op is re-materialized by an eager as_strided program on
    # every call (measured: one per layer per step, ~48 us of host dispatch
    # each). The kernel's [1, N] / [H, 1] shapes are made inside the program.
    return (w1s.data, s1s.data, w2s.data, s2s.data, wg.data)


def _admit(runner, n_ranks: int) -> tuple | None:
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    cfg = runner.moe_config
    par = cfg.moe_parallel_config
    qm = runner._quant_method
    layer = runner.routed_experts
    if not isinstance(qm, VllmFp8MoEMethodTPU) or par.use_ep or par.tp_size != n_ranks:
        return None
    if getattr(cfg, "skip_final_all_reduce", False):
        return None
    if (
        runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
    ):
        return None
    if (
        getattr(runner, "routed_scaling_factor", 1.0) != 1.0
        or runner.shared_expert_gate is not None
    ):
        return None
    shared = _shared_slice(runner)
    if shared is None:
        return None
    w1s, s1s, w2s, s2s, wg = shared
    s1s, s2s, wg = (
        s1s.reshape(1, -1),
        s2s.reshape(1, -1),
        wg.reshape(-1, 1),
    )  # shapes only
    w1, w2 = layer.w13_weight, layer.w2_weight
    s1, s2 = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    e, h, two_i = w1.shape
    i = two_i // 2
    if (
        w1.dtype != torch.float8_e4m3fn
        or tuple(w2.shape) != (e, i, h)
        or tuple(s1.shape) != (e, 1, 1, two_i)
        or tuple(s2.shape) != (e, 1, 1, h)
        or i % 128
        or h % (n_ranks * 128)
        or tuple(w1s.shape) != (h, two_i)
        or tuple(w2s.shape) != (i, h)
        or tuple(wg.shape) != (h, 1)
    ):
        return None
    return shared


def arm_fused_tp_ar(model: torch.nn.Module) -> int:
    """Arm every eligible MoE layer of `model`; returns how many were armed."""
    if not envs.USE_MOE_FUSED_TP_AR:
        return 0
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runners = [m for m in model.modules() if isinstance(m, MoERunner)]
    if not runners:
        return 0
    try:
        mesh = build_ep_mesh()
    except Exception as exc:  # refuse rather than serve a wrong reduction
        logger.warning("Fused TP all-reduce MoE not armed: no TP mesh (%s)", exc)
        return 0
    if mesh is None:
        return 0
    n_ranks = mesh.shape[EP_AXIS_NAME]
    global _RANK_BUFFER
    armed = 0
    for runner in runners:
        shared = _admit(runner, n_ranks)
        if shared is None:
            continue
        layer = runner.routed_experts
        if _RANK_BUFFER is None:
            _RANK_BUFFER = torch.tensor(
                [[ep_mesh_index()]], dtype=torch.int32, device=layer.w13_weight.device
            )
        setattr(
            runner._quant_method,
            FUSED_MOE_TP_AR_OP_ATTR,
            _build_op(mesh, n_ranks, layer.moe_config.experts_per_token),
        )
        if getattr(runner.moe_config, "is_sequence_parallel", False):
            setattr(
                runner._quant_method,
                FUSED_MOE_TP_SP_OP_ATTR,
                _build_sp_op(mesh, n_ranks, layer.moe_config.experts_per_token),
            )
        for name, tensor in zip(_SHARED_BUFFERS, shared):
            layer.register_buffer(name, tensor, persistent=False)
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        object.__setattr__(runner, _RUNNER_ARMED_ATTR, True)
        armed += 1
    if armed:
        logger.info(
            "Armed fused TP MoE + shared expert + all-reduce on %d of %d MoE layers "
            "(tp=%d, decode kernel up to %d tokens, prefill kernel above)",
            armed,
            len(runners),
            n_ranks,
            MAX_DECODE_TOKENS,
        )
    return armed


def fused_moe_tp_ar(
    quant_method: Any, layer: Any, x: torch.Tensor, topk_weights, topk_ids
):
    """Forward for an armed layer: the TP-reduced (or SP-scattered) output.

    When `_CURRENT_LAYER_USE_SP` is True on a sequence-parallel layer, runs the
    SP prefill program (`[T/TP, H] -> [T/TP, H]`); otherwise runs the replicated
    program (`[T, H] -> [T, H]`).
    """
    w1s, s1s, w2s, s2s, wg = (getattr(layer, name) for name in _SHARED_BUFFERS)
    use_sp = _CURRENT_LAYER_USE_SP and getattr(
        layer.moe_config, "is_sequence_parallel", False
    )
    op_attr = FUSED_MOE_TP_SP_OP_ATTR if use_sp else FUSED_MOE_TP_AR_OP_ATTR
    op = getattr(quant_method, op_attr, None) or getattr(
        quant_method, FUSED_MOE_TP_AR_OP_ATTR
    )
    return op(
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        w1s,
        s1s,
        w2s,
        s2s,
        wg,
        topk_weights,
        topk_ids,
        _RANK_BUFFER,
    )


def arm_runner_models(model_runner: Any) -> int:
    """Arm the target model and, when it is a separate module, the drafter's
    model (the MTP layer). Called once after the drafter has loaded, when every
    module has its final weight layout."""
    armed = arm_fused_tp_ar(model_runner.model)
    drafter = getattr(model_runner, "drafter", None)
    draft = getattr(drafter, "draft_model", None) or getattr(drafter, "model", None)
    if isinstance(draft, torch.nn.Module) and draft is not model_runner.model:
        armed += arm_fused_tp_ar(draft)
    return armed
