# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for kernels/experimental/moe_tp_decode_nvfp4.

Mirrors ``layers/adapter/moe_tp_decode.py`` for ``VllmNvfp4MoEMethod`` layers
behind ``USE_MOE_TP_DECODE_NVFP4_KERNEL`` (default off):
  - At ``n < 16``, pads ``x`` and ``scores`` to ``m = 16``, runs
    ``moe_tp_decode_nvfp4`` (W4A8 grouped FP4 decode kernel + in-kernel
    all-reduce), and slices ``[:n]``.
  - At ``n in (16, 32, 64)``, runs ``moe_tp_decode_nvfp4`` directly.
  - At ``n > 64`` (mixed / prefill buckets), runs ``moe_tp_decode._tp_chain``
    (``fused_moe_func`` -> ``gmm_v2`` W4A16 + online-FP8 shared expert) plus
    ``lax.psum`` on the same K-major FP4 weights and 4D block scales, with zero
    weight duplication.

Supports both EP x TP MoE (e.g. ``EP2xTP4`` via ``ENABLE_EXPERT_PARALLEL=true``
and ``TPU_MOE_EP_TP_SIZE=4``, where ``ep_size * tp_size == n_ranks`` with
linear expert placement) and plain TP MoE (``tp_size == n_ranks``).
"""

import dataclasses
import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental import moe_tp_decode_nvfp4 as kernel
from vllm_torchtpu.layers.adapter import (
    moe_routing,
    token_padding,
)
from vllm_torchtpu.layers.adapter import moe_tp_decode as fp8_decode
from vllm_torchtpu.layers.core.fused_moe_gmm import resolve_onehot_permute_threshold
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_ENV_FLAG = "USE_MOE_TP_DECODE_NVFP4_KERNEL"
_OP_ATTR = "_moe_tp_decode_nvfp4_op"
_BUFFERS = (
    "_moe_tp_decode_nvfp4_sw13",
    "_moe_tp_decode_nvfp4_ss13",
    "_moe_tp_decode_nvfp4_sw2",
    "_moe_tp_decode_nvfp4_ss2",
    "_moe_tp_decode_nvfp4_bias",
    "_moe_tp_decode_nvfp4_rank",
    "_moe_tp_decode_nvfp4_e0",
)

_OPS: dict[tuple, Any] = {}
_SELFTEST_DONE = False


@dataclasses.dataclass(frozen=True)
class _OpStatics:
    topk: int
    routed_scale: float
    activation: str
    use_sparse_core: bool
    onehot_threshold: int
    use_ep: bool = False
    router_bias: bool = False
    fast: bool = False
    hbm_operands: bool = False
    split_meta: bool = True
    w2_n_chunk: int = 2048


@dataclasses.dataclass(frozen=True)
class _Admitted:
    shared: tuple[torch.Tensor, ...]
    rows: tuple[int, ...]
    hidden: int
    statics: _OpStatics
    experts_start: int


def moe_tp_decode_nvfp4_supported(quant_method: Any) -> bool:
    """Whether arming gave this layer's NVFP4 quant method the fused TP op."""
    return getattr(quant_method, _OP_ATTR, None) is not None


def _sds(t: torch.Tensor) -> jax.ShapeDtypeStruct:
    """ShapeDtypeStruct matching what ``pallas.jax_op`` passes into JAX."""
    shape = tuple(t.shape)
    if t.dtype == torch.float4_e2m1fn_x2 and shape:
        shape = (*shape[:-1], shape[-1] * 2)
    return jax.ShapeDtypeStruct(shape, pallas.pallas.TORCH_TO_JAX_DTYPE_MAP[t.dtype])


def _admit(runner: Any, n_ranks: int) -> _Admitted | str:
    """Admission check for one ``MoERunner``, or a human-readable refusal."""
    from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
        ZeroExpertRouter,
    )

    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod

    layer = runner.routed_experts
    qm = layer.quant_method
    cfg = runner.moe_config
    par = cfg.moe_parallel_config
    if type(qm) is not VllmNvfp4MoEMethod:
        return f"quant method {type(qm).__name__} is not VllmNvfp4MoEMethod"
    if qm._tpu_activation_str != "silu":
        return f"activation {qm._tpu_activation_str!r} is not silu"
    if qm._use_adaptive_fused_moe or fused_moe_ep_supported(qm):
        return "a fused EP MoE kernel serves this layer"
    if envs.FUSED_MOE_DECODE_KERNEL:
        return "FUSED_MOE_DECODE_KERNEL is set; moe_tp_decode_nvfp4 stands down"
    if enable_pipelined_collective_and_compute() or envs.TPU_MOE_FP8_ALL_GATHER:
        return (
            "the MoE path owns its collectives "
            "(TPU_MOE_COLLECTION_CHUNK_SIZE or TPU_MOE_FP8_ALL_GATHER)"
        )
    if par.use_ep:
        if par.ep_size * par.tp_size != n_ranks:
            return (
                f"needs EP x TP MoE over {n_ranks} ranks "
                f"(ep {par.ep_size}, tp {par.tp_size})"
            )
        if getattr(layer, "expert_placement_strategy", "linear") != "linear":
            return "EP placement is not linear"
        if layer.w13_weight.shape[0] * par.ep_size != layer.global_num_experts:
            return (
                f"{layer.w13_weight.shape[0]} local experts x ep {par.ep_size} != "
                f"{layer.global_num_experts} experts"
            )
        experts_start = moe_routing.get_experts_start(layer)
        assert experts_start is not None
    elif par.tp_size != n_ranks:
        return f"needs TP MoE over {n_ranks} ranks (ep {par.ep_size}, tp {par.tp_size})"
    else:
        experts_start = 0
    if par.dp_size != 1 or par.pcp_size != 1 or cfg.is_sequence_parallel:
        return "needs dp 1, pcp 1 and no sequence-parallel MoE"
    if par.enable_eplb:
        return "EPLB is on"
    if runner.gate is None or runner.shared_expert_gate is not None:
        return "needs the runner to hold the gate and no shared-expert gate"
    if runner.routed_input_transform is not None:
        return "a routed input transform is set"
    if runner.routed_output_transform is not None:
        return "a routed output transform is set"
    if cfg.skip_final_all_reduce:
        return "skip_final_all_reduce is set"
    if isinstance(runner.router, ZeroExpertRouter):
        return "zero-expert router"
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "the routed experts have biases"
    grouped = (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    )
    if layer.scoring_func != "sigmoid" or not layer.renormalize or grouped:
        return "routing is not sigmoid + renormalized ungrouped top-k"
    if layer.e_score_correction_bias is None:
        return "no e_score_correction_bias"
    if layer.custom_routing_function is not None:
        return "custom routing function"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing"
    if moe_routing._SIMULATION_STRATEGY is not None:
        return "routing simulation is on"
    if moe_routing._ROUTER_TOPK != "rowmax":
        return f"TPU_MOE_ROUTER_TOPK={moe_routing._ROUTER_TOPK} (needs rowmax)"
    if layer.routed_scaling_factor != 1.0:
        return f"layer routed_scaling_factor {layer.routed_scaling_factor} != 1"
    if not kernel.exact_in_bf16(runner.routed_scaling_factor):
        return f"routed_scaling_factor {runner.routed_scaling_factor} is not bf16"
    shared = fp8_decode._shared_slice(runner)
    if isinstance(shared, str):
        return shared
    w13_scale = getattr(layer, "w13_weight_scale", None)
    w2_scale = getattr(layer, "w2_weight_scale", None)
    if w13_scale is None or w2_scale is None:
        return "missing w13_weight_scale / w2_weight_scale"
    hidden = layer.w13_weight.shape[1]
    experts = layer.global_num_experts
    topk = int(cfg.experts_per_token)
    weights = (
        layer.e_score_correction_bias,
        layer.w13_weight,
        layer.w2_weight,
        w13_scale,
        w2_scale,
        *shared,
    )
    rows = tuple(
        m
        for m in kernel.ROW_BUCKETS
        if kernel.can_use_moe_tp_decode_nvfp4(
            jax.ShapeDtypeStruct((m, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((m, experts), jnp.float32),
            *(_sds(t) for t in weights),
            topk=topk,
            n_ranks=n_ranks,
        )
    )
    if not rows:
        return (
            f"the kernel admits no token count (w13 {list(layer.w13_weight.shape)} "
            f"{layer.w13_weight.dtype}, s13 {list(w13_scale.shape)}, "
            f"s2 {list(w2_scale.shape)}, shared w13 {list(shared[0].shape)})"
        )
    statics = _OpStatics(
        topk=topk,
        routed_scale=float(runner.routed_scaling_factor),
        activation=qm._tpu_activation_str,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_threshold=resolve_onehot_permute_threshold(),
        use_ep=bool(par.use_ep),
        router_bias=bool(moe_routing._ROUTER_TOPK_BIAS),
        fast=bool(envs.TPU_MOE_TP_DECODE_FAST),
        split_meta=bool(envs.TPU_MOE_TP_DECODE_SPLIT_META),
    )
    return _Admitted(
        shared=shared,
        rows=rows,
        hidden=hidden,
        statics=statics,
        experts_start=int(experts_start),
    )


def _op_body(statics: _OpStatics, n_ranks: int, axis: str):
    """Per-rank body under ``shard_map`` over ``axis``."""

    def body(
        x,
        scores,
        bias,
        pad_mask,
        w13,
        w2,
        s13,
        s2,
        sw13,
        ss13,
        sw2,
        ss2,
        rank,
        e0,
    ):
        n = x.shape[0]
        is_pad = pad_mask[:n]
        n_valid = n - jnp.sum(is_pad.astype(jnp.int32))
        weights = (w13, w2, s13, s2, sw13, ss13, sw2, ss2)
        m = kernel.ROW_BUCKETS[0]
        if n < m:
            xp = jnp.pad(x, ((0, m - n), (0, 0)))
            sp = jnp.pad(scores, ((0, m - n), (0, 0)))
            if kernel.can_use_moe_tp_decode_nvfp4(
                xp,
                sp,
                bias,
                *weights,
                topk=statics.topk,
                n_ranks=n_ranks,
                w2_n_chunk=statics.w2_n_chunk,
            ):
                return kernel.moe_tp_decode_nvfp4(
                    xp,
                    sp,
                    bias,
                    n_valid,
                    rank.reshape(()),
                    *weights,
                    e0.reshape(()),
                    routed_scale=statics.routed_scale,
                    topk=statics.topk,
                    n_ranks=n_ranks,
                    fast=statics.fast,
                    hbm_operands=statics.hbm_operands,
                    split_meta=statics.split_meta,
                    w2_n_chunk=statics.w2_n_chunk,
                )[:n]
        if kernel.can_use_moe_tp_decode_nvfp4(
            x,
            scores,
            bias,
            *weights,
            topk=statics.topk,
            n_ranks=n_ranks,
            w2_n_chunk=statics.w2_n_chunk,
        ):
            return kernel.moe_tp_decode_nvfp4(
                x,
                scores,
                bias,
                n_valid,
                rank.reshape(()),
                *weights,
                e0.reshape(()),
                routed_scale=statics.routed_scale,
                topk=statics.topk,
                n_ranks=n_ranks,
                fast=statics.fast,
                hbm_operands=statics.hbm_operands,
                split_meta=statics.split_meta,
                w2_n_chunk=statics.w2_n_chunk,
            )
        partial = fp8_decode._tp_chain(
            x, scores, bias, is_pad, *weights, statics, e0.reshape(())
        )
        return lax.psum(partial, axis)

    return body


def _get_op(mesh: Any, statics: _OpStatics):
    """One ``pallas.jax_op`` per distinct closure, built at arming."""
    key = (mesh, statics)
    op = _OPS.get(key)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis), P(), P()) + (P(axis),) * 10
    body = _op_body(statics, int(mesh.size), axis)
    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    def run(*args):
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    def fn(
        x: jax.Array,
        scores: jax.Array,
        bias: jax.Array,
        pad_mask: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        sw13: jax.Array,
        ss13: jax.Array,
        sw2: jax.Array,
        ss2: jax.Array,
        rank: jax.Array,
        e0: jax.Array,
    ) -> jax.Array:
        return run(
            x, scores, bias, pad_mask, w13, w2, s13, s2, sw13, ss13, sw2, ss2, rank, e0
        )

    name = "pallas::moe_tp_decode_nvfp4" + (f"_{len(_OPS)}" if _OPS else "")
    # torch_tpu annotates partition specs as axis-name tuples; jax_op accepts
    # PartitionSpecs at runtime.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    _OPS[key] = op
    return op


def should_arm() -> bool:
    """Whether ``load_model`` should call ``arm`` for NVFP4 MoE layers."""
    return envs.USE_MOE_TP_DECODE_NVFP4_KERNEL


def arm(model: torch.nn.Module, draft_model: torch.nn.Module | None = None) -> int:
    """Arm every admitted NVFP4 MoE layer of ``model`` and ``draft_model``."""
    global _SELFTEST_DONE
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    topo = fp8_decode._topology()
    target = [m for m in model.modules() if isinstance(m, MoERunner)]
    known = {id(r) for r in target}
    drafter = (
        []
        if draft_model is None
        else [
            m
            for m in draft_model.modules()
            if isinstance(m, MoERunner) and id(m) not in known
        ]
    )
    plan, refused = [], {}
    for runner in target + drafter:
        if moe_tp_decode_nvfp4_supported(
            runner._quant_method
        ) or fp8_decode.moe_tp_decode_supported(runner._quant_method):
            continue
        if isinstance(topo, str):
            res: _Admitted | str = topo
        elif (
            not envs.TPU_MOE_SKIP_PADDED_TOKENS or token_padding._padding_state is None
        ):
            res = "needs TPU_MOE_SKIP_PADDED_TOKENS=1 (n_valid comes from its mask)"
        else:
            res = _admit(runner, int(topo.mesh.size))
        if isinstance(res, str):
            refused[res] = refused.get(res, 0) + 1
        else:
            plan.append((runner, res))
    buckets = sorted({(m, adm.hidden) for _, adm in plan for m in adm.rows})
    entries = [(r.layer_name, repr(adm.statics)) for r, adm in plan]
    sig = hashlib.sha256(
        repr((entries, len(target), len(drafter), buckets)).encode()
    ).hexdigest()[:16]
    mesh_index = -1 if isinstance(topo, str) else topo.mesh_index
    fp8_decode._check_ranks_agree(dict(sig=sig, mesh_index=mesh_index, armed=len(plan)))
    if plan and not _SELFTEST_DONE and not fp8_decode._SELFTEST_DONE:
        assert not isinstance(topo, str)
        device = plan[0][0].routed_experts.w13_weight.device
        failures = fp8_decode._selftest(topo, buckets, device)
        seen = fp8_decode._gather(dict(mesh_index=topo.mesh_index, failures=failures))
        if any(s["failures"] for s in seen):
            raise RuntimeError(
                "moe_tp_decode_nvfp4: the all-reduce self-test failed (per rank: "
                f"mesh index, failing (rows, hidden, n_valid)): {seen}"
            )
        _SELFTEST_DONE = True
    for runner, adm in plan:
        assert not isinstance(topo, str)
        layer = runner.routed_experts
        bias = layer.e_score_correction_bias.detach().to(torch.float32).contiguous()
        rank = fp8_decode._rank_tensor(layer.w13_weight.device, topo.mesh_index)
        e0 = fp8_decode._e0_tensor(layer.w13_weight.device, adm.experts_start)
        for name, tensor in zip(_BUFFERS, (*adm.shared, bias, rank, e0)):
            layer.register_buffer(name, tensor, persistent=False)
        statics = adm.statics
        mode = envs.TPU_MOE_TP_DECODE_HBM_OPERANDS
        if mode not in ("", "0", "drafter", "all"):
            raise ValueError(
                f"TPU_MOE_TP_DECODE_HBM_OPERANDS={mode!r}: use '', 'drafter' or 'all'"
            )
        if mode == "all" or (mode == "drafter" and id(runner) not in known):
            statics = dataclasses.replace(statics, hbm_operands=True)
        setattr(layer.quant_method, _OP_ATTR, _get_op(topo.mesh, statics))
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        runner.routed_scaling_factor = 1.0
    armed_drafter = sum(1 for r, _ in plan if id(r) not in known)
    log = logger.info if plan or not envs.is_set(_ENV_FLAG) else logger.warning
    log(
        "moe_tp_decode_nvfp4: armed %d/%d target and %d/%d drafter MoE layers "
        "(mesh index %d, kernel token counts %s, sig %s); refused: %s",
        len(plan) - armed_drafter,
        len(target),
        armed_drafter,
        len(drafter),
        mesh_index,
        sorted({m for m, _ in buckets}),
        sig,
        refused or "none",
    )
    return len(plan)


def moe_tp_decode_nvfp4_forward(
    quant_method: Any, layer: Any, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    """Forward of an armed NVFP4 layer: reduced ``shared + scale * routed``."""
    padding = token_padding._padding_state
    assert padding is not None, "moe_tp_decode_nvfp4 needs the token-padding mask"
    sw13, ss13, sw2, ss2, bias, rank, e0 = (getattr(layer, n) for n in _BUFFERS)
    scores = moe_routing._apply_scoring_fn(layer.scoring_func, router_logits)
    op = getattr(quant_method, _OP_ATTR)
    return op(
        x,
        scores,
        bias,
        padding.local_padding_mask,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
        sw13,
        ss13,
        sw2,
        ss2,
        rank,
        e0,
    )
