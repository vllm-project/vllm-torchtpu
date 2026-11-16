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
# Provenance: copied from src/vllm_torchtpu/layers/adapter/fused_moe_decode.py at
# 7d62e73201948ae35d23dbb6574ed14d5917784e,
# then #2624 (53e2d0ce8e) and #3219 (e57555be64) applied to the copy.
#   Owned by models/qwen4_exp from this copy on.
"""Serve a MoE layer's decode steps with the fused MoE decode kernel.

Under MoE tensor parallelism vLLM computes the shared expert separately, adds
it to the routed output and all-reduces the sum. Expert parallelism with TP
attention (DP 1) ends the same way, so its layers are served too, each rank
with its own expert range. An eligible layer runs this model's copy of
``fused_moe_decode`` (experimental/kernels/fused_moe_decode) under shard_map
over the mesh for up to ``MAX_TOKENS`` (256) tokens:
shared + routed experts and the all-reduce in one Pallas call per rank.

The model wires two entry points: ``arm_model`` once per loaded model (target
and draft) and ``fused_moe_decode_forward`` at the top of the MoE method's
forward, which returns the armed layer's output or None.

Arming happens once after the model is loaded (``arm_model``), when
the shared expert's linears have their final FP8 per-channel layout. For an
armed layer the runner is told it has no shared expert (the kernel owns it),
and ``MoERunner._fused_output_is_reduced`` reports True (see ``VllmMoERunner``
in ``layers/adapter/custom_ops/moe_runner.py``), so vLLM issues neither the
shared MLP nor any all-reduce. Above the token cap the same exported program
computes what the runner would have with the kernel off: the regular GMM MoE
(``fused_moe_func``, as the GMM custom op calls it), the shared expert and one
all-reduce of their sum. The token count is static there, so the branch costs
nothing at run time; branching in torch instead gives the two sides different
graph inputs, which vLLM's per-bucket specialization of one traced graph
refuses (``compilation.shape_variants``).

Two shared-expert conventions are served, each as vLLM's MoERunner computes it
with the kernel off: a shared expert gated by sigmoid(x . w_g) (Qwen3.5,
``expert_gate``) or an ungated ``DeepseekV2MLP`` (GLM-5.2). A
``runner.routed_scaling_factor`` (GLM-5.2: 2.5) scales the ROUTED output only,
before the shared output is added (``_maybe_apply_routed_scale_to_output``);
the op applies it the same way, and arming sets the runner's factor to 1.0,
since the runner would otherwise scale the op's shared + routed sum. A shared
expert the checkpoint leaves unquantized (Qwen3.8-Flash: BF16, TP slice of 80)
is served in BF16, its slice zero-padded to 128 at arming.

An armed layer takes precedence over the adaptive fused EP kernel
(MOE_FUSED_EP_KERNEL_IMPL=adaptive): the caller runs
``fused_moe_decode_forward`` before any other MoE path. Arming logs, per model,
how many layers it declined and why. It stands down while the shared
``FUSED_MOE_DECODE_KERNEL`` is set, and declines a layer moe_tp_decode armed.

Every rank must arm the same layers: the kernel's collectives start at entry,
and a rank serving a layer with the GMM path leaves its peers waiting. Arming
compares a signature across the EP group and raises on any mismatch.
"""

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs  # experimental-env-read: shared kernel switch
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, build_ep_mesh, ep_mesh_index
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.logger import init_logger

from .kernels.fused_moe_decode import fused_moe_decode as _kernel_body
from .kernels.fused_moe_decode import plan_weight_ring, vmem_bytes

logger = init_logger(__name__)

# The op lives on the quant method under a name of this model's own, so the
# shared fused_moe_decode adapter never sees a layer this copy armed.
_OP_ATTR = "_qwen38_fused_moe_decode_op"
# Read by the shared VllmMoERunner._fused_output_is_reduced: an armed layer
# reduces shared + routed itself. Same name as the shared adapter's on purpose.
_RUNNER_ARMED_ATTR = "_fused_moe_decode_armed"
# The shared expert's tensors become non-persistent buffers of the routed layer,
# so they stay module state Dynamo traces like any weight (a tensor reached
# through a plain attribute makes traces of the same graph disagree on inputs).
_SHARED_BUFFERS = (
    "_q38_fmd_w1s",
    "_q38_fmd_s1s",
    "_q38_fmd_w2s",
    "_q38_fmd_s2s",
    "_q38_fmd_wg",
)
# One exported program per distinct closure; the mesh never enters a traced
# frame (ops are built at arming time and only looked up in forward).
_OPS: dict[tuple, Any] = {}
# Token cap of the kernel path (served 256); larger steps run the GMM MoE.
MAX_TOKENS = 256
# Per rank [[mesh index, first global expert]]: the kernel addresses peers by
# mesh index, and the expert range follows vLLM's EP placement, which is not
# tied to mesh order.
_RANK_BUFFER: Any = None
_RANK_BUFFER_START = 0
_RANK_BUFFER_INDEX: int | None = None


def is_armed(quant_method: Any) -> bool:
    return getattr(quant_method, _OP_ATTR, None) is not None


def _gmm_layer(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    start,
    *,
    topk,
    shared_gated,
    routed_scale,
):
    """Above the cap: the layer as the runner computes it with the kernel off,
    shared expert + routed GMM MoE (scaled by routed_scale), all-reduced. Per
    rank, inside shard_map."""
    routed = fused_moe_func(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        None,
        None,
        tw,
        ti,
        jnp.zeros((), jnp.int32) if start is None else start,
        topk=topk,
        activation="silu",
        use_ep=start is not None,
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
        rhs_quant_dtype=None,
        skip_padded_tokens=True,
    )
    if routed_scale != 1.0:
        routed = routed * routed_scale
    # Qwen2MoeMLP.forward: FP8 (or, unquantized in the checkpoint, BF16)
    # gate_up / down linears, SiluAndMul, and the sigmoid of the replicated
    # expert_gate linear (DeepseekV2MLP: no gate).
    i_s = shared_w2.shape[0]
    if shared_w1.dtype == jnp.bfloat16:
        gate_up = jnp.dot(x, shared_w1)
        shared = jnp.dot(
            (jax.nn.silu(gate_up[:, :i_s]) * gate_up[:, i_s:]).astype(x.dtype),
            shared_w2,
        )
    else:
        gate_up = _quantized_matmul_jax(x, shared_w1, shared_w1_scale.reshape(-1))
        shared = _quantized_matmul_jax(
            jax.nn.silu(gate_up[:, :i_s]) * gate_up[:, i_s:],
            shared_w2,
            shared_w2_scale.reshape(-1),
        )
    if shared_gated:
        shared = (
            jax.nn.sigmoid(
                jnp.dot(x, shared_gate_weight.reshape(-1, 1).astype(x.dtype))
            )
            * shared
        )
    return jax.lax.psum(shared + routed, EP_AXIS_NAME)


def _layer(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    *,
    n_ranks,
    topk,
    expert_parallel,
    max_tokens,
    shared_gated=True,
    routed_scale=1.0,
):
    """Per-rank body of the op: the kernel up to `max_tokens`, GMM MoE above.
    An ungated layer still passes its placeholder shared_gate_weight operand
    (fixed op arity); shared_gated=False ignores it."""
    start = rank[0, 1] if expert_parallel else None
    if x.shape[0] > max_tokens:
        return _gmm_layer(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w2,
            shared_w2_scale,
            shared_gate_weight,
            tw,
            ti,
            start,
            topk=topk,
            shared_gated=shared_gated,
            routed_scale=routed_scale,
        )
    # Trace-time, once per token bucket: the evidence that the kernel serves
    # this shape, and in which mode (the kernel's name carries it too).
    ib, nb = plan_weight_ring(
        x.shape[0] + (-x.shape[0]) % 16,
        x.shape[1],
        w2.shape[1],
        shared_w2.shape[0],
        w2.shape[0],
        n_ranks,
        w1.dtype,
    )
    logger.info(
        "Fused MoE decode kernel traced for %d tokens (%s, ring %d)",
        x.shape[0],
        "whole experts" if ib is None else f"blocked ib={ib}",
        nb,
    )
    # Routing prologue folded into the kernel with the gate in XLA ("xla"
    # fold, served everywhere in agentx). Operands as stored: per-call
    # reshapes are XLA ops of their own.
    return _kernel_body(
        rank,
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight if shared_gated else None,
        tw,
        ti,
        n_ranks=n_ranks,
        expert_start="rank" if expert_parallel else None,
        routed_scale=routed_scale,
        fold=True,
        gate_in_kernel=False,
    )


def _build_op(
    mesh,
    n_ranks: int,
    topk: int,
    expert_parallel: bool,
    max_tokens: int,
    shared_gated: bool = True,
    routed_scale: float = 1.0,
):
    key = (
        id(mesh),
        n_ranks,
        topk,
        expert_parallel,
        max_tokens,
        shared_gated,
        routed_scale,
    )
    op = _OPS.get(key)
    if op is not None:
        return op
    spec = PartitionSpec(EP_AXIS_NAME)
    local = functools.partial(
        _layer,
        n_ranks=n_ranks,
        topk=topk,
        expert_parallel=expert_parallel,
        max_tokens=max_tokens,
        shared_gated=shared_gated,
        routed_scale=routed_scale,
    )

    # torch_tpu infers static arguments from annotations: every operand is a
    # jax.Array. Per-rank operands are sharded on axis 0; the gate is replicated.
    def fn(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        shared_w1: jax.Array,
        shared_w1_scale: jax.Array,
        shared_w2: jax.Array,
        shared_w2_scale: jax.Array,
        shared_gate_weight: jax.Array,
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
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w2,
            shared_w2_scale,
            shared_gate_weight,
            tw,
            ti.astype(jnp.int32),
            rank,
        )

    name = "pallas::qwen4_exp_fused_moe_decode"
    if _OPS:
        name = f"pallas::qwen4_exp_fused_moe_decode_{len(_OPS)}"
    # torch_tpu types the partition specs as tuples of axis names, but takes
    # (and fused_moe_ep passes) jax PartitionSpecs.
    # pyrefly: ignore[no-matching-overload]
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


def _shared_mlp_is_ungated_silu(mlp) -> bool:
    """A shared expert with no gate of its own: gate_up -> SiluAndMul -> down
    (vLLM's DeepseekV2MLP, GLM-5.2's shared expert), nothing else in forward."""
    from vllm.model_executor.layers.activation import SiluAndMul

    return (
        type(mlp).__name__ == "DeepseekV2MLP"
        and isinstance(getattr(mlp, "act_fn", None), SiluAndMul)
        and getattr(mlp, "expert_gate", None) is None
    )


def pad_shared_intermediate(
    w1: torch.Tensor, w2: torch.Tensor, i_pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad a shared-expert slice's intermediate size to i_pad: w1 [h, 2 i_s]
    (gate | up) to [h, 2 i_pad] (gate | 0 | up | 0), w2 [i_s, h] to [i_pad, h].
    Exact: a padded column has gate = up = 0, so SiLU(0) * 0 = 0 meets a zero
    row of w2. The kernel slices gate and up at lane-aligned offsets only
    (Qwen3.8-Flash at TP8: i_s = 640 / 8 = 80, padded to 128)."""
    h, two_i_s = w1.shape
    i_s = two_i_s // 2
    if i_pad == i_s:
        return w1, w2
    pad1 = w1.new_zeros((h, i_pad - i_s))
    w1 = torch.cat([w1[:, :i_s], pad1, w1[:, i_s:], pad1], dim=1).contiguous()
    w2 = torch.cat([w2, w2.new_zeros((i_pad - i_s, w2.shape[1]))], dim=0)
    return w1, w2.contiguous()


def _is_flipped_bf16_linear(linear) -> bool:
    """An unquantized linear stored K-major ([in, out]) by
    VllmUnquantizedLinearMethod: BF16 weight, no scale, no bias."""
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and getattr(linear, WEIGHT_FLIPPED_ATTR, False)
        and getattr(linear, "weight_scale", None) is None
        and getattr(linear, "weight_scale_inv", None) is None
        and getattr(linear, "bias", None) is None
    )


def _shared_slice(runner, why: list | None = None) -> tuple | None:
    """This rank's shared-expert tensors in the kernel's layout, or None (with
    the reason appended to `why`). The last entry, the gate weight, is None for
    an ungated shared expert.

    Two storage layouts: FP8 per-channel (K-major weight, 1-D scale), handed
    over as stored; or BF16 (unquantized in the checkpoint, Qwen3.8-Flash),
    copied with its intermediate zero-padded to a multiple of 128 and unit
    scales the kernel ignores."""
    why = [] if why is None else why
    shared = getattr(runner, "_shared_experts", None)
    mlp = getattr(shared, "_layer", None)
    if mlp is None:
        why.append("no shared expert module")
        return None
    gate_up = getattr(mlp, "gate_up_proj", None)
    down = getattr(mlp, "down_proj", None)
    gate = getattr(mlp, "expert_gate", None)
    if gate_up is None or down is None:
        why.append(f"shared expert {type(mlp).__name__} has no gate_up/down_proj")
        return None
    if gate is None and not _shared_mlp_is_ungated_silu(mlp):
        why.append(f"ungated shared expert {type(mlp).__name__} not DeepseekV2MLP")
        return None
    shared_gate_weight = None if gate is None else getattr(gate, "weight", None)
    if gate is not None and (
        shared_gate_weight is None
        or getattr(gate, "bias", None) is not None
        or shared_gate_weight.dim() != 2
        or 1 not in tuple(shared_gate_weight.shape)
    ):
        why.append(
            "shared expert gate not a bias-free [1, h] / [h, 1] linear: "
            f"{None if shared_gate_weight is None else tuple(shared_gate_weight.shape)}"
        )
        return None
    if _is_flipped_bf16_linear(gate_up) and _is_flipped_bf16_linear(down):
        w1, w2 = gate_up.weight.data, down.weight.data
        i_s = w2.shape[0]
        if tuple(w1.shape) != (w2.shape[1], 2 * i_s):
            why.append(f"BF16 shared expert shapes {tuple(w1.shape)} {tuple(w2.shape)}")
            return None
        w1, w2 = pad_shared_intermediate(w1, w2, -(-i_s // 128) * 128)
        return (
            w1,
            torch.ones(w1.shape[1], dtype=torch.float32, device=w1.device),
            w2,
            torch.ones(w2.shape[1], dtype=torch.float32, device=w1.device),
            None if shared_gate_weight is None else shared_gate_weight.data,
        )
    shared_w1, shared_w1_scale = (
        getattr(gate_up, "weight", None),
        getattr(gate_up, "weight_scale", None),
    )
    shared_w2, shared_w2_scale = (
        getattr(down, "weight", None),
        getattr(down, "weight_scale", None),
    )
    if (
        shared_w1 is None
        or shared_w1_scale is None
        or shared_w2 is None
        or shared_w2_scale is None
    ):
        why.append(
            "shared expert neither FP8 with weight_scale nor flipped BF16 "
            f"(gate_up {getattr(shared_w1, 'dtype', None)})"
        )
        return None
    # Per-channel FP8 linears are stored K-major ([in, out]) with a 1-D scale.
    if (
        shared_w1_scale.dim() != 1
        or shared_w2_scale.dim() != 1
        or shared_w1.dtype != torch.float8_e4m3fn
    ):
        why.append(
            f"shared expert FP8 scales not per-channel ({shared_w1.dtype}, "
            f"scale {tuple(shared_w1_scale.shape)} / {tuple(shared_w2_scale.shape)})"
        )
        return None
    # Hand over the parameters themselves, never torch views of them: a view
    # passed to the op is re-materialized by an eager as_strided program on
    # every call (measured: one per layer per step, ~48 us of host dispatch
    # each). The kernel's [1, N] / [H, 1] shapes are made inside the program.
    return (
        shared_w1.data,
        shared_w1_scale.data,
        shared_w2.data,
        shared_w2_scale.data,
        None if shared_gate_weight is None else shared_gate_weight.data,
    )


def _admit(
    runner, n_ranks: int, why: list | None = None
) -> tuple[tuple, bool, float] | None:
    """(shared-expert tensors, expert_parallel, routed scale) for an eligible
    layer, else None, with the reason appended to `why`."""
    why = [] if why is None else why
    result = _admit_checked(runner, n_ranks, why)
    if result is None and not why:
        why.append("unknown")
    return result


def _admit_checked(runner, n_ranks: int, why: list) -> tuple[tuple, bool, float] | None:
    from vllm.distributed import get_tensor_model_parallel_world_size

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    cfg = runner.moe_config
    par = cfg.moe_parallel_config
    qm = runner._quant_method
    layer = runner.routed_experts
    expert_parallel = bool(par.use_ep)
    if expert_parallel:
        sharded_ok = par.ep_size == n_ranks and par.dp_size == 1
    else:
        sharded_ok = par.tp_size == n_ranks
    # TP attention over the whole mesh: every rank holds every token and one
    # shared-expert slice, and MoERunner runs no DP/PCP collective after the op.
    tp_world = get_tensor_model_parallel_world_size()
    sharded_ok = sharded_ok and tp_world == n_ranks
    if not isinstance(qm, VllmFp8MoEMethodTPU):
        why.append(f"quant method {type(qm).__name__}")
        return None
    from vllm_torchtpu.layers.adapter.moe_tp_decode import moe_tp_decode_supported

    if moe_tp_decode_supported(qm):
        # moe_tp_decode stands down only for the shared FUSED_MOE_DECODE_KERNEL.
        why.append("armed by moe_tp_decode")
        return None
    if not sharded_ok:
        why.append(
            f"parallelism (use_ep {expert_parallel}, ep {par.ep_size}, dp "
            f"{par.dp_size}, moe tp {par.tp_size}, tp world {tp_world}, mesh "
            f"{n_ranks})"
        )
        return None
    if qm._tpu_activation_str != "silu":
        why.append(f"activation {qm._tpu_activation_str}")
        return None
    if getattr(layer, "w13_bias", None) is not None or (
        getattr(layer, "w2_bias", None) is not None
    ):
        why.append("expert bias")
        return None
    if getattr(cfg, "is_sequence_parallel", False) or getattr(
        cfg, "skip_final_all_reduce", False
    ):
        why.append("sequence parallel / skip_final_all_reduce")
        return None
    if (
        runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
    ):
        why.append("routed input/output transform")
        return None
    if runner.shared_expert_gate is not None:
        # The gate lives on the runner only when the model fuses or replicates
        # its shared expert outside the runner; not served.
        why.append("runner-level shared_expert_gate")
        return None
    routed_scale = float(getattr(runner, "routed_scaling_factor", 1.0))
    if routed_scale != 1.0 and getattr(cfg, "in_dtype", None) == torch.float16:
        # FP16: vLLM divides the shared output instead; not mirrored here.
        why.append("routed scale under FP16")
        return None
    shared = _shared_slice(runner, why)
    if shared is None:
        return None
    shared_w1, _, shared_w2, _, shared_gate_weight = shared
    w1, w2 = layer.w13_weight, layer.w2_weight
    w1_scale, w2_scale = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    e, h, two_i = w1.shape
    i = two_i // 2
    i_s = shared_w2.shape[0]  # the shared expert is a TP slice, even under EP
    # The kernel pads the all-reduce chunk (h / n_ranks) to whole lanes itself.
    if (
        w1.dtype != torch.float8_e4m3fn
        or tuple(w2.shape) != (e, i, h)
        or tuple(w1_scale.shape) != (e, 1, 1, two_i)
        or tuple(w2_scale.shape) != (e, 1, 1, h)
        or i % 128
        or h % 512
        or i_s % 128
        or i_s > i
        or tuple(shared_w1.shape) != (h, 2 * i_s)
        or (shared_gate_weight is not None and shared_gate_weight.numel() != h)
    ):
        why.append(
            f"shapes (w13 {tuple(w1.shape)} {w1.dtype}, w2 {tuple(w2.shape)}, "
            f"scales {tuple(w1_scale.shape)} / {tuple(w2_scale.shape)}, shared w1 "
            f"{tuple(shared_w1.shape)}, i_s {i_s})"
        )
        return None
    return shared, expert_parallel, routed_scale


def _check_vmem(max_tokens: int, layer, i_s: int, n_ranks: int) -> None:
    """Refuse a token cap the kernel cannot be built for, at arming rather than
    as a Mosaic VMEM error when the first bucket above it compiles."""
    e, h, two_i = layer.w13_weight.shape
    weight_dtype = jnp.dtype(str(layer.w13_weight.dtype).removeprefix("torch."))
    need = vmem_bytes(max_tokens, h, two_i // 2, i_s, e, n_ranks, weight_dtype)
    have = pltpu.get_tpu_info().vmem_capacity_bytes
    if need > have:
        raise ValueError(
            f"Fused MoE decode token cap {max_tokens} needs "
            f"{need / 2**20:.1f} MiB of VMEM for this layer (experts {e}, "
            f"hidden {h}, intermediate {two_i // 2}, {n_ranks} ranks) and the "
            f"TPU has {have / 2**20:.1f} MiB; lower MAX_TOKENS."
        )


def _check_same_arming_on_every_rank(signature: tuple, what: str) -> None:
    """Raise unless every EP rank computed the same arming signature: a rank
    that serves a layer with the GMM path while its peers run the kernel hangs
    the slice, so failing at load with every rank's answer is the safe outcome."""
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group

    group = get_ep_group()
    if group is None:
        return
    gathered: list = [None] * int(group.world_size)
    torch.distributed.all_gather_object(gathered, signature, group=group.cpu_group)
    if any(g != gathered[0] for g in gathered):
        summary = sorted({(len(g[1]), str(g)[:120]) for g in gathered})
        raise RuntimeError(
            f"fused MoE decode kernel ({what}): EP ranks disagree on which layers "
            f"to arm; refusing to serve a collective kernel on some ranks only: "
            f"{summary}"
        )


def arm_model(model: torch.nn.Module, what: str = "model") -> int:
    """Arm every eligible MoE layer of `model` (target or draft); returns how
    many were armed.

    Call once per loaded model, after its weights have their final layout.
    Collective over the EP group when EP is on: every rank must call it."""
    if envs.FUSED_MOE_DECODE_KERNEL:  # experimental-env-read: shared kernel switch
        logger.warning(
            "Model fused MoE decode not armed: the shared FUSED_MOE_DECODE_KERNEL "
            "is set and arms the layers itself"
        )
        return 0
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runners = [m for m in model.modules() if isinstance(m, MoERunner)]
    if not runners:
        return 0
    try:
        mesh = build_ep_mesh()
    except Exception as exc:  # refuse rather than serve a wrong reduction
        logger.warning("Fused MoE decode kernel not armed: no mesh (%s)", exc)
        mesh = None
    n_ranks = mesh.shape[EP_AXIS_NAME] if mesh is not None else 0
    global _RANK_BUFFER, _RANK_BUFFER_START, _RANK_BUFFER_INDEX
    max_tokens = MAX_TOKENS
    plan = []
    rejected: dict[str, int] = {}
    for runner in runners if mesh is not None else ():
        why: list = []
        admitted = _admit(runner, n_ranks, why)
        if admitted is not None:
            plan.append((runner, *admitted))
        else:
            reason = "; ".join(why)
            rejected[reason] = rejected.get(reason, 0) + 1
            logger.debug(
                "Fused MoE decode kernel: %s not armed: %s",
                getattr(runner, "layer_name", "?"),
                reason,
            )
    for reason, count in rejected.items():
        logger.info(
            "Fused MoE decode kernel declined %d MoE layer(s) of the %s: %s",
            count,
            what,
            reason,
        )
    signature = (
        n_ranks,
        tuple(
            (getattr(r, "layer_name", "?"), ep, shared[4] is not None, scale)
            for r, shared, ep, scale in plan
        ),
    )
    _check_same_arming_on_every_rank(signature, what)
    armed = 0
    for runner, shared, expert_parallel, routed_scale in plan:
        layer = runner.routed_experts
        w13_weight = cast(torch.Tensor, layer.w13_weight)
        _check_vmem(max_tokens, layer, shared[2].shape[0], n_ranks)
        start = moe_routing.get_experts_start(layer) or 0
        if _RANK_BUFFER is None:
            _RANK_BUFFER_INDEX = ep_mesh_index()
            _RANK_BUFFER = torch.tensor(
                [[_RANK_BUFFER_INDEX, start]],
                dtype=torch.int32,
                device=w13_weight.device,
            )
            _RANK_BUFFER_START = start
        elif start != _RANK_BUFFER_START:
            # A collective kernel must arm on every rank or none.
            raise RuntimeError(
                "fused MoE decode kernel: layers on this rank start at different "
                f"experts ({_RANK_BUFFER_START} vs {start})"
            )
        shared_gated = shared[4] is not None
        if not shared_gated:
            # Placeholder for the op's fixed gate operand; the program ignores it.
            shared = (
                *shared[:4],
                torch.zeros(
                    (1, w13_weight.shape[1]),
                    dtype=torch.bfloat16,
                    device=w13_weight.device,
                ),
            )
        setattr(
            runner._quant_method,
            _OP_ATTR,
            _build_op(
                mesh,
                n_ranks,
                layer.moe_config.experts_per_token,
                expert_parallel,
                max_tokens,
                shared_gated,
                routed_scale,
            ),
        )
        for name, tensor in zip(_SHARED_BUFFERS, shared):
            layer.register_buffer(name, tensor, persistent=False)
        runner._shared_experts = None
        # The op scales the routed experts only; the runner would scale the
        # op's shared + routed sum.
        runner.routed_scaling_factor = 1.0
        runner._forward_entry = runner._select_forward()
        object.__setattr__(runner, _RUNNER_ARMED_ATTR, True)
        armed += 1
    # Every rank logs: the evidence that all ranks armed the same layers.
    logger.info(
        "Armed the fused MoE decode kernel on %d of %d MoE layers of the %s "
        "(ep_mesh_index %s, experts_start %s, %d ranks, shared gate %s, "
        "routed_scale %s, up to %d tokens; GMM MoE above)",
        armed,
        len(runners),
        what,
        _RANK_BUFFER_INDEX if armed else None,
        _RANK_BUFFER_START if armed else None,
        n_ranks,
        sorted({s[2] for s in signature[1]}),
        sorted({s[3] for s in signature[1]}),
        max_tokens,
    )
    return armed


def _run_armed(
    quant_method: Any, layer: Any, x: torch.Tensor, topk_weights, topk_ids
) -> torch.Tensor:
    """Forward for an armed layer: the all-reduced (shared + routed) output.

    One torch op for every token count; above the cap the exported program
    runs the GMM MoE path (with its own all-reduce), so an armed layer always
    returns the reduced sum.
    """
    shared_w1, shared_w1_scale, shared_w2, shared_w2_scale, shared_gate_weight = (
        getattr(layer, name) for name in _SHARED_BUFFERS
    )
    op = getattr(quant_method, _OP_ATTR)
    return op(
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight,
        topk_weights,
        topk_ids,
        _RANK_BUFFER,
    )


def fused_moe_decode_forward(
    method: Any, layer: Any, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor | None:
    """The armed layer's output, or None when `method` is not armed.

    Armed: replicated tokens in, the all-reduced shared + routed sum out (above
    the token cap the same op runs the GMM MoE). Call it before any other MoE
    path, so an armed layer also takes precedence over the adaptive fused EP
    kernel."""
    if not is_armed(method):
        return None
    topk_weights, topk_ids = moe_routing.route(layer, x, router_logits)
    topk_ids, topk_weights = token_padding.zero_routing_weights_for_padding(
        topk_ids, topk_weights, is_local_tensor=False
    )
    return _run_armed(method, layer, x, topk_weights, topk_ids)
