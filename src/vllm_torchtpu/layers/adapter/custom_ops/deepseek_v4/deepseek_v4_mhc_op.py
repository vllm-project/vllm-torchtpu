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
"""DeepSeek-V4 mHC Pallas kernels.

Drop-in replacements for vLLM's ``mhc_pre_torch`` / ``mhc_post_torch``, plus
the fused seam op that runs one layer's post together with the next layer's
pre as a single kernel.
"""

import functools
from typing import Any, NamedTuple, cast

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.deepseek_v4.mhc import (
    fused_post_pre_kernel,
    norm_quant_kernel,
    post_kernel,
    pre_kernel,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)

# Every mHC tensor is token-major, so its leading axis is the attention-DP
# token axis. As in the other DeepSeek-V4 ops, the mesh carries no such axis
# today, so it resolves to None.
ATTN_DATA_AXIS = None

_mhc_op_cache: dict[str, Any] = {}


def _name_float(value: float) -> str:
    """A float rendered so it is safe inside a torch op name."""
    return f"{value:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _gate_suffix(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> str:
    """The gate constants are baked into the traced op, so they name it."""
    return (
        f"_r{_name_float(rms_eps)}_p{_name_float(hc_pre_eps)}"
        f"_s{_name_float(hc_sinkhorn_eps)}"
        f"_a{_name_float(hc_post_mult_value)}_i{sinkhorn_repeat}"
    )


# Module-level so `pallas.jax_op` can trace and register them as torch ops.
def _mhc_pre_jax(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mix GEMM + stream collapse in Pallas, gates/Sinkhorn in XLA."""
    return pre_kernel.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def _mhc_post_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    """Recombine the sublayer output into the mHC residual streams."""
    return post_kernel.mhc_post(x, residual, post_layer_mix, comb_res_mix)


def _mhc_fused_post_pre_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One layer's post and the next layer's pre as a single kernel."""
    return fused_post_pre_kernel.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def _fake_pre_outputs(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(post_mix, comb_mix, layer_input) placeholders for Dynamo tracing."""
    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    return (
        torch.empty(
            (*outer_shape, hc_mult, 1), dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            (*outer_shape, hc_mult, hc_mult),
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            (*outer_shape, hidden_size), dtype=residual.dtype, device=residual.device
        ),
    )


def _fake_mhc_pre(residual, fn, hc_scale, hc_base, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return _fake_pre_outputs(residual)


def _fake_mhc_post(x, residual, post_layer_mix, comb_res_mix, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return torch.empty_like(residual)


def _fake_mhc_fused_post_pre(
    x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, *args, **kwargs
):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return (torch.empty_like(residual), *_fake_pre_outputs(residual))


def _mhc_pre_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    op_name = "pallas::deepseek_v4_mhc_pre_v1" + _gate_suffix(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_pre_jax,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(),  # fn
                P(),  # hc_scale
                P(),  # hc_base
            ),
        )
        op.register_fake(_fake_mhc_pre)
        _mhc_op_cache[op_name] = op
    return op


def get_mhc_post_op():
    """The ``post`` op alone, for callers that settle a deferred post.

    ``post`` takes no gate constants, so this needs none. Same
    outside-the-trace requirement as ``get_mhc_ops``.
    """
    op_name = "pallas::deepseek_v4_mhc_post_v1"
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            _mhc_post_jax,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
            ),
        )
        op.register_fake(_fake_mhc_post)
        _mhc_op_cache[op_name] = op
    return op


def _mhc_fused_post_pre_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    op_name = "pallas::deepseek_v4_mhc_fused_post_pre_v1" + _gate_suffix(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_fused_post_pre_jax,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
                P(),  # fn
                P(),  # hc_scale
                P(),  # hc_base
            ),
        )
        op.register_fake(_fake_mhc_fused_post_pre)
        _mhc_op_cache[op_name] = op
    return op


class MHCOps(NamedTuple):
    """Registered mHC ops for one set of gate constants.

    ``pre`` and ``post`` match vLLM's ``mhc_pre_torch`` / ``mhc_post_torch``
    contracts; ``fused`` is the seam op, equivalent to ``post`` followed by
    ``pre`` on its output but sharing one pass over the residual streams.

    ``pre(residual, fn, hc_scale, hc_base)``
        residual (..., hc_mult, hidden_size) bf16 in; post_mix
        (..., hc_mult, 1) f32, comb_mix (..., hc_mult, hc_mult) f32,
        layer_input (..., hidden_size) bf16 out.
    ``post(x, residual, post_layer_mix, comb_res_mix)``
        (..., hc_mult, hidden_size) bf16 out.
    ``fused(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base)``
        residual_cur -- the ``post`` output -- then ``pre``'s three outputs
        computed from it.
    """

    # Norm/quant ops retain BF16 output for the other attention consumers.
    pre: Any
    fused: Any
    post: Any
    pre_norm_quant: Any = None
    fused_norm_quant: Any = None
    fused_norm: Any = None
    broadcast_pre_norm_quant: Any = None


def get_mhc_ops(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> MHCOps:
    """Build (or fetch) every mHC op for these gate constants.

    Must be called outside the compiled region: the builders construct jax
    ``PartitionSpec``s, which Dynamo cannot trace, so a cold cache inside a
    compiled forward is a hard error rather than a graph break. Callers hold
    the returned ops and invoke them directly, keeping the op lookup out of
    the traced graph entirely.
    """
    from vllm_torchtpu.layers.adapter.linear_common import (
        get_prequantized_bf16_matmul_op,
    )

    get_prequantized_bf16_matmul_op()
    constants = (
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return MHCOps(
        pre=_mhc_pre_op(
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        ),
        fused=_mhc_fused_post_pre_op(
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        ),
        post=get_mhc_post_op(),
        pre_norm_quant=_mhc_norm_quant_op(constants, fused=False),
        fused_norm_quant=_mhc_norm_quant_op(constants, fused=True),
        fused_norm=_mhc_fused_norm_op(constants),
        broadcast_pre_norm_quant=_mhc_broadcast_pre_norm_quant_op(constants),
    )


def _mhc_pre_norm_quant_jax(
    residual: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    weight: jax.Array,
    *,
    constants: norm_quant_kernel.GateConstants,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    post, comb, y, quant = norm_quant_kernel.mhc_pre_norm_quant(
        residual,
        fn,
        scale,
        base,
        weight,
        constants=constants,
        norm_eps=constants[0],
        quant_dtype=jax.numpy.float8_e4m3fn,
    )
    return post, comb, y, quant[0], quant[1]


def _mhc_broadcast_pre_norm_quant_jax(
    x: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    weight: jax.Array,
    *,
    constants: norm_quant_kernel.GateConstants,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    residual, post, comb, y, quant = norm_quant_kernel.mhc_broadcast_pre_norm_quant(
        x, fn, scale, base, weight, constants=constants, norm_eps=constants[0]
    )
    return residual, post, comb, y, quant[0], quant[1]


def _fake_broadcast_pre_norm_quant(x, fn, scale, base, weight):
    residual = torch.empty((x.shape[0], 4, x.shape[1]), dtype=x.dtype, device=x.device)
    return residual, *_fake_pre_norm_quant(residual, fn, scale, base, weight)


def _mhc_broadcast_pre_norm_quant_op(constants: norm_quant_kernel.GateConstants):
    name = "pallas::mhc_broadcast_pre_norm_quant" + _gate_suffix(*constants)
    op = _mhc_op_cache.get(name)
    if op is None:
        op = pallas.jax_op(
            name,
            functools.partial(_mhc_broadcast_pre_norm_quant_jax, constants=constants),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=cast(tuple[tuple[str, ...], ...], (P(),) * 5),
        )
        op.register_fake(_fake_broadcast_pre_norm_quant)
        _mhc_op_cache[name] = op
    return op


def _mhc_fused_norm_quant_jax(
    x: jax.Array,
    residual: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    weight: jax.Array,
    *,
    constants: norm_quant_kernel.GateConstants,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    residual, post, comb, y, quant = norm_quant_kernel.mhc_fused_post_pre_norm_quant(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        weight,
        constants=constants,
        norm_eps=constants[0],
        quant_dtype=jax.numpy.float8_e4m3fn,
    )
    return residual, post, comb, y, quant[0], quant[1]


def _fake_pre_norm_quant(residual, fn, scale, base, weight):
    post, comb, y = _fake_pre_outputs(residual)
    return (
        post,
        comb,
        y,
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty((*y.shape[:-1], 1), dtype=torch.float32, device=y.device),
    )


def _fake_fused_norm_quant(x, residual, post, comb, fn, scale, base, weight):
    return (
        torch.empty_like(residual),
        *_fake_pre_norm_quant(residual, fn, scale, base, weight),
    )


def _mhc_norm_quant_op(constants: norm_quant_kernel.GateConstants, *, fused: bool):
    suffix = _gate_suffix(*constants)
    name = "pallas::mhc_" + ("fused" if fused else "pre") + "_norm_quant" + suffix
    op = _mhc_op_cache.get(name)
    if op is None:
        mesh = get_pallas_op_mesh()
        op = pallas.jax_op(
            name,
            functools.partial(
                _mhc_fused_norm_quant_jax if fused else _mhc_pre_norm_quant_jax,
                constants=constants,
            ),
            mesh=mesh,
            input_partition_specs=cast(
                tuple[tuple[str, ...], ...], (P(),) * (8 if fused else 5)
            ),
        )
        op.register_fake(_fake_fused_norm_quant if fused else _fake_pre_norm_quant)
        _mhc_op_cache[name] = op
    return op


def _mhc_fused_norm_jax(
    x: jax.Array,
    residual: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    weight: jax.Array,
    *,
    constants: norm_quant_kernel.GateConstants,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    residual, post, comb, y, _ = norm_quant_kernel.mhc_fused_post_pre_norm_quant(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        weight,
        constants=constants,
        norm_eps=constants[0],
        quant_dtype=None,
    )
    return residual, post, comb, y


def _fake_fused_norm(x, residual, post, comb, fn, scale, base, weight):
    return torch.empty_like(residual), *_fake_pre_outputs(residual)


def _mhc_fused_norm_op(constants: norm_quant_kernel.GateConstants):
    name = "pallas::mhc_fused_norm" + _gate_suffix(*constants)
    op = _mhc_op_cache.get(name)
    if op is None:
        op = pallas.jax_op(
            name,
            functools.partial(_mhc_fused_norm_jax, constants=constants),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=cast(tuple[tuple[str, ...], ...], (P(),) * 8),
        )
        op.register_fake(_fake_fused_norm)
        _mhc_op_cache[name] = op
    return op
