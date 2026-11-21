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

With ``TPU_DSPARK_FUSED_MLP=1``, `arm_fused_mlp` (once after load, when the
linears hold their K-major e4m3 weights) hands each draft layer's MLP to one
torch op: kernels/dspark_mlp.py at up to MAX_ROWS rows, the same math in XLA
above (decided on the static row count inside the op, so every bucket traces
the same graph). The TP all-reduce after it is unchanged.
"""

import jax
import torch
from torch_tpu._internal import pallas
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMulWithClamp

from vllm_torchtpu.kernels import dspark_mlp as kernel
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP
from vllm_torchtpu.models.minimax_m3.quant import (
    _KERNEL_SWIGLU,
    MiniMaxM3Mxfp8LinearMethod,
)
from vllm_torchtpu.utils import fresh_buffer

logger = init_logger(__name__)

# Non-persistent buffers of an armed MLP, in kernel.fused_mlp's order.
_BUFFERS = ("_fused_w_gu", "_fused_s_gu", "_fused_w_d", "_fused_s_d")


def _mlp(
    x: jax.Array, w_gu: jax.Array, s_gu: jax.Array, w_d: jax.Array, s_d: jax.Array
) -> jax.Array:
    if kernel.supported(x.shape[0], x.shape[1], w_d.shape[0]):
        return kernel.fused_mlp(x, w_gu, s_gu, w_d, s_d)
    return kernel.reference(x, w_gu, s_gu, w_d, s_d)


# Registered at import: Dynamo cannot trace a registration inside a forward.
_op = pallas.jax_op("pallas::dspark_fused_mlp", _mlp)
_op.register_fake(lambda x, w_gu, s_gu, w_d, s_d: torch.empty_like(x))


class MiniMaxM3DSparkMLP(MiniMaxM3MLP):
    """MiniMaxM3MLP that runs as the fused op once `arm_fused_mlp` armed it."""

    def forward(self, x):
        if (
            getattr(self, _BUFFERS[0], None) is None
            or not isinstance(x, torch.Tensor)
            or x.dim() != 2
        ):
            return super().forward(x)
        out = _op(x, *(getattr(self, name) for name in _BUFFERS))
        if self.down_proj.reduce_results and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out


def _operands(mlp: MiniMaxM3DSparkMLP) -> tuple[torch.Tensor, ...] | None:
    """(w_gu, s_gu, w_d, s_d) in kernel.fused_mlp's layout, or None when the
    MLP is not bias-free SwiGLU-OAI on per-channel e4m3 linears."""
    act = mlp.act_fn
    if not isinstance(act, SiluAndMulWithClamp) or (
        (act.alpha, act.swiglu_limit, act.beta) != _KERNEL_SWIGLU
    ):
        return None
    linears = (mlp.gate_up_proj, mlp.down_proj)
    for linear in linears:
        if (
            not isinstance(linear.quant_method, MiniMaxM3Mxfp8LinearMethod)
            or linear.weight.dtype != torch.float8_e4m3fn
            or linear.weight.dim() != 2
            or linear.weight_scale.dim() != 1
            or getattr(linear, "bias", None) is not None
        ):
            return None
    (w1, s1), (w2, s2) = ((m.weight.data, m.weight_scale.data) for m in linears)
    h, two_i = w1.shape
    if two_i % 2 or tuple(w2.shape) != (two_i // 2, h):
        return None
    # Weights as the parameters themselves (a view would be re-materialized on
    # every call); the f32 [1, n] scales are buffers of their own.
    return (
        w1,
        fresh_buffer(s1.float().reshape(1, -1)),
        w2,
        fresh_buffer(s2.float().reshape(1, -1)),
    )


def arm_fused_mlp(model: torch.nn.Module | None) -> int:
    """Arm every eligible draft MLP of `model`; returns how many were armed."""
    if model is None:
        return 0
    mlps = [m for m in model.modules() if isinstance(m, MiniMaxM3DSparkMLP)]
    armed = 0
    for mlp in mlps:
        operands = _operands(mlp)
        if operands is None:
            continue
        for name, tensor in zip(_BUFFERS, operands):
            mlp.register_buffer(name, tensor, persistent=False)
        armed += 1
    if mlps:
        logger.info(
            "DSpark draft: %d of %d MLPs run as one kernel at up to %d rows.",
            armed,
            len(mlps),
            kernel.MAX_ROWS,
        )
    return armed
