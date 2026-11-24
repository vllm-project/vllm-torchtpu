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
"""TPU_MINIMAX_M3_NORM_WEIGHT_HOIST: the Gemma RMSNorms' ``1 + w`` weights in
FP32, computed once after loading instead of in every forward.

`GemmaRMSNorm.forward_native` computes ``self.weight.float() + 1.0`` per
call, and `attention.norm_rope_heads` builds its per-head ``[heads, D]``
FP32 table of the four q/k norms per call. Each is a separate small XLA
fusion per layer in the decode step (0.6-1.0 us each at c48 T=96, ~150 us per
step, j-224a8aae). Both depend only on the weights, so they are computed here
once into non-persistent buffers (the same FP32 values: the same expression
on the same weights) and the forwards read the buffers.
"""

import types

import torch
from torch import nn
from vllm import ir
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_WEIGHT = "_hoisted_weight_f32"
QK_TABLE = "_hoisted_qk_norm_table"


def gemma_weight(norm: GemmaRMSNorm) -> torch.Tensor:
    """``norm.weight.float() + 1.0``, from the hoisted buffer when present."""
    hoisted = getattr(norm, _WEIGHT, None)
    if hoisted is not None:
        return hoisted
    return norm.weight.float() + 1.0


def _forward_hoisted(self, x, residual=None):
    """`GemmaRMSNorm.forward_native` with the hoisted weight."""
    weight = getattr(self, _WEIGHT)
    if residual is None:
        return ir.ops.rms_norm(x, weight, self.variance_epsilon)
    return ir.ops.fused_add_rms_norm(x, residual, weight, self.variance_epsilon)


def qk_norm_table(
    norms: tuple[GemmaRMSNorm, GemmaRMSNorm, GemmaRMSNorm, GemmaRMSNorm],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """`attention.norm_rope_heads`' per-head ``1 + w`` table, FP32
    ``[num_heads + 2 * num_kv_heads + 1, head_dim]``."""
    q_norm, k_norm, index_q_norm, index_k_norm = norms
    return (
        torch.cat(
            [
                q_norm.weight.expand(num_heads, head_dim),
                k_norm.weight.expand(num_kv_heads, head_dim),
                index_q_norm.weight.expand(num_kv_heads, head_dim),
                index_k_norm.weight.expand(1, head_dim),
            ]
        ).float()
        + 1.0
    )


@torch.no_grad()
def hoist(root: nn.Module) -> int:
    """Hoist every GemmaRMSNorm weight and every MiniMax M3 attention's q/k
    norm table under `root`; returns the number of norms hoisted."""
    from vllm_torchtpu.models.minimax_m3.attention import MiniMaxM3SparseAttention

    count = 0
    for module in root.modules():
        if isinstance(module, GemmaRMSNorm) and type(module).forward_native is (
            GemmaRMSNorm.forward_native
        ):
            module.register_buffer(
                _WEIGHT, module.weight.detach().float() + 1.0, persistent=False
            )
            module._forward_method = types.MethodType(_forward_hoisted, module)
            count += 1
        elif isinstance(module, MiniMaxM3SparseAttention):
            norms = (module.q_norm, module.k_norm, module.index_q_norm, module.index_k_norm)
            module.register_buffer(
                QK_TABLE,
                qk_norm_table(
                    norms, module.num_heads, module.num_kv_heads, module.head_dim
                ).detach(),
                persistent=False,
            )
    logger.warning(
        "TPU_MINIMAX_M3_NORM_WEIGHT_HOIST=1 on: %d Gemma RMSNorm weights and "
        "the q/k norm tables computed once after loading.",
        count,
    )
    return count
