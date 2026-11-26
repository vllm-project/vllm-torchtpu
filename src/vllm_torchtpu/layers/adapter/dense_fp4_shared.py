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
"""Hand a MoE layer's shared expert to the routed experts' op (from r15-merge).

vLLM's MoERunner computes the shared expert with its own linears and adds it to
the routed output before the all-reduce. With ``TPU_MOE_DENSE_FP4_SHARED=1``
an eligible layer (NVFP4 experts on the GMM path, EP or TP-sliced, a
SwiGLU-OAI shared expert on per-channel e4m3 MiniMax-M3 linears) passes the
shared expert's tensors to the routed op instead: the dense fp4 decode kernel
evaluates it in the same call, overlapping its weight DMA with the routed
experts', and at other token counts the op computes it with XLA
(``fused_moe_gmm.fused_moe_func``). The runner is told it has no shared
expert, so the all-reduce of the sum is unchanged.

Exactly one place runs an armed layer's shared expert on every step:

- the model (``model_side_shared``: TPU_MOE_SHARED_MODEL_SIDE_MIN_TOKENS on
  steps whose MoE reduction the model owns), else
- the routed op. On TPU_MINIMAX_M3_SHARED_IN_COMBINE's steps the op fuses
  the down projection into the TensorCore combine itself, and MiniMaxM3MoE
  does not take its own shared-in-combine path for an armed layer.

Every bucket so consumes the shared tensors through the same layer buffers
(``_dfs_*``) after the routed experts' weights, so the traced subgraph inputs
keep one order across buckets (shape_variants hand-over).

Arming happens once after load, when the shared linears hold their final
K-major e4m3 layout.
"""

from typing import Any

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Non-persistent buffers of the routed layer, in SharedExpert's order.
_SHARED_BUFFERS = ("_dfs_w1", "_dfs_s1", "_dfs_w2", "_dfs_s2")

# Attribute of a routed layer holding a step's gathered e4m3 row pieces for
# the shared expert, a tuple of (rows [T_p, H] e4m3, scales [T_p, 1] f32) in
# token order: MiniMaxM3MoE sets it around the routed call within one traced
# forward (it ends as None, so it adds no graph output) and the routed op
# reads it. The op takes the piece counts `shared_rows_piece_counts` lists
# (each count is its own prebuilt op variant).
SHARED_ROWS_ATTR = "_dfs_step_shared_rows"
# r15-merge's two-piece producer (SP row pieces, TPU_MINIMAX_M3_SP_FFN_PIECES=2)
# is ported separately; the op variant for it is prebuilt here once that
# producer exists (see shared_rows_piece_counts).
SHARED_ROWS_PIECES = 2


def shared_rows_piece_counts() -> tuple[int, ...]:
    """Row-piece counts the armed op is prebuilt for: 0 (bf16 tokens), and 1
    (an SP step's whole e4m3 row block) with TPU_MINIMAX_M3_SP_E4M3_SHARED.
    The SP row-pieces port adds SHARED_ROWS_PIECES when its producer hands
    the routed op TPU_MINIMAX_M3_SP_FFN_PIECES=2 pieces."""
    counts = (0,)
    if envs.TPU_MINIMAX_M3_SP_E4M3_SHARED:
        counts += (1,)
    if (
        envs.TPU_MINIMAX_M3_SP_E4M3_SHARED
        and getattr(envs, "TPU_MINIMAX_M3_SP_FFN_PIECES", 1) == SHARED_ROWS_PIECES
    ):
        counts += (SHARED_ROWS_PIECES,)
    return counts


def shared_operands(layer: Any) -> tuple[torch.Tensor, ...] | None:
    """The armed shared expert's tensors of a routed layer, else None."""
    if getattr(layer, _SHARED_BUFFERS[0], None) is None:
        return None
    return tuple(getattr(layer, name) for name in _SHARED_BUFFERS)


def model_side_shared(layer: Any, num_tokens: int) -> bool:
    """Whether a step of ``num_tokens`` leaves the armed shared expert to the
    model (TPU_MOE_SHARED_MODEL_SIDE_MIN_TOKENS) instead of the routed op;
    the routed op and the model both ask, so exactly one of them runs it.
    Only while the model owns the MoE reduction (skip_final_all_reduce, set
    in the same trace for both): the model's partial shared expert is then
    reduced with the routed partial."""
    min_tokens = envs.TPU_MOE_SHARED_MODEL_SIDE_MIN_TOKENS
    return (
        0 < min_tokens <= num_tokens
        and bool(getattr(layer.moe_config, "skip_final_all_reduce", False))
        and shared_operands(layer) is not None
    )


def _shared_tensors(runner) -> tuple[torch.Tensor, ...] | None:
    """The shared expert's (w1, s1, w2, s2) in SharedExpert's layout, or None
    when it is not a bias-free SwiGLU-OAI MLP on e4m3 per-channel linears."""
    from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, e4m3_mlp_operands

    mlp = getattr(getattr(runner, "_shared_experts", None), "_layer", None)
    if type(mlp) is not MiniMaxM3MLP:
        return None
    return e4m3_mlp_operands(mlp)


def _admit(runner) -> tuple[torch.Tensor, ...] | None:
    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod

    qm = runner._quant_method
    cfg = runner.moe_config
    if (
        not isinstance(qm, VllmNvfp4MoEMethod)
        or qm._tpu_activation_str != "swigluoai"
        or qm._use_adaptive_fused_moe
        or fused_moe_ep_supported(qm)
        or enable_pipelined_collective_and_compute()
        or getattr(cfg, "is_sequence_parallel", False)
        or runner.routed_scaling_factor != 1.0
        or runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
        or runner.shared_expert_gate is not None
    ):
        return None
    return _shared_tensors(runner)


def arm_dense_fp4_shared(model: torch.nn.Module, what: str = "model") -> int:
    """Arm every eligible MoE layer of `model`; returns how many were armed."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm_torchtpu.layers.adapter.fused_moe import prebuild_fused_moe_kernel

    runners = [m for m in model.modules() if isinstance(m, MoERunner)]
    armed = 0
    for runner in runners:
        shared = _admit(runner)
        if shared is None:
            continue
        layer = runner.routed_experts
        # A traced forward cannot build the op. TP-sliced experts (no EP) hold
        # every expert's slice; the op then runs its non-EP routing.
        use_ep = getattr(layer, "_experts_start", None) is not None
        for pieces in shared_rows_piece_counts():
            prebuild_fused_moe_kernel(
                topk=runner.moe_config.experts_per_token,
                activation=runner._quant_method._tpu_activation_str,
                use_ep=use_ep,
                with_shared=True,
                with_shared_rows=pieces,
            )
        for name, tensor in zip(_SHARED_BUFFERS, shared):
            layer.register_buffer(name, tensor, persistent=False)
        setattr(layer, SHARED_ROWS_ATTR, None)
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        armed += 1
    logger.info(
        "Dense fp4 MoE carries the shared expert on %d of %d MoE layers of the %s",
        armed,
        len(runners),
        what,
    )
    return armed
