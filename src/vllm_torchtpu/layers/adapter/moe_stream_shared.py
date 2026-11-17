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
"""Compute a MoE layer's dense shared expert inside its routed MoE op.

vLLM's MoE runner runs the shared expert as its own MLP (two weight-bound
matmuls per decode step on this rank's TP slice) and adds it to the routed
output before the all-reduce. With ``TPU_MOE_STREAM_SHARED_EXPERT=1`` an
eligible NVFP4 layer whose shared expert is an e4m3 SwiGLU-OAI MLP (MiniMax M3)
hands that slice to the GMM MoE op instead: decode calls served by the streamed
kernel (``TPU_GMM_V2_DECODE_STREAM``) stream its weights in the same pipeline
as the routed experts' and compute it under their DMAs; every other call
computes it with XLA dots in the same op. The sum is added in bf16 as the
runner adds it, and the runner's all-reduce is unchanged.

Arming runs once after the model is loaded, when the shared linears have their
final e4m3 per-channel layout. The runner is told it has no shared expert, so
vLLM issues no shared MLP. Every rank arms the same layers (the decision reads
only shapes and config, identical across TP ranks).
"""

from typing import Any

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_BUFFERS = (
    "_tpu_stream_shared_w1",
    "_tpu_stream_shared_s1",
    "_tpu_stream_shared_w2",
    "_tpu_stream_shared_s2",
)
# The TPU MoE kernels' SwiGLU-OAI (limit, alpha, beta).
_SWIGLU = (7.0, 1.702, 1.0)


def _shared_slice(runner) -> tuple | None:
    """(w13 e4m3 [H, 2Fs], scale [2Fs], w2 e4m3 [Fs, H], scale [H]) of the
    runner's shared expert, or None if it is not an e4m3 SwiGLU-OAI MLP."""
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

    mlp = getattr(getattr(runner, "_shared_experts", None), "_layer", None)
    if mlp is None:
        return None
    act = getattr(mlp, "act_fn", None)
    gate_up = getattr(mlp, "gate_up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if not isinstance(act, SiluAndMulWithClamp) or gate_up is None or down is None:
        return None
    if (act.swiglu_limit, act.alpha, act.beta) != _SWIGLU:
        return None
    if getattr(mlp, "expert_gate", None) is not None:
        return None
    tensors = []
    for lin in (gate_up, down):
        w = getattr(lin, "weight", None)
        s = getattr(lin, "weight_scale", None)
        if (
            w is None
            or s is None
            or w.dtype != torch.float8_e4m3fn
            or w.dim() != 2
            or s.dim() != 1
            or s.shape[0] != w.shape[1]
            or not getattr(lin, WEIGHT_FLIPPED_ATTR, False)
            or getattr(lin, "bias", None) is not None
        ):
            return None
        # The parameters themselves: a view passed to the op is
        # re-materialized by an eager program on every call.
        tensors += [w.data, s.data]
    w1, s1, w2, s2 = tensors
    hidden, n1 = w1.shape
    fs, n2 = w2.shape
    if n1 != 2 * fs or n2 != hidden or fs % 128:
        return None
    if getattr(down, "reduce_results", False):
        return None
    return w1, s1, w2, s2


def _admit(runner) -> tuple | None:
    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod

    qm = runner._quant_method
    cfg = runner.moe_config
    if not isinstance(qm, VllmNvfp4MoEMethod):
        return None
    if getattr(qm, "_use_adaptive_fused_moe", False) or fused_moe_ep_supported(qm):
        return None
    if enable_pipelined_collective_and_compute():
        return None
    if getattr(qm, "_tpu_activation_str", None) != "swigluoai":
        return None
    if getattr(cfg, "is_sequence_parallel", False):
        return None
    if (
        runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
        or runner.shared_expert_gate is not None
    ):
        return None
    # The runner would scale the op's shared + routed sum.
    if float(getattr(runner, "routed_scaling_factor", 1.0)) != 1.0:
        return None
    return _shared_slice(runner)


def arm_stream_shared(model: torch.nn.Module) -> int:
    """Arm every eligible MoE layer of `model`; returns how many were armed."""
    if not envs.TPU_MOE_STREAM_SHARED_EXPERT:
        return 0
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm_torchtpu.layers.adapter.fused_moe import prebuild_fused_moe_kernel

    runners = [m for m in model.modules() if isinstance(m, MoERunner)]
    armed = 0
    for runner in runners:
        shared = _admit(runner)
        if shared is None:
            continue
        layer = runner.routed_experts
        for name, tensor in zip(_BUFFERS, shared):
            layer.register_buffer(name, tensor, persistent=False)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=runner._quant_method._tpu_activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
            with_shared=True,
        )
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        armed += 1
    logger.info(
        "TPU_MOE_STREAM_SHARED_EXPERT=1: shared expert computed inside the MoE "
        "op on %d of %d MoE layers",
        armed,
        len(runners),
    )
    return armed


def arm_runner_model(model_runner: Any) -> int:
    return arm_stream_shared(model_runner.model)
