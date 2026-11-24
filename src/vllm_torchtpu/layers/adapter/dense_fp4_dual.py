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
"""Dual expert layout: a TP-sliced copy of an EP layer's routed experts.

Under EP a rank streams its own touched experts whole, so a small decode step
waits for the rank that owns the most touched experts. With
``TPU_MOE_DUAL_LAYOUT_MAX_TOKENS > 0`` every eligible layer (NVFP4 experts on
the EP GMM path, linear placement over the whole EP group) also holds rank r's
slice r of every expert's intermediate dim, built once after load by an
all-to-all of the EP weights; steps up to the cap run the dense fp4 kernel on
the copy, where every rank streams the same share
(``fused_moe_gmm.fused_moe_func``). It costs the routed experts' HBM a second
time.
"""

import functools
from typing import Any

import jax
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Non-persistent buffers of the routed layer, in TpSlicedExperts' order.
_DUAL_BUFFERS = ("_dual_w1", "_dual_w2", "_dual_s1", "_dual_s2")


def dual_operands(layer: Any) -> tuple[torch.Tensor, ...] | None:
    """The armed TP-sliced experts of a routed layer, else None."""
    if getattr(layer, _DUAL_BUFFERS[0], None) is None:
        return None
    return tuple(getattr(layer, name) for name in _DUAL_BUFFERS)


def _admit(runner, ep_group) -> bool:
    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod

    qm = runner._quant_method
    layer = runner.routed_experts
    cfg = layer.moe_config
    if (
        not isinstance(qm, VllmNvfp4MoEMethod)
        or qm._use_adaptive_fused_moe
        or fused_moe_ep_supported(qm)
        or enable_pipelined_collective_and_compute()
        or getattr(layer, "_experts_start", None) is None
        or getattr(cfg, "is_sequence_parallel", False)
    ):
        return False
    r = ep_group.world_size
    w1, w2 = layer.w13_weight, layer.w2_weight
    s1, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    if s1.dim() != 4 or s2.dim() != 4:
        return False
    # float4_e2m1fn_x2 tensors count pairs on their last axis.
    e, _, _, two_i = s1.shape
    i, h = two_i // 2, s2.shape[-1]
    # Linear placement over the whole group: rank r holds experts [r e, r e + e).
    return (
        w1.dtype == torch.float4_e2m1fn_x2
        and w2.dtype == torch.float4_e2m1fn_x2
        and cfg.ep_size == r
        and cfg.ep_rank == ep_group.rank_in_group
        and layer.global_num_experts == r * e
        and i % (r * 128) == 0
        and h % 128 == 0
        and tuple(w1.shape) == (e, h, i)
        and tuple(w2.shape) == (e, i, h // 2)
        and tuple(s1.shape) == (e, h // 128, 1, two_i)
        and tuple(s2.shape) == (e, i // 128, 1, h)
    )


@functools.cache
def _ops(num_ranks: int):
    from torch_tpu._internal import pallas

    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        tp_slice_sends,
        tp_slices_from_received,
    )

    sends = pallas.jax_op(
        f"pallas::moe_dual_layout_sends_r{num_ranks}",
        functools.partial(tp_slice_sends, num_ranks=num_ranks),
    )

    def received_fn(
        w1: jax.Array, w2: jax.Array, s1: jax.Array, s2: jax.Array
    ) -> tuple[jax.Array, ...]:
        return tuple(tp_slices_from_received(w1, w2, s1, s2))

    received = pallas.jax_op("pallas::moe_dual_layout_received", received_fn)
    return sends, received


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _exchange(
    w1: torch.Tensor, w2: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Block d of each leading axis goes to EP rank d; block r of each result
    came from EP rank r."""
    from torch.distributed import _functional_collectives as funcol
    from vllm.distributed.parallel_state import get_ep_group

    group = get_ep_group().device_group
    outs = []
    for x in (w1, w2, s1, s2):
        out = funcol.all_to_all_single(x, None, None, group)
        if isinstance(out, funcol.AsyncCollectiveTensor):
            out = funcol.wait_tensor(out)
        outs.append(out)
    return tuple(outs)


def _memory_allocated() -> int | None:
    try:
        return int(torch.accelerator.memory_allocated())
    except Exception:  # noqa: BLE001 - a log line, not a requirement
        return None


def arm_dense_fp4_dual(model: torch.nn.Module, what: str = "model") -> int:
    """Build the TP-sliced copy on every eligible MoE layer of `model`, a
    collective every EP rank must reach; returns how many were armed."""
    from vllm.distributed.parallel_state import get_ep_group
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm_torchtpu.layers.adapter.dense_fp4_shared import shared_operands
    from vllm_torchtpu.layers.adapter.fused_moe import prebuild_fused_moe_kernel
    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import _DENSE_ROUTE_ATTR

    ep_group = get_ep_group()
    runners = [m for m in model.modules() if isinstance(m, MoERunner)]
    before = _memory_allocated()
    armed = copy_bytes = 0
    for runner in runners:
        # _exchange is a collective: admission must agree on every EP rank, as
        # its config, flags and shapes do.
        if not _admit(runner, ep_group):
            continue
        layer = runner.routed_experts
        if (
            envs.TPU_MOE_SHARED_MODEL_SIDE_MIN_TOKENS > 0
            and shared_operands(layer) is not None
        ):
            # Steps the model's shared expert serves (model_side_shared) read
            # the shared and the dual weights in another order than steps the
            # routed op serves, and vLLM's warmup cannot hand compiled buckets
            # between the two traces (ShapeSpecializationError).
            raise ValueError(
                "The dual expert layout (TPU_MOE_DUAL_LAYOUT_MAX_TOKENS) needs "
                "the shared expert in the routed op: set "
                "TPU_MOE_SHARED_MODEL_SIDE_MIN_TOKENS=0."
            )
        sends_op, received_op = _ops(ep_group.world_size)
        sends = sends_op(
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
        copy = received_op(*_exchange(*sends))
        del sends
        # Steps the dense fp4 kernel routes itself pass the layer's route.
        layer_route = getattr(layer, _DENSE_ROUTE_ATTR, None)
        # A fixed order: op names come from a counter, and must match on
        # every rank.
        for route in (None,) if layer_route is None else (None, layer_route):
            prebuild_fused_moe_kernel(
                topk=layer.moe_config.experts_per_token,
                activation=runner._quant_method._tpu_activation_str,
                use_ep=True,
                with_shared=shared_operands(layer) is not None,
                with_dual=True,
                route=route,
            )
        for name, tensor in zip(_DUAL_BUFFERS, copy):
            layer.register_buffer(name, tensor, persistent=False)
            copy_bytes += tensor.numel() * tensor.element_size()
        armed += 1
    after = _memory_allocated()
    logger.info(
        "Dual expert layout: TP-sliced copies on %d of %d MoE layers of the %s, "
        "%.3f GB of tensors; device memory allocated %s -> %s",
        armed,
        len(runners),
        what,
        copy_bytes / 1e9,
        before,
        after,
    )
    return armed
