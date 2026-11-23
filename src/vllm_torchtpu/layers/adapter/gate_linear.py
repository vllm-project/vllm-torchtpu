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
"""fp32-accurate MoE gate for fp32 router weights on TPU.

Upstream ``GateLinear`` has no TPU tier for an fp32 router weight and falls
back to ``F.linear``, which TorchTPU runs as one bf16 pass, so the router
effectively uses ``bf16(W)`` and flips some top-k picks. ``TpuGateLinear``
adds upstream's ROCm bf16x3 design: the weight is split once after loading
into three bf16 terms that sum back to it exactly.
"""

from typing import cast

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_rocm import (
    split_bf16x3,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

# The non-persistent buffer upstream GateLinear registers for the split; AOT
# export lays it out as a weight.
BF16X3_WEIGHT_BUFFER = "_bf16x3_weight"


def _bf16x3_router_logits(x: torch.Tensor, weight_split: torch.Tensor) -> torch.Tensor:
    """fp32 ``x @ W.T`` from bf16 ``x`` and ``W``'s ``[3, E, H]`` split.

    bf16 x bf16 products are exact in fp32, so the one-pass matmul loses
    nothing, and the fp32 sum over terms restores fp32 precision. One stacked
    matmul needs no precision context, which would break a fullgraph trace.
    The terms are summed as column slices: XLA rewrites ``unflatten().sum()``
    into a windowed convolution whose expert-minor output layout slows the
    grouped top-k sort after the gate.
    """
    assert x.dtype == weight_split.dtype == torch.bfloat16
    # fp32 operands make F.linear return fp32; TorchTPU still runs one bf16 pass.
    stacked_logits = torch.nn.functional.linear(
        x.float(), weight_split.flatten(0, 1).float()
    )
    hi, mid, lo = stacked_logits.split(weight_split.shape[1], dim=-1)
    return hi + mid + lo


class _TpuGateLinearMethod(UnquantizedLinearMethod):
    """Builds the split after loading, as upstream's ``_GateLinearMethod``."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        gate = cast("TpuGateLinear", layer)
        # Decided here, once out_dtype is final; a reload decides again.
        gate.allow_tpu_bf16x3_router_gemm = False
        gate._bf16x3_weight = None
        # Split before the base hook, which may drop layer.weight on CPU.
        if gate.out_dtype == torch.float32:
            weight = cast(torch.Tensor, gate.weight).detach()
            # On the host: TPU flushes subnormals, so a device split of small
            # weights would not sum back to them exactly.
            host_weight = weight.cpu()
            # A weight that bf16 holds exactly (a bf16 checkpoint in an fp32
            # gate) is already exact in upstream's one pass; its split would
            # add only zero terms.
            if not torch.equal(host_weight, host_weight.bfloat16().float()):
                try:
                    split = split_bf16x3(host_weight)
                except ValueError as exc:
                    logger.warning_once(
                        "Disabling the TPU BF16x3 router GEMM for %s, which "
                        "routes with bf16(W) instead: %s",
                        gate.prefix,
                        exc,
                    )
                else:
                    gate._bf16x3_weight = split.to(weight.device)
                    gate.allow_tpu_bf16x3_router_gemm = True
                    if weight.device.type == "tpu":
                        # Materialize the upload now, so HBM readings taken
                        # before the first forward (KV cache sizing) count it.
                        synchronize_tensors([gate._bf16x3_weight])
        super().process_weights_after_loading(layer)


@GateLinear.register_oot
class TpuGateLinear(GateLinear):
    """``GateLinear`` with a bf16x3 tier for fp32 router weights.

    The weight stays fp32 ``[E, H]``; the split is a non-persistent
    ``[3, E, H]`` bf16 buffer.
    """

    allow_tpu_bf16x3_router_gemm: bool
    _bf16x3_weight: torch.Tensor | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.allow_tpu_bf16x3_router_gemm = False
        # Exactly the plain method: a TPU or quantized one may change the
        # weight's layout or dtype. Unlike upstream's tiers, VLLM_BATCH_INVARIANT
        # does not turn this one off: it never picks a kernel by token count.
        if self.bias is not None or self.weight.dtype != torch.float32:
            return
        if type(self.quant_method) is UnquantizedLinearMethod:
            # Safe: UnquantizedLinearMethod holds no per-layer state.
            # pyrefly: ignore[bad-assignment]
            self.quant_method = _TpuGateLinearMethod()
        elif self.out_dtype in (None, torch.float32):
            logger.warning_once(
                "TPU BF16x3 router GEMM is off for fp32 router gates using %s; "
                "they keep upstream GateLinear's precision.",
                type(self.quant_method).__name__,
            )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Python-static checks only, so this traces with fullgraph=True. fp32
        # activations keep upstream's path, and its one-pass precision.
        if self.allow_tpu_bf16x3_router_gemm and x.dtype == torch.bfloat16:
            weight_split = cast(torch.Tensor, self._bf16x3_weight)
            return self._return(_bf16x3_router_logits(x, weight_split))
        return super().forward(x)
