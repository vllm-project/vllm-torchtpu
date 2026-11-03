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
"""fp32-accurate MoE router for Beam on TPU.

Beam's checkpoint router is fp32 ``[E, H]``. Upstream ``GateLinear`` keeps it
fp32, and on TPU its plain ``F.linear`` path runs at TorchTPU's default fp32
matmul precision (one bf16 pass, fp32 accumulation), so the router effectively
uses ``bf16(W)`` and picks a different top-k on some tokens. ``BeamRouter``
replaces it: the fp32 weight is stored as three bf16 parts, and one stacked
matmul at one-pass precision gives the fp32 logits to fp32 accuracy
(upstream's GPU "bf16x3" router tier does the same split).
"""

import torch
from torch import nn
from vllm.model_executor.utils import set_weight_attrs

# Three bf16 parts carry the 24-bit fp32 mantissa (3 x 8 bits).
ROUTER_SPLIT_TERMS = 3


def split_weight_bf16(weight: torch.Tensor) -> torch.Tensor:
    """Splits ``weight`` into ``ROUTER_SPLIT_TERMS`` stacked bf16 parts.

    Returns ``[ROUTER_SPLIT_TERMS, *weight.shape]``. Each part is the bf16
    rounding of what the previous parts leave, computed in float64, so the
    parts sum to ``weight`` to fp32 accuracy.
    """
    parts: list[torch.Tensor] = []
    remainder = weight.double()
    for _ in range(ROUTER_SPLIT_TERMS):
        part = remainder.to(torch.bfloat16)
        parts.append(part)
        remainder = remainder - part.double()
    return torch.stack(parts)


def router_logits_split(x: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
    """fp32-accurate router logits from ``[terms, E, H]`` bf16 weight parts.

    The input must be bf16, like Beam's pre-FFN norm output. Every bf16 x bf16
    product is exact in fp32, so one fp32 matmul loses nothing to the default
    one-pass TPU precision, and summing the per-term slices in fp32 recovers
    fp32 GEMM precision. A single matmul over the stacked parts keeps XLA from
    re-associating separate matmuls, and needs no precision context, which
    would break the fullgraph trace.
    """
    assert x.dtype == torch.bfloat16, f"split router needs bf16 input, got {x.dtype}"
    stacked_logits = torch.nn.functional.linear(x.float(), parts.flatten(0, 1).float())
    return stacked_logits.unflatten(-1, parts.shape[:2]).sum(dim=-2)


# TODO(parfann): move to layers/adapter as a GateLinear subclass for any fp32
# router on TPU.
class BeamRouter(nn.Module):
    """fp32-accurate stand-in for ``GateLinear`` as the MoE runner's gate.

    Replicated on every rank, like ``GateLinear``, so it works under any
    TP / EP / DP layout.
    """

    # The router logits dtype the MoE runner is built with.
    out_dtype = torch.float32

    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                ROUTER_SPLIT_TERMS, num_experts, hidden_size, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Splits the checkpoint's fp32 ``[E, H]`` weight into the bf16 parts.

        The split runs on the host, so no fp32 copy reaches the device, and a
        weight reload replays it.
        """
        expected_shape = param.shape[1:]
        if (
            loaded_weight.dtype != torch.float32
            or loaded_weight.shape != expected_shape
        ):
            raise ValueError(
                "The Beam router expects the checkpoint's fp32 mlp.gate.weight of "
                f"shape {list(expected_shape)}, got {loaded_weight.dtype} "
                f"{list(loaded_weight.shape)}; load a checkpoint that keeps the "
                "router in fp32."
            )
        param.data.copy_(split_weight_bf16(loaded_weight.cpu()))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # The MoE runner calls its gate and unpacks (logits, bias).
        return router_logits_split(x, self.weight), None

    def extra_repr(self) -> str:
        terms, num_experts, hidden_size = self.weight.shape
        return f"hidden_size={hidden_size}, num_experts={num_experts}, terms={terms}"
