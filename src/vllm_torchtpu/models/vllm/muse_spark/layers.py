# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Muse Spark RMSNorms for TPU, plus reference and layout helpers.

Upstream source: ``vllm/models/muse_spark/common/layers.py``.

TPU changes:

- ``MuseRMSNorm`` is ``MuseSpark1xRMSNorm`` with an FP32 weight, and always
  runs its ``forward_native`` instead of dispatching to the Triton kernel.
- ``MuseRMSNormWithInputScale`` is the weight-first variant
  (``MuseSpark1xRMSNormWithInputScale``); it normalizes with the same explicit
  FP32 mean square as ``forward_native`` instead of ``F.rms_norm``.
- ``selected_expert_outputs`` and ``attention_layout`` have no upstream
  counterpart.
"""

import torch
from torch import nn
from torch.nn import functional as F
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNorm


class MuseRMSNorm(MuseSpark1xRMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        has_weight: bool = True,
        zero_centered: bool = False,
    ):
        super().__init__(
            hidden_size, eps, has_weight=has_weight, zero_centered_gamma=zero_centered
        )
        if has_weight:
            self.weight = nn.Parameter(
                torch.empty(hidden_size, dtype=torch.float32), requires_grad=False
            )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual, out_dtype)


class MuseRMSNormWithInputScale(MuseRMSNorm):
    """Weight-first RMSNorm: ``rms_norm(x * weight)``."""

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert residual is None and self.weight is not None
        weight = self.weight.float()
        if self.zero_centered_gamma:
            weight = weight + self.gain_center
        y = x.float() * weight
        y = y * torch.rsqrt(
            y.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return y.to(out_dtype or x.dtype)


def selected_expert_outputs(
    x: torch.Tensor, ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """Reference BF16 expert outputs [tokens, top_k, hidden], before normalization."""
    tokens, top_k = ids.shape
    flat_ids = ids.reshape(-1)
    inputs = x[:, None, :].expand(-1, top_k, -1).reshape(-1, 1, x.shape[-1])
    gate, up = torch.bmm(inputs, w13.index_select(0, flat_ids)).chunk(2, dim=-1)
    activated = (F.silu(gate.float()) * up.float()).to(x.dtype)
    rows = torch.bmm(activated, w2.index_select(0, flat_ids))
    return rows.reshape(tokens, top_k, x.shape[-1])


def attention_layout(config, layer_idx: int) -> tuple[bool, int | None]:
    use_rope = config.layer_rope_theta[layer_idx] != 0
    window = (
        config.sliding_window
        if config.layer_types[layer_idx] == "sliding_attention"
        else None
    )
    return use_rope, window
