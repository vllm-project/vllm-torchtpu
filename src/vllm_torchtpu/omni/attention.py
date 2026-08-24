# SPDX-License-Identifier: Apache-2.0
"""PyTorch SDPA diffusion attention backend for TPU execution."""

from typing import Any

import torch
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl


class TpuSDPAImpl(SDPAImpl):
    """TPU implementation of SDPA attention for diffusion models."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query,
                                  key,
                                  value,
                                  attn_metadata,
                                  mask_mode="broadcast_k")


class TpuSDPABackend(SDPABackend):
    """TPU attention backend for diffusion models using PyTorch SDPA."""

    @staticmethod
    def get_name() -> str:
        return "TPU_SDPA"

    @staticmethod
    def get_impl_cls() -> type[TpuSDPAImpl]:
        return TpuSDPAImpl
