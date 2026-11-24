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
"""3-step execution plan for Qwen4Exp attention.

One attention layer runs three stages in sequence:
    Step 1: QKVProj  -> produces (q, kv) in ``qkv_layout``
    Step 2: CoreAttn -> takes ``qkv_layout``, produces ``o`` in ``o_layout``
    Step 3: OutProj  -> takes ``o_layout``, produces ``hidden_states``

``AttentionPlan.resolve`` picks one implementation for each stage once at
``__init__`` so that:
    1. ``qkv_proj.output_layouts(desc)`` intersects ``core_attn.input_layouts(desc)``
    2. ``core_attn.output_layouts(desc)`` intersects ``out_proj.input_layouts(desc)``
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass

import torch
from vllm.config import CacheConfig, ModelConfig
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.layers.adapter.attention import TPU_STR_DTYPE_TO_TORCH_DTYPE
from vllm_torchtpu.models.qwen4_exp.layout import (
    QKVLayout,
    TensorLayout,
    unfused_o,
    unfused_qkv,
)
from vllm_torchtpu.models.qwen4_exp.qsa import qsa_input_layouts, qsa_output_layouts


@dataclass(frozen=True)
class AttentionDesc:
    """Static facts resolved once from vLLM configs."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    kv_layout: KVCacheLayout
    sparse: bool

    @classmethod
    def from_configs(
        cls,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sparse: bool,
        model_config: ModelConfig,
        cache_config: CacheConfig | None,
    ) -> AttentionDesc:
        dtype = model_config.dtype
        assert isinstance(dtype, torch.dtype)
        raw_kv_dtype = (
            cache_config.cache_dtype
            if cache_config is not None and cache_config.cache_dtype
            else "auto"
        )
        kv_dtype = (
            dtype
            if raw_kv_dtype == "auto"
            else TPU_STR_DTYPE_TO_TORCH_DTYPE[raw_kv_dtype.lower().strip()]
        )
        kv_layout = (
            cache_config.get_resolved_kv_cache_layout()
            if cache_config is not None
            else KVCacheLayout.LBNHC
        )
        return cls(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            kv_cache_dtype=kv_dtype,
            kv_layout=kv_layout,
            sparse=sparse,
        )


class QKVProj(enum.StrEnum):
    """Step 1: hidden_states -> (q, kv, gate)."""

    UNFUSED = "unfused"
    FUSED = "fused"

    def output_layouts(self, desc: AttentionDesc) -> tuple[QKVLayout, ...]:
        match self:
            case QKVProj.UNFUSED:
                return (unfused_qkv(desc.dtype),)
            case QKVProj.FUSED:
                # Wired in a follow-up PR alongside tokamax fused_qkv.
                return ()


class CoreAttn(enum.StrEnum):
    """Step 2: (q, kv) -> o."""

    RPA_V3 = "rpa_v3"
    BATCHED_RPA = "batched_rpa"
    QSA = "qsa"

    def input_layouts(self, desc: AttentionDesc) -> tuple[QKVLayout, ...]:
        match self:
            case CoreAttn.RPA_V3:
                return () if desc.sparse else (unfused_qkv(desc.dtype),)
            case CoreAttn.BATCHED_RPA:
                # Wired in a follow-up PR alongside ops/dense_attn.py.
                return ()
            case CoreAttn.QSA:
                return (
                    qsa_input_layouts(desc.dtype, desc.kv_layout) if desc.sparse else ()
                )

    def output_layouts(self, desc: AttentionDesc) -> tuple[TensorLayout, ...]:
        match self:
            case CoreAttn.RPA_V3:
                return () if desc.sparse else (unfused_o(desc.dtype),)
            case CoreAttn.BATCHED_RPA:
                return ()
            case CoreAttn.QSA:
                return (
                    qsa_output_layouts(desc.dtype, desc.kv_layout)
                    if desc.sparse
                    else ()
                )


class OutProj(enum.StrEnum):
    """Step 3: (o, gate) -> hidden_states."""

    UNFUSED = "unfused"
    FUSED = "fused"

    def input_layouts(self, desc: AttentionDesc) -> tuple[TensorLayout, ...]:
        match self:
            case OutProj.UNFUSED:
                return (unfused_o(desc.dtype),)
            case OutProj.FUSED:
                # Wired in a follow-up PR alongside tokamax fused_out_proj.
                return ()


@dataclass(frozen=True)
class AttentionPlan:
    """Resolved 3-step plan; ``Qwen4ExpAttention.forward`` only matches on this."""

    qkv_proj: QKVProj
    core_attn: CoreAttn
    out_proj: OutProj
    qkv_layout: QKVLayout
    o_layout: TensorLayout

    @classmethod
    def resolve(
        cls,
        desc: AttentionDesc,
        *,
        qkv_proj: QKVProj | None = None,
        core_attn: CoreAttn | None = None,
        out_proj: OutProj | None = None,
    ) -> AttentionPlan:
        """Selects one compatible (qkv_proj, core_attn, out_proj) triple."""
        cores: tuple[CoreAttn, ...] = (
            (core_attn,)
            if core_attn is not None
            else (CoreAttn.QSA,)
            if desc.sparse
            else (CoreAttn.BATCHED_RPA, CoreAttn.RPA_V3)
        )
        qkv_projs = (
            (qkv_proj,) if qkv_proj is not None else (QKVProj.FUSED, QKVProj.UNFUSED)
        )
        out_projs = (
            (out_proj,) if out_proj is not None else (OutProj.FUSED, OutProj.UNFUSED)
        )

        for core, qkv, out in itertools.product(cores, qkv_projs, out_projs):
            shared_qkv = [
                lay
                for lay in core.input_layouts(desc)
                if lay in qkv.output_layouts(desc)
            ]
            shared_o = [
                lay
                for lay in core.output_layouts(desc)
                if lay in out.input_layouts(desc)
            ]
            if shared_qkv and shared_o:
                return cls(
                    qkv_proj=qkv,
                    core_attn=core,
                    out_proj=out,
                    qkv_layout=shared_qkv[0],
                    o_layout=shared_o[0],
                )

        raise ValueError(
            "No compatible Qwen4Exp attention plan "
            f"(qkv_proj={qkv_proj}, core_attn={core_attn}, out_proj={out_proj}, "
            f"sparse={desc.sparse}, kv_layout={desc.kv_layout}, dtype={desc.dtype})."
        )
