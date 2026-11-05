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
"""Qwen3.5's pool slot facts: `Qwen35.pool_layout` against the framework's
layout of the Qwen3.5 GDN slot. The model-free checks of the slot facts and
their comparison are in `tests/megakernel_runtime/test_pool_layout_facts.py`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_torchtpu import gdn_pool_layout as gpl
from vllm_torchtpu.megakernel.runtime import (
    PoolLayoutFacts,
    compare_layout_facts,
    facts_from_pool_layout,
)
from vllm_torchtpu.megakernel_runtime.pool_facts import framework_pool_layout_facts
from vllm_torchtpu.models.qwen3_5.megakernel.model import Qwen35

TP = 8


def _mk_facts() -> PoolLayoutFacts:
    return facts_from_pool_layout(Qwen35.pool_layout(R=TP), TP)


def _qwen35_gdn_layer() -> Any:
    """The attributes of the TP=8 Qwen3.5 GDN layer the framework lays out
    its pool slot from (`VllmGatedDeltaNetAttention`)."""
    return SimpleNamespace(
        get_state_dtype=lambda: (torch.bfloat16, torch.bfloat16),
        tp_head_geometry=SimpleNamespace(local_num_kq_heads=16 // TP),
        num_v_heads=64,
        tp_size=TP,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
    )


def test_qwen35_pool_layout_slot_facts_tp8() -> None:
    assert _mk_facts() == PoolLayoutFacts(
        page_dtype="float8_e4m3fn",
        page_shape=(2, 64, 4, 128),
        page_size=128,
        token_bytes=512,
        slot_tokens=640,
        ssm_dtype="bfloat16",
        ssm_bytes=8 * 128 * 128 * 2,
        ssm_offset_bytes=0,
        conv_dtype="bfloat16",
        conv_bytes=3 * 1536 * 2,
        conv_offset_bytes=4 * 65536,
        conv_order="taps_qkv",
        local_kh=2,
        local_vh=8,
        d_k=128,
        d_v=128,
        conv_taps=3,
        conv_dim=1536,
    )
    layout = Qwen35.pool_layout(R=TP)
    assert (layout.num_pools, layout.slots_per_pool) == (15, 3)


def test_qwen35_framework_slot_matches_its_pool_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The framework's slot for the Qwen3.5 GDN layer (bf16 SSM, bf16 conv
    pool, 640-token block) is exactly what `Qwen35.pool_layout` expects."""
    monkeypatch.setattr(gpl, "DEFAULT_POOLED_GDN_CONV_STATE_DTYPE", "torch.bfloat16")
    pool = torch.empty((96, 2, 64, 4, 128), dtype=torch.float8_e4m3fn, device="meta")
    fw = framework_pool_layout_facts(
        pool=pool,
        linear_attn=_qwen35_gdn_layer(),
        block_size=640,
        conv_qk_pair=False,
    )
    assert fw == _mk_facts()
    assert compare_layout_facts(_mk_facts(), fw).eligible
