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
"""Bind-time recurrent pool slot facts: megakernel `PoolLayout` vs framework."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_torchtpu import gdn_pool_layout as gpl
from vllm_torchtpu.megakernel.models.qwen35 import Qwen35
from vllm_torchtpu.megakernel.runtime import (
    PoolLayoutFacts,
    compare_layout_facts,
    facts_from_pool_layout,
)
from vllm_torchtpu.megakernel_runtime.pool_facts import (
    framework_pool_layout_facts,
    runner_block_size,
)

TP = 8


def _mk_facts() -> PoolLayoutFacts:
    return facts_from_pool_layout(Qwen35.pool_layout(R=TP), TP)


def _gdn_layer(ssm_dtype: torch.dtype = torch.bfloat16) -> Any:
    """The attributes of the TP=8 Qwen3.5 GDN layer the framework lays out
    its pool slot from (`VllmGatedDeltaNetAttention`)."""
    return SimpleNamespace(
        get_state_dtype=lambda: (torch.bfloat16, ssm_dtype),
        tp_head_geometry=SimpleNamespace(local_num_kq_heads=16 // TP),
        num_v_heads=64,
        tp_size=TP,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
    )


def _pool() -> torch.Tensor:
    return torch.empty((96, 2, 64, 4, 128), dtype=torch.float8_e4m3fn, device="meta")


@pytest.fixture
def bf16_conv_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A framework whose pooled conv state is bf16 (`gdn_pool_layout`'s
    default decides this; tests pin it so they don't depend on the tree)."""
    monkeypatch.setattr(gpl, "DEFAULT_POOLED_GDN_CONV_STATE_DTYPE", "torch.bfloat16")


def _fw_facts(**kw: Any) -> PoolLayoutFacts:
    args: dict[str, Any] = dict(
        pool=_pool(), linear_attn=_gdn_layer(), block_size=640, conv_qk_pair=False
    )
    args.update(kw)
    return framework_pool_layout_facts(**args)


def test_megakernel_slot_facts_for_qwen35_tp8() -> None:
    f = _mk_facts()
    assert f == PoolLayoutFacts(
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
    assert len(f.fingerprint()) == 64
    assert f.fingerprint() == dataclasses.replace(f).fingerprint()
    assert f.fingerprint() != dataclasses.replace(f, conv_dtype="float32").fingerprint()


def test_framework_slot_facts_match_the_megakernel(bf16_conv_pool: None) -> None:
    """A framework layout with a bf16 SSM, a bf16 conv pool and a 640-token
    block is exactly what the megakernel's `PoolLayout` expects."""
    fw = _fw_facts()
    assert fw == _mk_facts()
    assert compare_layout_facts(_mk_facts(), fw).eligible


def test_framework_conv_fp32_is_a_named_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A framework that stores the pooled conv state as fp32 (same slot
    geometry, twice the conv bytes) is declined as a capability gap."""
    monkeypatch.setattr(gpl, "DEFAULT_POOLED_GDN_CONV_STATE_DTYPE", "torch.float32")
    fw = _fw_facts()
    assert (fw.conv_dtype, fw.conv_bytes) == ("float32", 3 * 1536 * 4)
    assert fw.slot_tokens == 640 and fw.conv_offset_bytes == 4 * 65536
    res = compare_layout_facts(_mk_facts(), fw)
    assert not res.eligible and res.code == "conv_state_dtype_unsupported"
    assert "conv_dtype: megakernel='bfloat16' framework='float32'" in res.message
    assert "conv_bytes" in res.message


def test_framework_fp32_ssm_slot_is_declined(bf16_conv_pool: None) -> None:
    """An fp32 SSM doubles the SSM region; the framework then needs a
    1152-token (9-page) slot while the megakernel addresses 5-page slots."""
    fw = _fw_facts(linear_attn=_gdn_layer(torch.float32), block_size=1152)
    assert fw.ssm_bytes == 2 * _mk_facts().ssm_bytes
    assert fw.conv_offset_bytes == 2 * _mk_facts().conv_offset_bytes
    res = compare_layout_facts(_mk_facts(), fw)
    assert not res.eligible and res.code == "pool_layout_mismatch"
    for field in ("slot_tokens", "ssm_dtype", "ssm_bytes", "conv_offset_bytes"):
        assert f"{field}:" in res.message


@pytest.mark.parametrize(
    ("kw", "field"),
    [
        (dict(block_size=1280), "slot_tokens"),
        (dict(conv_qk_pair=True), "conv_order"),
        (
            dict(pool=torch.empty((8, 2, 64, 4, 256), dtype=torch.float8_e4m3fn)),
            "page_shape",
        ),
    ],
)
def test_framework_slot_differences_are_declined(
    bf16_conv_pool: None, kw: dict[str, Any], field: str
) -> None:
    res = compare_layout_facts(_mk_facts(), _fw_facts(**kw))
    assert not res.eligible and res.code == "pool_layout_mismatch"
    assert f"{field}:" in res.message


def test_runner_block_size() -> None:
    assert runner_block_size(SimpleNamespace()) is None
    vc = SimpleNamespace(cache_config=SimpleNamespace(block_size=640))
    assert runner_block_size(SimpleNamespace(vllm_config=vc)) == 640
    cc = SimpleNamespace(block_size=1152)
    assert runner_block_size(SimpleNamespace(cache_config=cc)) == 1152


def test_framework_as_checked_out_matches_or_declines_by_conv_dtype() -> None:
    """With the tree's own `gdn_pool_layout` default: a bf16 conv pool matches,
    an fp32 one (the stock GDN kernel's only option once it requires fp32
    conv) is declined at bind as `conv_state_dtype_unsupported`."""
    fw = _fw_facts()
    res = compare_layout_facts(_mk_facts(), fw)
    if fw.conv_dtype == "bfloat16":
        assert res.eligible
    else:
        assert fw.conv_dtype == "float32"
        assert not res.eligible and res.code == "conv_state_dtype_unsupported"
