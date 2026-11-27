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
"""Bind-time recurrent pool slot facts, model-free: a `PoolLayout` built
directly (`megakernel.core.runtime.layout_facts`) against the framework's view of
the same slot (`megakernel.adapter.pool_facts`)."""

from __future__ import annotations

import dataclasses
import importlib
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.megakernel.adapter.vllm_kv_cache_layout import kv_cache_layout
from vllm_torchtpu import gdn_pool_layout as gpl
from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry
from vllm_torchtpu.megakernel.adapter.pool_facts import (
    conv_qk_pair_layout_enabled,
    framework_pool_layout_facts,
)
from vllm_torchtpu.megakernel.core.pool.layout import PoolLayout
from vllm_torchtpu.megakernel.core.runtime import (
    LayoutDecline,
    PoolLayoutFacts,
    compare_layout_facts,
    facts_from_pool_layout,
)
from vllm_torchtpu.megakernel.core.runtime.layout_facts import dtype_name

TP = 8
# A unified pool with 15 pools of 3 recurrent slots each; 16 key and 64 value
# heads (global) of 128 dims; the default fp8 KV page `(2, 64, 4, 128)`, bf16
# recurrent state over 4 pages and a bf16 3-tap conv window on 1 page.
_LAYOUT = PoolLayout(num_pools=15, slots_per_pool=3, kh=16, vh=64)

_FACTS = PoolLayoutFacts(
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


@pytest.fixture(autouse=True)
def _kv_cache_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    kv_cache_layout(monkeypatch)


def _kv_layout(name: str) -> Any:
    """vLLM's `KVCacheLayout` member `name` (or its stand-in without vLLM)."""
    return importlib.import_module("vllm.v1.kv_cache_layout").KVCacheLayout[name]


def _mk_facts(layout: PoolLayout = _LAYOUT) -> PoolLayoutFacts:
    return facts_from_pool_layout(layout, TP)


def _gdn_layer(ssm_dtype: torch.dtype = torch.bfloat16) -> Any:
    """The attributes of one TP rank's GDN layer the framework lays out its
    pool slot from, for the same heads as `_LAYOUT`."""
    return SimpleNamespace(
        get_state_dtype=lambda: (torch.bfloat16, ssm_dtype),
        tp_head_geometry=derive_gdn_head_geometry(16, 64, TP),
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
    )


def _pool() -> torch.Tensor:
    return torch.empty((96, 2, 64, 4, 128), dtype=torch.float8_e4m3fn, device="meta")


def _fp8_pool(*page_shape: int) -> torch.Tensor:
    return torch.empty((8, *page_shape), dtype=torch.float8_e4m3fn, device="meta")


@pytest.fixture
def bf16_conv_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A framework whose pooled conv state is bf16 (`gdn_pool_layout`'s
    default decides this; tests pin it so they don't depend on the tree)."""
    monkeypatch.setattr(gpl, "DEFAULT_POOLED_GDN_CONV_STATE_DTYPE", "torch.bfloat16")


def _fw_facts(kv_layout: str = "LBHNC", **kw: Any) -> PoolLayoutFacts:
    args: dict[str, Any] = dict(
        pool=_pool(),
        linear_attn=_gdn_layer(),
        block_size=640,
        kv_layout=_kv_layout(kv_layout),
        pcp_size=1,
        conv_qk_pair=False,
    )
    args.update(kw)
    return framework_pool_layout_facts(**args)


def test_facts_from_pool_layout() -> None:
    f = _mk_facts()
    assert f == _FACTS
    assert len(f.fingerprint()) == 64
    assert f.fingerprint() == dataclasses.replace(f).fingerprint()
    assert f.fingerprint() != dataclasses.replace(f, conv_dtype="float32").fingerprint()
    assert f.payload()["schema"] == "megakernel-pool-slot-facts-v1"


def test_framework_slot_facts_match_the_layout(bf16_conv_pool: None) -> None:
    """A framework layout with a bf16 SSM, a bf16 conv pool and a 640-token
    block is exactly what the `PoolLayout` expects."""
    fw = _fw_facts()
    assert fw == _mk_facts()
    res = compare_layout_facts(_mk_facts(), fw)
    assert res.eligible and res.code == "ok"


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
    1152-token (9-page) slot while the layout addresses 5-page slots."""
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
            dict(pool=_fp8_pool(2, 64, 4, 256)),
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


def test_layout_differences_are_declined(bf16_conv_pool: None) -> None:
    """The comparison is field by field on the megakernel side too: a layout
    with fewer recurrent pages per slot no longer matches the framework."""
    short = dataclasses.replace(_LAYOUT, rec_pages_per_slot=3)
    res = compare_layout_facts(_mk_facts(short), _fw_facts())
    assert not res.eligible and res.code == "pool_layout_mismatch"
    assert "slot_tokens: megakernel=512 framework=640" in res.message


def test_framework_nhd_pool_reads_tokens_from_the_first_page_axis(
    bf16_conv_pool: None,
) -> None:
    """An NHD pool page is `(page_size, 2 * kvh / pack, pack, hd)`: the facts
    take the page size from axis 0, so the layout comparison names it. Even a
    page whose dims happen to equal the megakernel's HND page doesn't match."""
    # `PallasAttentionBackend.get_kv_cache_shape(n, 128, 1, 256, fp8)`.
    fw = _fw_facts(pool=_fp8_pool(128, 1, 4, 256), kv_layout="LBNHC")
    assert (fw.page_size, fw.token_bytes) == (128, 1024)
    res = compare_layout_facts(_mk_facts(), fw)
    assert not res.eligible and res.code == "pool_layout_mismatch"
    assert "page_size:" not in res.message and "token_bytes:" in res.message

    fw = _fw_facts(kv_layout="LBNHC")
    assert fw.page_shape == _mk_facts().page_shape
    assert (fw.page_size, fw.token_bytes) == (2, 32768)
    res = compare_layout_facts(_mk_facts(), fw)
    assert not res.eligible and res.code == "pool_layout_mismatch"
    assert "page_size: megakernel=128 framework=2" in res.message


@pytest.mark.parametrize(
    ("kw", "code", "match"),
    [
        # HND, 1 KV head of 128 dims in fp8: 256-byte token rows.
        (dict(pool=_fp8_pool(2, 32, 4, 128)), "pool_layout_mismatch", "granules"),
        (dict(pool=_fp8_pool(2, 64, 512)), "pool_layout_mismatch", "4-D page"),
        (dict(kv_layout="BLNHC"), "pool_layout_unknown", "BLNHC"),
        (dict(pcp_size=2), "pool_layout_unknown", "prefill context parallel"),
    ],
)
def test_framework_slot_it_cannot_describe_is_a_layout_decline(
    kw: dict[str, Any], code: str, match: str
) -> None:
    """A pool the facts can't read (non-granule token rows, another page
    rank), a KV layout they don't know and PCP are named declines, not raw
    errors that would vote `bind_error`."""
    with pytest.raises(LayoutDecline, match=match) as exc:
        _fw_facts(**kw)
    assert exc.value.result.code == code and not exc.value.result.eligible


def test_conv_qk_pair_layout_enabled_reads_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    assert conv_qk_pair_layout_enabled() is True
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "0")
    assert conv_qk_pair_layout_enabled() is False


def test_f32_conv_pool_layout_matches_an_fp32_conv_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `PoolLayout` with an f32 conv window describes exactly the slot a
    framework storing fp32 conv lays out, so the bind check passes."""
    monkeypatch.setattr(gpl, "DEFAULT_POOLED_GDN_CONV_STATE_DTYPE", "torch.float32")
    mk = _mk_facts(dataclasses.replace(_LAYOUT, conv_dtype=jnp.float32))
    fw = _fw_facts()
    assert mk.conv_dtype == fw.conv_dtype == "float32"
    assert mk.conv_bytes == fw.conv_bytes == 18432
    assert compare_layout_facts(mk, fw).eligible
    assert mk.fingerprint() != _mk_facts().fingerprint()


def test_dtype_name_accepts_jax_scalar_types_and_dtypes() -> None:
    """`dtype_name` names jax scalar types (`jnp.float32`, the form a
    `PoolLayout` stores), dtypes, strings and torch dtypes alike."""
    assert dtype_name(jnp.float32) == "float32"
    assert dtype_name(jnp.bfloat16) == "bfloat16"
    assert dtype_name(np.dtype("float32")) == "float32"
    assert dtype_name("bfloat16") == "bfloat16"
    assert dtype_name(torch.bfloat16) == "bfloat16"
    assert dtype_name(torch.float8_e4m3fn) == "float8_e4m3fn"
