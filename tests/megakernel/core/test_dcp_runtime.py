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
"""Decode context parallelism (DCP) in the runtime contract: the DCP fields
of the pool slot facts and their bind-time comparison, the host pool-metadata
check on the rank-local page table, and `align_cache_exec`."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vllm_torchtpu.megakernel import core as mk
from vllm_torchtpu.megakernel.core.pool.layout import PoolLayout
from vllm_torchtpu.megakernel.core.profile import DEFAULT_PROFILE
from vllm_torchtpu.megakernel.core.runtime import (
    align_cache_exec,
    compare_layout_facts,
    facts_from_pool_layout,
)
from vllm_torchtpu.megakernel.core.runtime.layout_facts import dcp_facts
from vllm_torchtpu.models.qwen3_5.megakernel.model import Qwen35

TP = 8
# The Qwen3.5 slot of `tests/megakernel/adapter/test_pool_layout_facts.py`.
_LAYOUT = PoolLayout(num_pools=15, slots_per_pool=3, kh=16, vh=64)
# sha256 of that slot's canonical payload as the facts were before they had
# DCP fields (schema v1, 18 fields): a DCP 1 slot must keep it.
_DCP1_FINGERPRINT = "44c4e4ce1fc73050045446c84802b5c5f384a08a5f35e9b772e667c1b11fd572"
_V1_KEYS = {
    "schema",
    "page_dtype",
    "page_shape",
    "page_size",
    "token_bytes",
    "slot_tokens",
    "ssm_dtype",
    "ssm_bytes",
    "ssm_offset_bytes",
    "conv_dtype",
    "conv_bytes",
    "conv_offset_bytes",
    "conv_order",
    "local_kh",
    "local_vh",
    "d_k",
    "d_v",
    "conv_taps",
    "conv_dim",
}


def test_dcp1_facts_keep_their_payload_and_fingerprint() -> None:
    f = facts_from_pool_layout(_LAYOUT, TP)
    assert (f.dcp_size, f.dcp_rank, f.dcp_interleave, f.dcp_page_owner) == (
        1,
        0,
        0,
        "replicated",
    )
    assert f == facts_from_pool_layout(_LAYOUT, TP, dcp=1, dcp_rank=0)
    assert set(f.payload()) == _V1_KEYS
    assert f.payload()["schema"] == "megakernel-pool-slot-facts-v1"
    assert f.fingerprint() == _DCP1_FINGERPRINT


def test_dcp_facts_name_the_ownership_rule() -> None:
    assert dcp_facts(1, 0, 1, 128) == {}
    assert dcp_facts(1, 0, 128, 128) == {}
    assert dcp_facts(4, 2, 128, 128) == dict(
        dcp_size=4, dcp_rank=2, dcp_interleave=128, dcp_page_owner="page_mod_dcp"
    )
    assert dcp_facts(4, 2, 1, 128)["dcp_page_owner"] == "interleave_1_tokens_mod_dcp"
    assert (
        dcp_facts(2, 1, 256, 128)["dcp_page_owner"] == "interleave_256_tokens_mod_dcp"
    )


def test_dcp_facts_enter_the_payload() -> None:
    f1 = facts_from_pool_layout(_LAYOUT, TP)
    f4 = facts_from_pool_layout(_LAYOUT, TP, dcp=4, dcp_rank=2)
    assert (f4.dcp_size, f4.dcp_rank, f4.dcp_interleave, f4.dcp_page_owner) == (
        4,
        2,
        128,
        "page_mod_dcp",
    )
    p4 = f4.payload()
    assert set(p4) == _V1_KEYS | {
        "dcp_size",
        "dcp_rank",
        "dcp_interleave",
        "dcp_page_owner",
    }
    assert p4["dcp_size"] == 4 and p4["dcp_page_owner"] == "page_mod_dcp"
    assert f4.fingerprint() != f1.fingerprint()
    other_rank = facts_from_pool_layout(_LAYOUT, TP, dcp=4, dcp_rank=1)
    assert other_rank.fingerprint() != f4.fingerprint()


@pytest.mark.parametrize(
    "framework, fields",
    [
        # a megakernel built for DCP 4 against a framework at DCP 1
        (
            lambda f: dataclasses.replace(
                f, dcp_size=1, dcp_rank=0, dcp_interleave=0, dcp_page_owner="replicated"
            ),
            ("dcp_size", "dcp_page_owner"),
        ),
        # the framework's DCP rank is not `tp_rank % dcp`
        (lambda f: dataclasses.replace(f, dcp_rank=1), ("dcp_rank",)),
        # token-interleaved DCP (vLLM's default interleave of 1)
        (
            lambda f: dataclasses.replace(f, **dcp_facts(4, 2, 1, 128)),
            ("dcp_interleave", "dcp_page_owner"),
        ),
    ],
)
def test_dcp_facts_are_compared(framework, fields) -> None:  # noqa: ANN001
    mk4 = facts_from_pool_layout(_LAYOUT, TP, dcp=4, dcp_rank=2)
    assert compare_layout_facts(mk4, dataclasses.replace(mk4)).eligible
    res = compare_layout_facts(mk4, framework(mk4))
    assert not res.eligible and res.code == "pool_layout_mismatch"
    for field in fields:
        assert f"{field}:" in res.message
    # and DCP 1 on both sides still compares equal
    f1 = facts_from_pool_layout(_LAYOUT, TP)
    assert compare_layout_facts(f1, dataclasses.replace(f1)).eligible


def _ex(dcp: int, T: int = 4) -> mk.ExecSpec:
    return mk.ExecSpec(
        R=TP,
        B=2 * T,
        T=T,
        ctx=1024,
        cache=mk.CacheExec(paged_kv=True, num_pages=64, dcp=dcp),
    )


def test_host_pool_metadata_check_reads_dcp_columns() -> None:
    """Sequence 0 writes page 2, sequence 1 pages 4 and 5. With DCP 4 those
    are columns 0 and 1 of the rank-local page table, so two columns hold
    them; with DCP 2 pages 4 and 5 are at column 2, so it takes three; at
    DCP 1 page 2 is already past a two-column table."""
    pt = np.asarray([[10, 11], [20, 21]], np.int32)
    pt3 = np.asarray([[10, 11, 12], [20, 21, 22]], np.int32)
    pos = np.asarray([300, 301, 302, 303, 638, 639, 640, 641], np.int32)
    mk.pool.check_step_pool_metadata(_LAYOUT, _ex(4), pt, pos)
    mk.pool.check_step_pool_metadata(_LAYOUT, _ex(2), pt3, pos)
    with pytest.raises(
        ValueError, match="seq 1 token 0 pos=638 maps to logical page 2"
    ):
        mk.pool.check_step_pool_metadata(_LAYOUT, _ex(2), pt, pos)
    with pytest.raises(
        ValueError, match="seq 0 token 0 pos=300 maps to logical page 2"
    ):
        mk.pool.check_step_pool_metadata(_LAYOUT, _ex(1), pt, pos)


def test_host_pool_metadata_check_keys_writes_by_owner() -> None:
    """Under DCP pages 0 and 1 share column 0, so positions 5 and 133 write the
    same physical page id and lane, on two different ranks: not a duplicate.
    The columns past the sequence's last one are not checked."""
    pt = np.asarray([[10, 0, 0, 0]], np.int32)
    pos = np.asarray([5, 133], np.int32)
    ex = mk.ExecSpec(
        R=TP, B=2, T=2, ctx=1024, cache=mk.CacheExec(paged_kv=True, num_pages=64, dcp=4)
    )
    mk.pool.check_step_pool_metadata(_LAYOUT, ex, pt, pos)
    ex1 = dataclasses.replace(ex, cache=mk.CacheExec(paged_kv=True, num_pages=64))
    with pytest.raises(ValueError, match="KV write page 0 outside"):
        mk.pool.check_step_pool_metadata(_LAYOUT, ex1, pt, pos)


def test_align_cache_exec_binds_the_dcp_size() -> None:
    base = mk.ExecSpec(R=TP, B=8, T=1, ctx=256)
    g = Qwen35.derive_geometry()
    kw = dict(num_pages=96, page_size=128, max_pages_per_seq=8)
    ex4 = align_cache_exec(base, DEFAULT_PROFILE, g, dcp=4, **kw)
    ex1 = align_cache_exec(base, DEFAULT_PROFILE, g, **kw)
    assert mk.cache_exec(ex4).dcp == 4 and mk.cache_exec(ex1).dcp == 1
    assert ex4 == dataclasses.replace(ex1, cache=dataclasses.replace(ex1.cache, dcp=4))
