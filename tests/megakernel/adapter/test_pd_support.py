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
"""Prefill / decode disaggregated (PD) decode legs: the runner's decode
context parallel (DCP) facts and the framework's DCP slot facts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_torchtpu.distributed.dcp as dcp_group_mod
from tests.megakernel.adapter.test_pool_layout_facts import _LAYOUT, _fw_facts
from vllm_torchtpu import gdn_pool_layout as gpl
from vllm_torchtpu.megakernel.adapter import testing
from vllm_torchtpu.megakernel.adapter.plugin import RunnerView
from vllm_torchtpu.megakernel.core.runtime import (
    compare_layout_facts,
    facts_from_pool_layout,
)

TP = 8


def _dcp_runner(dcp: int = 4, interleave: int = 128) -> SimpleNamespace:
    return testing.fake_runner(
        tp_size=TP,
        decode_context_parallel_size=dcp,
        cp_kv_cache_interleave_size=interleave,
    )


# --- the runner view's DCP facts ------------------------------------------------


def test_runner_view_dcp_facts() -> None:
    view = RunnerView(testing.fake_runner(), tp_rank=0)
    assert (view.dcp_size, view.dcp_rank, view.dcp_interleave) == (1, 0, 1)
    view = RunnerView(_dcp_runner(), tp_rank=6, dcp_rank=2)
    assert (view.dcp_size, view.dcp_rank, view.dcp_interleave) == (4, 2, 128)


def test_runner_view_dcp_rank_comes_from_the_dcp_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an explicit rank the view asks vLLM's DCP group, and refuses to
    guess when DCP is on but there is no group; without DCP it is 0."""
    monkeypatch.setattr(
        dcp_group_mod, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=3)
    )
    # TP rank 6 is DCP rank 2 by `tp_rank % dcp`; the view must report the
    # group's 3, not guess.
    assert RunnerView(_dcp_runner(), tp_rank=6).dcp_rank == 3
    monkeypatch.setattr(dcp_group_mod, "get_dcp_group", lambda: None)
    assert RunnerView(testing.fake_runner(), tp_rank=7).dcp_rank == 0
    with pytest.raises(RuntimeError, match="no DCP group"):
        _ = RunnerView(_dcp_runner(), tp_rank=7).dcp_rank
    bad = _dcp_runner()
    bad.parallel_config.decode_context_parallel_size = 0
    with pytest.raises(RuntimeError, match="expected a positive int"):
        _ = RunnerView(bad).dcp_size


# --- the framework's DCP slot facts ------------------------------------------------


def test_framework_dcp_slot_facts_against_the_megakernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DCP 4 framework with page-sized interleave on DCP rank 2 is exactly
    what a megakernel built for DCP 4 on TP rank 6 expects; a framework at
    DCP 1, token interleave or another DCP rank is declined at bind."""
    monkeypatch.setattr(gpl, "DEFAULT_POOLED_GDN_CONV_STATE_DTYPE", "torch.bfloat16")
    mk4 = facts_from_pool_layout(_LAYOUT, TP, dcp=4, dcp_rank=6 % 4)
    fw4 = _fw_facts(dcp_size=4, dcp_rank=2, dcp_interleave=128)
    assert fw4 == mk4 and compare_layout_facts(mk4, fw4).eligible
    for fw, field in (
        (_fw_facts(), "dcp_size"),
        (_fw_facts(dcp_size=4, dcp_rank=2, dcp_interleave=1), "dcp_page_owner"),
        (_fw_facts(dcp_size=4, dcp_rank=1, dcp_interleave=128), "dcp_rank"),
    ):
        res = compare_layout_facts(mk4, fw)
        assert not res.eligible and res.code == "pool_layout_mismatch"
        assert f"{field}:" in res.message
    # DCP 1 on both sides: the default slot, unchanged.
    assert _fw_facts(dcp_interleave=128) == facts_from_pool_layout(_LAYOUT, TP)
