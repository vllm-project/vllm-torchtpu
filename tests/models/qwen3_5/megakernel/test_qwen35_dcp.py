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
"""The Qwen3.5 template under decode context parallelism (DCP): the SPEC
declares it, `create` declines a DCP size the kernel can't shard (by name),
and `bind` checks the DCP slot facts against the runner's."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_torchtpu.megakernel.adapter.driver as driver_mod
import vllm_torchtpu.megakernel.adapter.guards as guards_mod
from tests.models.qwen3_5.megakernel.test_qwen35_pool_layout_facts import (
    _qwen35_gdn_layer,
)
from tests.models.qwen3_5.megakernel.test_qwen35_template import _full_size_runner
from vllm_torchtpu import gdn_pool_layout as gpl
from vllm_torchtpu.megakernel.adapter import testing
from vllm_torchtpu.megakernel.adapter.plugin import RunnerView
from vllm_torchtpu.megakernel.core.runtime import facts_from_pool_layout
from vllm_torchtpu.megakernel.core.runtime.contract import LayoutDecline
from vllm_torchtpu.models.qwen3_5 import megakernel as qwen_plugin
from vllm_torchtpu.models.qwen3_5.megakernel import adapter
from vllm_torchtpu.models.qwen3_5.megakernel.model import Qwen35

TP = 8


def _check(runner: Any) -> Any:
    return guards_mod.check_spec(
        runner, qwen_plugin.SPEC, view=RunnerView(runner, tp_rank=0)
    )


@pytest.mark.parametrize("dcp", [2, 4, 8])
def test_spec_declares_dcp(dcp: int) -> None:
    """The guard admits any DCP size (it reads no geometry); `create` checks
    the size against the KV heads."""
    runner = testing.fake_runner(
        tp_size=TP, mamba_ssm_cache_dtype="bfloat16", decode_context_parallel_size=dcp
    )
    assert guards_mod.detect_parallelism(runner, TP) == ["dcp"]
    assert _check(runner) is None


@pytest.mark.parametrize(
    "dcp, tp, nkv, ok",
    [
        (1, 8, 2, True),
        (2, 8, 2, True),
        (4, 8, 2, True),
        (8, 8, 2, False),
        (3, 8, 2, False),
    ],
)
def test_dcp_size_decline(dcp: int, tp: int, nkv: int, ok: bool) -> None:
    dec = adapter.dcp_size_decline(dcp, tp, nkv)
    if ok:
        assert dec is None
    else:
        assert dec is not None and dec.code == "dcp_geometry_unsupported"
        assert f"decode_context_parallel_size={dcp}" in dec.message


def test_create_declines_a_dcp_it_cannot_shard_and_keeps_the_rest() -> None:
    view = RunnerView(_full_size_runner(decode_context_parallel_size=8), tp_rank=0)
    dec = adapter.create(view, {"buckets": "8"})
    assert dec.code == "dcp_geometry_unsupported"
    for dcp in (1, 2, 4):
        view = RunnerView(
            _full_size_runner(decode_context_parallel_size=dcp), tp_rank=5
        )
        prog = adapter.create(view, {"buckets": "8"})
        assert isinstance(prog, adapter.Qwen35Program)
        assert prog.dcp_size == dcp and prog._dcp_rank == (5 % dcp if dcp > 1 else 0)


def test_dcp_decline_is_in_the_vote_table() -> None:
    assert "dcp_geometry_unsupported" in qwen_plugin.SPEC.decline_codes
    codes = driver_mod.vote_codes(qwen_plugin.SPEC.decline_codes)
    assert "dcp_geometry_unsupported" in codes[:-1]


def _slot_driver(dcp: int, tp_rank: int) -> Any:
    drv = object.__new__(adapter.Qwen35Program)
    drv.backbone = SimpleNamespace(layers=[SimpleNamespace(linear_attn=object())])
    drv.spec = SimpleNamespace(
        cache_backend=SimpleNamespace(layout=Qwen35.pool_layout(R=TP))
    )
    drv.base_exec = SimpleNamespace(R=TP)
    drv._display_name = "test"
    drv.dcp_size, drv.tp_rank = dcp, tp_rank
    return drv


def test_bind_checks_the_dcp_slot_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The megakernel's facts carry DCP rank `tp_rank % dcp`; the runner's DCP
    group must agree, and a runner without DCP is declined."""
    layout = Qwen35.pool_layout(R=TP)
    drv = _slot_driver(4, 6)

    def fw(dcp: int, rank: int) -> Any:
        facts = facts_from_pool_layout(layout, TP, dcp=dcp, dcp_rank=rank)
        return lambda *_a: facts

    monkeypatch.setattr(drv, "_framework_slot_facts", fw(4, 2), raising=False)
    drv._check_pool_layout_facts(SimpleNamespace(), object())
    assert (drv.pool_layout_facts.dcp_size, drv.pool_layout_facts.dcp_rank) == (4, 2)
    for bad in (fw(4, 1), fw(1, 0)):
        monkeypatch.setattr(drv, "_framework_slot_facts", bad, raising=False)
        with pytest.raises(LayoutDecline) as exc:
            drv._check_pool_layout_facts(SimpleNamespace(), object())
        assert exc.value.result.code == "pool_layout_mismatch"


def test_framework_slot_facts_read_the_runner_dcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_framework_slot_facts` hands the runner's DCP size, rank and interleave
    to the framework facts: TP rank 6 of a DCP 4 decode leg with 128-token
    interleave is DCP rank 2, as the kernel assumes."""
    monkeypatch.setattr(gpl, "DEFAULT_POOLED_GDN_CONV_STATE_DTYPE", "torch.bfloat16")
    runner = testing.fake_runner(
        tp_size=TP,
        block_size=640,
        decode_context_parallel_size=4,
        cp_kv_cache_interleave_size=128,
    )
    view = RunnerView(runner, tp_rank=6, dcp_rank=2)
    pool = torch.empty((96, 2, 64, 4, 128), dtype=torch.float8_e4m3fn, device="meta")
    drv = _slot_driver(4, 6)
    fw = adapter.Qwen35Program._framework_slot_facts(
        drv, view, _qwen35_gdn_layer(), pool
    )
    mk = facts_from_pool_layout(Qwen35.pool_layout(R=TP), TP, dcp=4, dcp_rank=2)
    assert fw == mk
    # without DCP the facts are the default slot's
    view1 = RunnerView(testing.fake_runner(tp_size=TP, block_size=640), tp_rank=6)
    fw1 = adapter.Qwen35Program._framework_slot_facts(
        drv, view1, _qwen35_gdn_layer(), pool
    )
    assert fw1 == facts_from_pool_layout(Qwen35.pool_layout(R=TP), TP)


def test_program_dcp_rank() -> None:
    drv = _slot_driver(4, 5)
    assert drv._dcp_rank == 1
    drv = _slot_driver(1, 5)
    assert drv._dcp_rank == 0
    # a program built without a DCP size (direct use) serves without DCP
    assert adapter.Qwen35Program.dcp_size == 1
