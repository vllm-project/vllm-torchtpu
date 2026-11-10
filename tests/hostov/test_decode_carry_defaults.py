# SPDX-License-Identifier: Apache-2.0
"""The decode carry is default OFF; =1 turns it on (band H j-81e043bf ran it on).
The decode-carry half of test_hostov_defaults.py; the fast-forward-context half
lands with the cached forward context."""
import importlib
import types

import pytest
import torch

from vllm_torchtpu.runner import decode_carry as dc

_VAR = "VLLM_TPU_DECODE_CARRY"


@pytest.fixture
def reload_with(monkeypatch):
    def _reload(**env):
        monkeypatch.delenv(_VAR, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        importlib.reload(dc)

    yield _reload
    monkeypatch.delenv(_VAR, raising=False)
    importlib.reload(dc)


def test_unset_is_off(reload_with):
    reload_with()
    assert dc.ENABLED is False


@pytest.mark.parametrize("val", ["0", "", "true"])
def test_only_1_turns_on(reload_with, val):
    reload_with(VLLM_TPU_DECODE_CARRY=val)
    assert dc.ENABLED is False


def test_1_turns_it_on(reload_with):
    reload_with(VLLM_TPU_DECODE_CARRY="1")
    assert dc.ENABLED is True


def test_off_never_carries_and_never_records(reload_with):
    reload_with()
    r = types.SimpleNamespace(_pre_async_results=types.SimpleNamespace())
    assert dc.check(r, start_index=0, num_reqs=1, num_scheduled_tokens_per_req_list=[4],
                    total_num_scheduled_tokens=4, padded_total_num_scheduled_tokens=4,
                    target_num_reqs=4, layout_plan=None) is None
    assert dc.record(r, start_index=0, num_reqs=1, num_scheduled_tokens_per_req_list=[4],
                     total_num_scheduled_tokens=4, padded_total_num_scheduled_tokens=4,
                     target_num_reqs=4, sub_pairs=((0, 4),), seq_lens=torch.zeros(4)) is False
    res = types.SimpleNamespace()
    dc.tag_results(res)
    assert not hasattr(res, "_decode_carry_seq")
