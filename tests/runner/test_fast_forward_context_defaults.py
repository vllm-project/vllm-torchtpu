# SPDX-License-Identifier: Apache-2.0
"""ix-unified-port: the hostov host-overhead stack (fast forward context, decode carry) is
default OFF, so configs that do not set it run the code the unified gates measured; =1 turns
each on (band H j-81e043bf ran both on)."""
import importlib
import types

import pytest
import torch
import vllm.forward_context as fc

from vllm_torchtpu.runner import fast_forward_context as ffc

_VARS = ("VLLM_TPU_FAST_FORWARD_CONTEXT",)


@pytest.fixture
def reload_with(monkeypatch):
    def _reload(**env):
        for v in _VARS:
            monkeypatch.delenv(v, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        importlib.reload(ffc)

    yield _reload
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    importlib.reload(ffc)


def test_unset_is_off(reload_with):
    reload_with()
    assert ffc._ENABLED is False


@pytest.mark.parametrize("val", ["0", "", "true"])
def test_only_1_turns_on(reload_with, val):
    reload_with(VLLM_TPU_FAST_FORWARD_CONTEXT=val)
    assert ffc._ENABLED is False


def test_1_turns_it_on(reload_with):
    reload_with(VLLM_TPU_FAST_FORWARD_CONTEXT="1")
    assert ffc._ENABLED is True


def test_off_takes_the_stock_forward_context(reload_with, monkeypatch):
    reload_with()
    monkeypatch.setattr(fc.current_platform, "set_additional_forward_context", lambda *a, **k: {}, raising=False)
    ffc._CACHE.clear()
    pc = types.SimpleNamespace(data_parallel_size=1, use_sequence_parallel_moe=True, is_moe_model=True,
                               data_parallel_rank=0)
    cc = types.SimpleNamespace(static_forward_context={"x": 1}, fast_moe_cold_start=False, static_all_moe_layers=None)
    cfg = types.SimpleNamespace(parallel_config=pc, compilation_config=cc)
    for _ in range(2):
        with ffc.fast_set_forward_context({}, cfg, num_tokens=4,
                                          num_tokens_across_dp=torch.tensor([4], dtype=torch.int32)):
            pass
    assert len(ffc._CACHE) == 0


def test_enabled_follows_the_switch(reload_with):
    reload_with()
    assert ffc.enabled() is False
    reload_with(VLLM_TPU_FAST_FORWARD_CONTEXT="1")
    assert ffc.enabled() is True
