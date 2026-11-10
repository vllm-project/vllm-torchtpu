# SPDX-License-Identifier: Apache-2.0
"""fast_set_forward_context must install a context equal to the stock one except for attn_metadata,
reuse the cached base only for the same key, and restore the previous context on exit."""
import dataclasses
import types

import torch
import vllm.forward_context as fc

from vllm_torchtpu.runner import fast_forward_context as ffc
import pytest


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    # Default off on ix-unified-port; these tests cover the enabled path (VLLM_TPU_FAST_FORWARD_CONTEXT=1).
    monkeypatch.setattr(ffc, "_ENABLED", True)


@pytest.fixture(autouse=True)
def _cpu_platform_hook(monkeypatch):
    # The TPU platform's hook moves the DP token tensor to the "tpu" device, which a CPU-only
    # container lacks. Stock and fast paths call the same hook, so a CPU stub keeps the comparison.
    try:
        torch.zeros(1).to("tpu")
    except Exception:
        monkeypatch.setattr(fc.current_platform, "set_additional_forward_context", lambda *a, **k: {}, raising=False)


class _PC:
    data_parallel_size = 1
    use_sequence_parallel_moe = True
    is_moe_model = True
    data_parallel_rank = 0


def _cfg():
    cc = types.SimpleNamespace(static_forward_context={"x": 1}, fast_moe_cold_start=False, static_all_moe_layers=None)
    return types.SimpleNamespace(parallel_config=_PC(), compilation_config=cc)


def _fields(ctx):
    d = {f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)}
    d.pop("attn_metadata")
    dp = d.pop("dp_metadata")
    return d, (None if dp is None else dp.num_tokens_across_dp_cpu.tolist())


def test_cached_context_matches_stock_and_swaps_metadata():
    ffc._CACHE.clear()
    cfg = _cfg()
    md1, md2 = {"a": 1}, {"a": 2}
    t = torch.tensor([4], dtype=torch.int32)
    with fc.set_forward_context(md1, cfg, num_tokens=4, num_tokens_across_dp=t.clone()):
        stock = _fields(fc.get_forward_context())
    with ffc.fast_set_forward_context(md1, cfg, num_tokens=4, num_tokens_across_dp=t.clone()):
        first = fc.get_forward_context()
        assert first.attn_metadata is md1
    with ffc.fast_set_forward_context(md2, cfg, num_tokens=4, num_tokens_across_dp=t.clone()):
        second = fc.get_forward_context()
        assert second.attn_metadata is md2
        assert second is not first
        f2 = _fields(second)
    s_d, s_dp = stock
    f_d, f_dp = f2
    assert f_d.keys() == s_d.keys()
    for k in s_d:
        if k == "additional_kwargs":
            continue
        assert f_d[k] == s_d[k], k
    assert len(ffc._CACHE) == 1


def test_restores_previous_context_and_keys_by_num_tokens():
    ffc._CACHE.clear()
    cfg = _cfg()
    before = fc._forward_context
    with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=torch.tensor([4], dtype=torch.int32)):
        pass
    with ffc.fast_set_forward_context({}, cfg, num_tokens=8, num_tokens_across_dp=torch.tensor([8], dtype=torch.int32)):
        assert fc.get_forward_context().dp_metadata is not None
    assert fc._forward_context is before
    assert len(ffc._CACHE) == 2


def test_extra_kwargs_take_stock_path():
    ffc._CACHE.clear()
    cfg = _cfg()
    with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=torch.tensor([4], dtype=torch.int32), skip_compiled=True):
        assert fc.get_forward_context().skip_compiled is True
    assert len(ffc._CACHE) == 0


def test_num_tokens_is_part_of_the_key():
    # Same (absent) DP tensor, different num_tokens: the stock path derives the DP token count from
    # num_tokens, so a cached context for 4 tokens must not be reused for 8.
    ffc._CACHE.clear()
    cfg = _cfg()
    with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=None):
        pass
    with ffc.fast_set_forward_context({}, cfg, num_tokens=8, num_tokens_across_dp=None):
        assert fc.get_forward_context().dp_metadata.num_tokens_across_dp_cpu.tolist() == [8]
    with ffc.fast_set_forward_context({}, cfg, num_tokens=8, num_tokens_across_dp=None):
        assert fc.get_forward_context().dp_metadata.num_tokens_across_dp_cpu.tolist() == [8]
    assert len(ffc._CACHE) == 2
