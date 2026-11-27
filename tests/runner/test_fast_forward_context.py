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
"""fast_set_forward_context must install a context equal to the stock one
except for attn_metadata, reuse the cached base only for the same key, and
restore the previous context on exit."""

import contextlib
import dataclasses
import types

import numpy as np
import pytest
import torch
import vllm.forward_context as fc

from vllm_torchtpu.runner import fast_forward_context as ffc


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    # Off by default; these tests cover the enabled path, each with an empty
    # cache of its own.
    monkeypatch.setattr(ffc, "_ENABLED", True)
    monkeypatch.setattr(ffc, "_CACHE", {})


@pytest.fixture(autouse=True)
def _cpu_platform_hook(monkeypatch):
    # The TPU platform's hook moves the DP token tensor to the "tpu" device,
    # which a CPU-only container lacks. Stock and fast paths call the same
    # hook, so a CPU stub keeps the comparison.
    try:
        torch.zeros(1).to("tpu")
    except Exception:
        monkeypatch.setattr(
            fc.current_platform,
            "set_additional_forward_context",
            lambda *a, **k: {},
            raising=False,
        )


class _PC:
    data_parallel_size = 1
    use_sequence_parallel_moe = True
    is_moe_model = True
    data_parallel_rank = 0


def _cfg():
    cc = types.SimpleNamespace(
        static_forward_context={"x": 1},
        fast_moe_cold_start=False,
        static_all_moe_layers=None,
    )
    return types.SimpleNamespace(parallel_config=_PC(), compilation_config=cc)


def _fields(ctx):
    d = {f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)}
    d.pop("attn_metadata")
    dp = d.pop("dp_metadata")
    return d, (None if dp is None else dp.num_tokens_across_dp_cpu.tolist())


def test_cached_context_matches_stock_and_swaps_metadata():
    cfg = _cfg()
    md1, md2 = {"a": 1}, {"a": 2}
    t = torch.tensor([4], dtype=torch.int32)
    with fc.set_forward_context(md1, cfg, num_tokens=4, num_tokens_across_dp=t.clone()):
        stock = _fields(fc.get_forward_context())
    with ffc.fast_set_forward_context(
        md1, cfg, num_tokens=4, num_tokens_across_dp=t.clone()
    ):
        first = fc.get_forward_context()
        assert first.attn_metadata is md1
    with ffc.fast_set_forward_context(
        md2, cfg, num_tokens=4, num_tokens_across_dp=t.clone()
    ):
        second = fc.get_forward_context()
        assert second.attn_metadata is md2
        assert second is not first
        f2 = _fields(second)
    s_d, s_dp = stock
    f_d, f_dp = f2
    assert f_dp == s_dp
    assert f_d.keys() == s_d.keys()
    for k in s_d:
        if k == "additional_kwargs":
            continue
        assert f_d[k] == s_d[k], k
    assert len(ffc._CACHE) == 1
    # The cached context keeps no call's attn_metadata alive.
    assert all(ctx.attn_metadata is None for ctx in ffc._CACHE.values())


def test_restores_previous_context_and_keys_by_num_tokens():
    cfg = _cfg()
    before = fc._forward_context
    with ffc.fast_set_forward_context(
        {}, cfg, num_tokens=4, num_tokens_across_dp=torch.tensor([4], dtype=torch.int32)
    ):
        pass
    with ffc.fast_set_forward_context(
        {}, cfg, num_tokens=8, num_tokens_across_dp=torch.tensor([8], dtype=torch.int32)
    ):
        assert fc.get_forward_context().dp_metadata is not None
    assert fc._forward_context is before
    assert len(ffc._CACHE) == 2


def test_extra_kwargs_take_stock_path():
    cfg = _cfg()
    with ffc.fast_set_forward_context(
        {},
        cfg,
        num_tokens=4,
        num_tokens_across_dp=torch.tensor([4], dtype=torch.int32),
        skip_compiled=True,
    ):
        assert fc.get_forward_context().skip_compiled is True
    assert len(ffc._CACHE) == 0


def test_a_dp_count_that_is_not_a_tensor_takes_the_stock_path(monkeypatch):
    # Serving passes a CPU tensor or None. Anything else, such as a NumPy
    # array (which has .tolist and a .device string but no .numel), gets no
    # key and takes the stock path instead of raising.
    cfg = _cfg()
    stock = []
    monkeypatch.setattr(
        ffc, "_stock", lambda *a, **k: stock.append(a) or contextlib.nullcontext()
    )
    for dp in (np.array([4], dtype=np.int32), [4]):
        assert ffc._key(cfg, 4, dp) is None
        with ffc.fast_set_forward_context(
            {}, cfg, num_tokens=4, num_tokens_across_dp=dp
        ):
            pass
    assert len(stock) == 2
    assert ffc._key(cfg, 4, torch.tensor([4], dtype=torch.int32)) == (id(cfg), 4, (4,))
    assert len(ffc._CACHE) == 0


def test_batch_size_tracking_takes_the_stock_path(monkeypatch):
    # VLLM_LOG_BATCHSIZE_INTERVAL >= 0 sets vLLM's module-level
    # track_batchsize; the stock path then times each forward.
    monkeypatch.setattr(fc, "track_batchsize", True)
    with ffc.fast_set_forward_context(
        None,
        _cfg(),
        num_tokens=4,
        num_tokens_across_dp=torch.tensor([4], dtype=torch.int32),
    ):
        pass
    assert len(ffc._CACHE) == 0


def test_num_tokens_is_part_of_the_key():
    # Same (absent) DP tensor, different num_tokens: the stock path derives
    # the DP token count from num_tokens, so a cached context for 4 tokens
    # must not be reused for 8.
    cfg = _cfg()
    with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=None):
        pass
    with ffc.fast_set_forward_context({}, cfg, num_tokens=8, num_tokens_across_dp=None):
        assert (
            fc.get_forward_context().dp_metadata.num_tokens_across_dp_cpu.tolist()
            == [8]
        )
    with ffc.fast_set_forward_context({}, cfg, num_tokens=8, num_tokens_across_dp=None):
        assert (
            fc.get_forward_context().dp_metadata.num_tokens_across_dp_cpu.tolist()
            == [8]
        )
    assert len(ffc._CACHE) == 2


def test_each_cached_context_starts_its_own_moe_layer_counter():
    """With fast_moe_cold_start, vLLM's MoE ops name their layer by reading
    `all_moe_layers[moe_layer_index]` and advancing the index on the current
    context (`moe_runner.get_layer_from_name`). Every fast call must start at
    0 and share only the read-only layer list with the cached context."""
    cfg = _cfg()
    cfg.compilation_config.fast_moe_cold_start = True
    cfg.compilation_config.static_all_moe_layers = ["moe.0", "moe.1"]
    t = torch.tensor([4], dtype=torch.int32)
    for step in range(3):
        with ffc.fast_set_forward_context(
            {"step": step}, cfg, num_tokens=4, num_tokens_across_dp=t.clone()
        ):
            ctx = fc.get_forward_context()
            assert ctx.attn_metadata == {"step": step}
            assert ctx.moe_layer_index == 0
            assert ctx.all_moe_layers is cfg.compilation_config.static_all_moe_layers
            for name in ("moe.0", "moe.1"):
                assert ctx.all_moe_layers[ctx.moe_layer_index] == name
                ctx.moe_layer_index += 1
    assert len(ffc._CACHE) == 1
    assert cfg.compilation_config.static_all_moe_layers == ["moe.0", "moe.1"]
