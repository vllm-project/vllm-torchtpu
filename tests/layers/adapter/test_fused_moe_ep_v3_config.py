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
"""Fused EP v3 settings beyond Config.for_shape: none by default; the
Qwen3.5 layout table's entry for this engine, read once at prebuild, changes
the named fields (and only those); an unknown field or a refused value
raises."""

import dataclasses
from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.fused_moe.v3 import Config
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as A

SHAPE = dict(hidden=4096, inter=1024, experts_per_shard=64, top_k=10)
QWEN3_5 = "Qwen3_5MoeForConditionalGeneration"


def _engine(arch, dp, tp):
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=[arch]),
        parallel_config=SimpleNamespace(data_parallel_size=dp, tensor_parallel_size=tp),
    )


def test_default_is_the_shape_default(monkeypatch):
    monkeypatch.setattr(A, "_CONFIG_OVERRIDES", {})
    assert A.served_config(**SHAPE) == Config.for_shape(**SHAPE)
    assert A.TILE_ROWS == 128


def test_overrides_only_named_fields(monkeypatch):
    monkeypatch.setattr(
        A,
        "_CONFIG_OVERRIDES",
        {"sorted_rows": False, "weight_slots": 4, "weight_prefetch": 3},
    )
    got, base = A.served_config(**SHAPE), Config.for_shape(**SHAPE)
    assert (
        got.sorted_rows is False and got.weight_slots == 4 and got.weight_prefetch == 3
    )
    changed = {
        f.name
        for f in dataclasses.fields(Config)
        if getattr(got, f.name) != getattr(base, f.name)
    }
    assert changed == {"sorted_rows", "weight_slots", "weight_prefetch"}


@pytest.mark.parametrize("bad", [{"sorted_row": False}, {"weight_slots": 1}])
def test_refuses(monkeypatch, bad):
    monkeypatch.setattr(A, "_CONFIG_OVERRIDES", bad)
    with pytest.raises(ValueError):
        A.served_config(**SHAPE)


def test_opt_in_kernel_flags_default_off():
    """The TP8 layouts serve the fused EP v3 kernel with these flags unset:
    each changes the kernel's programs, so each must stay opt-in (through
    the layout table), and an empty override keeps them off."""
    flags = (
        "combine_real_tokens",
        "sort_index_compare",
        "compact_message",
        "reorder_chain",
    )
    for config in (Config(), A.apply_config_overrides(Config(), {})):
        assert {f: getattr(config, f) for f in flags} == dict.fromkeys(flags, False)
    on = A.apply_config_overrides(
        Config(), {"compact_message": True, "reorder_chain": True}
    )
    assert on.compact_message is True and on.reorder_chain is True


def test_qwen3_5_dp4_tp2_takes_the_decode_entry():
    """The settings every DP4 x TP2 decode leg ran."""
    assert A.layout_overrides(_engine(QWEN3_5, 4, 2)) == {
        "combine_real_tokens": True,
        "sort_index_compare": True,
        "compact_message": True,
        "reorder_chain": True,
        "dense_slots": True,
        "dense_slots_fused": True,
        "dense_slots_merged": True,
        "dense_slots_skip_unrouted": True,
    }


@pytest.mark.parametrize(
    "engine",
    [
        _engine(QWEN3_5, 1, 8),
        _engine(QWEN3_5, 8, 1),
        _engine("Qwen3MoeForCausalLM", 4, 2),
        _engine("DeepseekV3ForCausalLM", 4, 2),
        None,
    ],
    ids=[
        "qwen3_5_tp8",
        "qwen3_5_dp8",
        "qwen3_moe_dp4_tp2",
        "deepseek_dp4_tp2",
        "no_engine",
    ],
)
def test_other_engines_keep_the_shape_defaults(engine):
    assert A.layout_overrides(engine) == {}


def test_every_table_entry_is_a_valid_config():
    for fields in A.QWEN3_5_CONFIG_BY_LAYOUT.values():
        A.apply_config_overrides(Config.for_shape(**SHAPE), fields)


def test_the_entry_is_a_copy():
    got = A.layout_overrides(_engine(QWEN3_5, 4, 2))
    got["dense_slots"] = False
    assert A.QWEN3_5_CONFIG_BY_LAYOUT[(4, 2)]["dense_slots"] is True


@pytest.mark.parametrize(
    "engine,want",
    [(_engine(QWEN3_5, 4, 2), True), (_engine(QWEN3_5, 1, 8), False)],
    ids=["qwen3_5_dp4_tp2", "qwen3_5_tp8"],
)
def test_prebuild_reads_the_entry_under_the_loader_config(monkeypatch, engine, want):
    """prebuild_fused_moe_ep selects the entry from the loader's config
    context; the traced forward later reads the stored overrides."""
    import vllm.config

    monkeypatch.setattr(A, "_CONFIG_OVERRIDES", {})
    monkeypatch.setattr(A.envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: engine)
    # Stop right after the read: prefill context parallelism refuses the layer.
    monkeypatch.setattr(A, "_pcp_size", lambda layer: 2)
    A.prebuild_fused_moe_ep(SimpleNamespace(), 10, True, "silu")
    assert bool(A._CONFIG_OVERRIDES) is want
    assert A.layout_overrides(engine) == A._CONFIG_OVERRIDES


def test_prebuild_off_leaves_no_overrides(monkeypatch):
    monkeypatch.setattr(A, "_CONFIG_OVERRIDES", {})
    monkeypatch.setattr(A.envs, "USE_MOE_FUSED_EP_KERNEL", False)
    A.prebuild_fused_moe_ep(SimpleNamespace(), 10, True, "silu")
    assert A._CONFIG_OVERRIDES == {}
