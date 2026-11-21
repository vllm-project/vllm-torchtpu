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
"""Decode lanes and tuned tiles of the batched RPA kernel: the lanes table, the
layers that look it up, and how the batched RPA and DCP entries hand both to the
kernel."""

import functools
from types import SimpleNamespace

import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.layers.adapter import attention, batched_rpa_decode, cp_attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

FP8 = torch.float8_e4m3fn
TPU7X, V6E = pltpu.ChipVersion.TPU_7X, pltpu.ChipVersion.TPU_V6E


@pytest.mark.parametrize(
    "chip, num_kv_heads, head_dim, dtype, max_model_len, lanes",
    [
        # The measured shape on the chip it was measured on: one KV head per
        # device, head_dim 256, fp8 KV, long context (the DP4xTP2 decode leg
        # serves max_model_len 262144).
        (TPU7X, 1, 256, FP8, 262144, 4),
        (TPU7X, 1, 256, torch.float8_e5m2, 98304, 4),
        # Anything else keeps the wrapper default, another chip included.
        (V6E, 1, 256, FP8, 262144, None),
        (TPU7X, 1, 256, FP8, 98303, None),
        (TPU7X, 1, 256, FP8, 8192, None),
        (TPU7X, 2, 256, FP8, 262144, None),
        (TPU7X, 1, 128, FP8, 262144, None),
        (TPU7X, 1, 256, None, 262144, None),
        (TPU7X, 1, 256, torch.uint8, 262144, None),
    ],
)
def test_decode_lanes_table(chip, num_kv_heads, head_dim, dtype, max_model_len, lanes):
    assert (
        batched_rpa_decode.decode_lanes(
            chip_version=chip,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_cache_dtype=dtype,
            max_model_len=max_model_len,
        )
        == lanes
    )


def _entry_rpa_func(
    monkeypatch,
    *,
    decode_query_size,
    decode_lanes,
    tuned_tiles=False,
    skip_empty_mixed=False,
):
    seen = {}

    def impl(*args, rpa_func, **kwargs):
        seen["rpa_func"] = rpa_func

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", impl)
    attention._pallas_rpa_kernel_batched(
        None,
        SimpleNamespace(shape=(64, 16, 256)),
        *([None] * 10),
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=None,
        decode_query_size=decode_query_size,
        decode_lanes=decode_lanes,
        tuned_tiles=tuned_tiles,
        skip_empty_mixed=skip_empty_mixed,
    )
    return seen["rpa_func"]


def test_decode_lanes_reach_the_kernel(monkeypatch):
    rpa_func = _entry_rpa_func(monkeypatch, decode_query_size=4, decode_lanes=4)
    assert isinstance(rpa_func, functools.partial)
    assert rpa_func.func is attention.ragged_paged_attention_batched
    assert rpa_func.keywords == {"decode_batch_size": 4, "decode_query_size": 4}


def test_no_lanes_keeps_the_kernel_default(monkeypatch):
    rpa_func = _entry_rpa_func(monkeypatch, decode_query_size=1, decode_lanes=None)
    assert rpa_func is attention.ragged_paged_attention_batched
    rpa_func = _entry_rpa_func(monkeypatch, decode_query_size=4, decode_lanes=None)
    assert rpa_func.keywords == {"decode_query_size": 4}


def test_tuned_tiles_reach_the_kernel(monkeypatch):
    rpa_func = _entry_rpa_func(
        monkeypatch, decode_query_size=4, decode_lanes=4, tuned_tiles=True
    )
    assert rpa_func.keywords == {
        "decode_batch_size": 4,
        "decode_query_size": 4,
        "tuned_tiles": True,
    }


def test_skip_empty_mixed_reaches_the_kernel_on_its_own(monkeypatch):
    rpa_func = _entry_rpa_func(
        monkeypatch,
        decode_query_size=4,
        decode_lanes=None,
        skip_empty_mixed=True,
    )
    assert rpa_func.keywords == {"decode_query_size": 4, "skip_empty_mixed": True}


def _lanes_after_init(monkeypatch, *, tuned_tiles, max_model_len=262144, chip=TPU7X):
    """decode_lanes of a batched RPA layer with one fp8 KV head of 256 per
    device, as its __init__ sets it; tuned_tiles stands in for the Qwen3.5
    rule."""

    def base_init(self, *args, **kwargs):
        self.head_size = 256
        self.num_kv_heads = 1
        self.kv_cache_quantized_dtype = FP8

    monkeypatch.setattr(attention.PallasAttentionBackendImpl, "__init__", base_init)
    monkeypatch.setattr(attention, "qwen3_5_tuned_tiles", lambda _: tuned_tiles)
    config = SimpleNamespace(
        speculative_config=None,
        model_config=SimpleNamespace(architectures=[], max_model_len=max_model_len),
        additional_config={},
    )
    monkeypatch.setattr(attention, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(attention, "get_chip_version", lambda: chip)
    return attention.PallasBatchedRPAAttentionBackendImpl().decode_lanes


def test_only_layers_with_tuned_tiles_look_up_the_lanes(monkeypatch):
    # The lanes table's own shape: another model keeps the kernel default.
    assert _lanes_after_init(monkeypatch, tuned_tiles=False) is None
    assert _lanes_after_init(monkeypatch, tuned_tiles=True) == 4
    # The layer looks the table up for its own chip.
    assert _lanes_after_init(monkeypatch, tuned_tiles=True, chip=V6E) is None


def test_dcp_kernels_take_the_tuned_tiles(monkeypatch):
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn: fn)
    kw = dict(
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=4,
        cp_rank=0,
        decode_query_size=4,
    )
    default = cp_attention.build_dcp_kernels(**kw)
    tuned = cp_attention.build_dcp_kernels(tuned_tiles=True, **kw)
    for off, on in zip(default, tuned, strict=True):
        assert off is not on
        assert off.keywords["tuned_tiles"] is False
        assert on.keywords["tuned_tiles"] is True

    seen = []

    def fake_wrapper(**kwargs):
        seen.append(kwargs)
        return None, None, None

    monkeypatch.setattr(
        cp_attention._batched_rpa_wrapper, "ragged_paged_attention", fake_wrapper
    )
    for kernel in tuned:
        kernel(*([None] * 11))
    # Both passes, the cache pass and the new-token pass, hand the tuned tiles
    # (and not the skip) to the wrapper.
    scope = cp_attention._batched_rpa_configs.AttentionScope
    assert [
        (c["attention_scope"], c["tuned_tiles"], c["skip_empty_mixed"]) for c in seen
    ] == [
        (scope.CACHE_ONLY, True, False),
        (scope.NEW_TOKENS_ONLY, True, False),
    ]


def test_dcp_kernels_take_the_skip_on_its_own(monkeypatch):
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn: fn)
    kw = dict(
        sliding_window=None,
        sm_scale=1.0,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=4,
        cp_rank=0,
        decode_query_size=4,
    )
    default = cp_attention.build_dcp_kernels(**kw)
    skip = cp_attention.build_dcp_kernels(skip_empty_mixed=True, **kw)
    for off, on in zip(default, skip, strict=True):
        assert off is not on
        assert off.keywords["skip_empty_mixed"] is False
        assert on.keywords["skip_empty_mixed"] is True
        assert on.keywords["tuned_tiles"] is False

    seen = []

    def fake_wrapper(**kwargs):
        seen.append(kwargs)
        return None, None, None

    monkeypatch.setattr(
        cp_attention._batched_rpa_wrapper, "ragged_paged_attention", fake_wrapper
    )
    for kernel in skip:
        kernel(*([None] * 11))
    # Both passes hand the skip, and not the tuned tiles, to the wrapper.
    assert [(c["skip_empty_mixed"], c["tuned_tiles"]) for c in seen] == [
        (True, False),
        (True, False),
    ]


# --- the batched RPA kernel builder --------------------------------------------

_MESH = object()


@pytest.fixture
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


class _FakeOp:
    def register_fake(self, _fake_impl):
        pass


@pytest.fixture
def registered(monkeypatch):
    """A fresh kernel registry; records every (op_name, wrapped_fn) that
    attention registers through its jax_op."""
    monkeypatch.setattr(attention.PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (_MESH, _MESH, None)),
    )
    out = []

    def fake_jax_op(name, wrapped_fn, **_kwargs):
        out.append((name, wrapped_fn))
        return _FakeOp()

    real = attention.pallas
    monkeypatch.setattr(
        attention, "pallas", SimpleNamespace(jax_op=fake_jax_op, pallas=real.pallas)
    )
    return out


def _build(**layer):
    impl = attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
    )
    for name, value in layer.items():
        setattr(impl, name, value)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
    )
    with set_vllm_model_wrapper_context(
        mesh=SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1}),
        vllm_config=vllm_config,
    ):
        return impl._build_rpa_kernel(None, None, None)


def _wrapper_kwargs(monkeypatch, entry_fn):
    """The keyword arguments a built entry's rpa_func gives the batched RPA
    wrapper (with `_pallas_rpa_kernel_impl` stubbed). The builder binds the
    q/k/v scales as keywords, so the entry takes 9 positional arguments."""
    seen = {}

    def fake_impl(*_args, rpa_func, **_kwargs):
        seen["rpa_func"] = rpa_func
        return None, None

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", fake_impl)
    args = [None] * 9
    args[1] = jnp.zeros((4, 2, 128))
    entry_fn(*args)
    fn = seen["rpa_func"]
    if fn is attention.ragged_paged_attention_batched:
        return {}
    assert fn.func is attention.ragged_paged_attention_batched
    return dict(fn.keywords)


@pytest.mark.parametrize(
    "setting, value, wrapper_kwarg",
    [
        ("decode_lanes", 4, ("decode_batch_size", 4)),
        ("tuned_tiles", True, ("tuned_tiles", True)),
        ("skip_empty_mixed", True, ("skip_empty_mixed", True)),
    ],
)
def test_the_builder_carries_the_layer_setting_to_the_kernel(
    monkeypatch, registered, cpu_vllm_config_context, setting, value, wrapper_kwarg
):
    """_build_rpa_kernel binds the layer's decode_lanes, tuned_tiles and
    skip_empty_mixed into the batched entry it registers, keys the registry on
    each (a layer with the setting gets its own op), and the entry hands it to
    the batched RPA wrapper."""
    default_op = _build()
    setting_op = _build(**{setting: value})
    assert setting_op is not default_op
    rpa = [fn for name, fn in registered if name.startswith("pallas::rpa_kernel")]
    assert len(rpa) == 2
    default_fn, setting_fn = rpa
    assert setting_fn.keywords[setting] == value
    assert default_fn.keywords[setting] != value
    name, kernel_value = wrapper_kwarg
    assert name not in _wrapper_kwargs(monkeypatch, default_fn)
    assert _wrapper_kwargs(monkeypatch, setting_fn)[name] == kernel_value


def test_a_local_entry_takes_none_of_the_batched_settings(
    registered, cpu_vllm_config_context
):
    """A tp=1 draft's local entry has none of the three arguments, so the
    builder binds none of them, whatever the layer carries."""
    _build(
        decode_lanes=4,
        tuned_tiles=True,
        skip_empty_mixed=True,
        _kernel_entry=attention._pallas_rpa_kernel_local,
        _kernel_op_prefix="pallas::rpa_kernel_local",
    )
    ((_, fn),) = registered
    assert not {"decode_lanes", "tuned_tiles", "skip_empty_mixed"} & set(fn.keywords)
