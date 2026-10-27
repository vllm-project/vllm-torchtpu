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
"""Reduced-precision attention modes are target-only.

A target attention layer's batched RPA call carries the layer's modes
(`native_fp8`, `native_fp8_prefill`, `decode_exp2`); a speculative draft's
layer carries none (the full-precision path the draft ships with). With
every mode off nothing changes: the draft binds nothing new and shares the
target's op as before.
"""

import functools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

import vllm_torchtpu.layers.adapter.attention as attention
from vllm_torchtpu.layers.adapter import cp_attention, native_fp8_scope
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

PVS = native_fp8_scope.AttentionModes(
    native_fp8="pvs", native_fp8_prefill="pvs", decode_exp2=True
)
PVS_KWARGS = {"native_fp8": "pvs", "native_fp8_prefill": "pvs", "decode_exp2": True}
_MESH = object()


@pytest.fixture
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _batched_impl():
    return PallasBatchedRPAAttentionBackendImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
    )


def _mesh():
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})


def _vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
    )


class _FakeOp:
    def register_fake(self, _fake_impl):
        pass


@pytest.fixture
def registered(monkeypatch):
    """Fresh kernel registry; records every (op_name, wrapped_fn) registered."""
    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (_MESH, _MESH, None)),
    )
    out = []

    def fake_jax_op(name, wrapped_fn, **_kwargs):
        out.append((name, wrapped_fn))
        return _FakeOp()

    monkeypatch.setattr(attention.pallas, "jax_op", fake_jax_op)
    return out


def _rpa_call_kwargs(monkeypatch, entry_fn, n_args):
    """Run a kernel entry with `_pallas_rpa_kernel_impl` stubbed; return the
    keyword arguments its `rpa_func` would pass to the batched RPA wrapper."""
    seen = {}

    def fake_impl(*_args, rpa_func, **_kwargs):
        seen["rpa_func"] = rpa_func
        return None, None

    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", fake_impl)
    query = jnp.zeros((4, 2, 128))
    args = [None] * n_args
    args[1] = query
    entry_fn(*args)
    fn = seen["rpa_func"]
    if fn is attention.ragged_paged_attention_batched:
        return {}
    assert fn.func is attention.ragged_paged_attention_batched
    return dict(fn.keywords)


# --- the scope helpers --------------------------------------------------------


def test_scope_helpers():
    assert native_fp8_scope.OFF.any() is False
    assert PVS.any() is True
    assert native_fp8_scope.effective(PVS, is_draft=True) == native_fp8_scope.OFF
    assert native_fp8_scope.effective(PVS, is_draft=False) == PVS
    assert native_fp8_scope.kernel_kwargs(native_fp8_scope.OFF) == {}
    assert native_fp8_scope.kernel_kwargs(PVS) == PVS_KWARGS
    line = native_fp8_scope.scope_line(PVS, 60, 1)
    assert "target decode=pvs prefill=pvs exp2=on, draft=off" in line
    assert "attention layers: 60 target, 1 draft" in line
    assert "decode=off prefill=off exp2=off" in native_fp8_scope.scope_line(
        native_fp8_scope.OFF, 60, 0
    )


# --- the kernel entry ----------------------------------------------------------


def test_entry_passes_the_modes_it_is_given(monkeypatch):
    common = dict(
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=attention.batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    entry = attention._pallas_rpa_kernel_batched
    off = _rpa_call_kwargs(monkeypatch, functools.partial(entry, **common), 12)
    assert off == {}
    on = _rpa_call_kwargs(
        monkeypatch, functools.partial(entry, attention_modes=PVS, **common), 12
    )
    assert on == PVS_KWARGS


# --- the kernel builders -------------------------------------------------------


def test_builder_gives_a_draft_its_own_full_precision_op(
    monkeypatch, registered, cpu_vllm_config_context
):
    target, draft = _batched_impl(), _batched_impl()
    target.attention_modes = draft.attention_modes = PVS
    draft.native_fp8_draft = True
    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        t_op = target._build_rpa_kernel(None, None, None)
        d_op = draft._build_rpa_kernel(None, None, None)
    assert t_op is not d_op
    rpa = [(n, fn) for n, fn in registered if n.startswith("pallas::rpa_kernel")]
    assert len(rpa) == 2, [n for n, _ in rpa]
    (_, t_fn), (_, d_fn) = rpa
    assert t_fn.keywords["attention_modes"] == PVS
    assert "attention_modes" not in d_fn.keywords
    # End to end through the bound entries. The builder binds q/k/v scales as
    # keywords: 9 positional arguments.
    assert _rpa_call_kwargs(monkeypatch, t_fn, 9) == PVS_KWARGS
    assert _rpa_call_kwargs(monkeypatch, d_fn, 9) == {}


def test_builder_unchanged_with_the_modes_off(
    monkeypatch, registered, cpu_vllm_config_context
):
    target, draft = _batched_impl(), _batched_impl()
    draft.native_fp8_draft = True
    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        t_op = target._build_rpa_kernel(None, None, None)
        d_op = draft._build_rpa_kernel(None, None, None)
    # As before the scoping: one shared op, nothing new bound.
    assert t_op is d_op
    rpa = [(n, fn) for n, fn in registered if n.startswith("pallas::rpa_kernel")]
    assert len(rpa) == 1
    assert "attention_modes" not in rpa[0][1].keywords


def test_dcp_kernels_take_the_modes(monkeypatch):
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn, **k: fn)
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
    off_cache, off_new = cp_attention.build_dcp_kernels(**kw)
    on_cache, on_new = cp_attention.build_dcp_kernels(attention_modes=PVS, **kw)
    assert off_cache is not on_cache and off_new is not on_new
    assert on_new.keywords["attention_modes"] == PVS
    assert off_new.keywords["attention_modes"] == native_fp8_scope.OFF

    calls = []

    def fake_rpa(**k):
        calls.append(k)
        return None, None, None

    monkeypatch.setattr(
        cp_attention._batched_rpa_wrapper, "ragged_paged_attention", fake_rpa
    )
    args = [jnp.zeros((4, 2, 128))] * 11
    off_new(*args)
    on_new(*args)
    assert not any(k in calls[0] for k in PVS_KWARGS)
    assert {k: calls[1][k] for k in PVS_KWARGS} == PVS_KWARGS


# --- the runner marks the draft before building its kernels --------------------


def _runner(layers, draft_names):
    runner = MagicMock(spec=TPUModelRunner)
    runner._attention_kernels_initialized = False
    runner.model = None
    runner.mesh = MagicMock()
    runner.vllm_config = MagicMock()
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=2)
    runner.reorder_batch_threshold = 1
    runner.drafter = SimpleNamespace(_draft_attn_layer_names=set(draft_names))
    runner.speculative_config = SimpleNamespace(draft_tensor_parallel_size=2)
    runner._initialize_attention_kernels = (
        TPUModelRunner._initialize_attention_kernels.__get__(runner)
    )
    return runner


def test_runner_marks_draft_layers_before_initialize_kernel():
    seen = {}
    layers = {}
    for name in ("model.layers.3.self_attn.attn", "mtp.layers.0.self_attn.attn"):
        impl = MagicMock(spec=PallasAttentionBackendImpl)
        impl.decode_query_size = 4
        impl.native_fp8_draft = False
        impl.attention_modes = PVS

        def record(_layer, impl=impl, name=name):
            seen[name] = impl.native_fp8_draft

        impl.initialize_kernel.side_effect = record
        layers[name] = SimpleNamespace(impl=impl)
    runner = _runner(layers, ["mtp.layers.0.self_attn.attn"])
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config",
            return_value=layers,
        ),
        patch("vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context"),
        patch("vllm_torchtpu.runner.tpu_runner.logger") as log,
    ):
        runner._initialize_attention_kernels()
    assert seen == {
        "model.layers.3.self_attn.attn": False,
        "mtp.layers.0.self_attn.attn": True,
    }
    lines = [c.args[0] for c in log.info.call_args_list if c.args]
    assert native_fp8_scope.scope_line(PVS, 1, 1) in lines
