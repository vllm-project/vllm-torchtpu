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
"""Which batched RPA kernels take the decode levers: the static fp8 decode
operands ("qks", "pvs") and the base-2 decode softmax."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, flash_attention

pytestmark = pytest.mark.cpu_test


def _serve(**kw):
    base = dict(
        num_seqs=8,
        page_size=128,
        total_q_tokens=64,
        num_page_indices=64,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
    )
    base.update(kw)
    return configs.ServingConfigs(**base)


def _cfgs(mode, soft_cap=None, **serve_kw):
    return SimpleNamespace(
        mode=mode,
        serve=_serve(**serve_kw),
        model=SimpleNamespace(soft_cap=soft_cap),
    )


DECODE, MIXED = configs.RpaCase.DECODE, configs.RpaCase.MIXED


@pytest.mark.parametrize(
    "mode, serve_kw, expected",
    [
        (DECODE, {"native_fp8": "pvs"}, True),
        (DECODE, {"native_fp8": "pv"}, False),
        (DECODE, {"native_fp8_prefill": "pvs"}, False),
        (MIXED, {"native_fp8_prefill": "pvs"}, True),
        (MIXED, {"native_fp8": "pvs"}, False),
        (DECODE, {"native_fp8": "pvs", "dtype_kv": jnp.bfloat16}, False),
        (DECODE, {"native_fp8": "pvs", "per_token_scale": True}, False),
    ],
)
def test_static_pv_per_kernel(mode, serve_kw, expected):
    assert flash_attention.static_pv(_cfgs(mode, **serve_kw)) is expected


@pytest.mark.parametrize(
    "mode, soft_cap, serve_kw, expected",
    [
        (DECODE, None, {"decode_exp2": True}, True),
        (DECODE, None, {}, False),
        (MIXED, None, {"decode_exp2": True}, False),
        (DECODE, 30.0, {"decode_exp2": True}, False),
        (DECODE, None, {"decode_exp2": True, "return_lse": True}, False),
    ],
)
def test_decode_exp2_gating(mode, soft_cap, serve_kw, expected):
    assert flash_attention.decode_exp2(_cfgs(mode, soft_cap, **serve_kw)) is expected


def test_p_prescaled_needs_exp2_and_static_pv():
    both = _cfgs(DECODE, decode_exp2=True, native_fp8="pvs")
    assert flash_attention.p_prescaled(both)
    assert not flash_attention.p_prescaled(_cfgs(DECODE, native_fp8="pvs"))
    assert not flash_attention.p_prescaled(_cfgs(DECODE, decode_exp2=True))


def _wrapper_serving_config(monkeypatch, **kwargs):
    """The ServingConfigs keyword arguments the wrapper resolves for one call."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    class _Stop(Exception):
        pass

    seen = {}

    def fake_serving_configs(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(wrapper.configs, "ServingConfigs", fake_serving_configs)
    q = jnp.zeros((8, 2, 128), jnp.bfloat16)
    with pytest.raises(_Stop):
        wrapper.ragged_paged_attention(
            q,
            jnp.zeros((8, 1, 128), jnp.bfloat16),
            jnp.zeros((8, 1, 128), jnp.bfloat16),
            jnp.zeros((4, 16, 2, 2, 128), jnp.bfloat16),
            jnp.zeros((2,), jnp.int32),
            jnp.zeros((8,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            vmem_limit_bytes=1 << 26,
            **kwargs,
        )
    return seen


def test_wrapper_defaults_are_off(monkeypatch):
    seen = _wrapper_serving_config(monkeypatch)
    assert seen["native_fp8"] == ""
    assert seen["native_fp8_prefill"] == ""
    assert seen["decode_exp2"] is False


def test_wrapper_accepts_static_decode_modes(monkeypatch):
    seen = _wrapper_serving_config(
        monkeypatch, native_fp8="pvs", native_fp8_prefill="pvs", decode_exp2=True
    )
    assert seen["native_fp8"] == "pvs"
    assert seen["native_fp8_prefill"] == "pvs"
    assert seen["decode_exp2"] is True


@pytest.mark.parametrize("bad", ["pvx", "pv2s", "spv"])
def test_wrapper_rejects_unknown_decode_modes(monkeypatch, bad):
    with pytest.raises(ValueError, match="native_fp8 must match"):
        _wrapper_serving_config(monkeypatch, native_fp8=bad)
