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
"""Native fp8 modes of the batched RPA kernels: typed, validated, and per kernel.

A mode that cannot run raises instead of falling back to the dequantizing
kernel: native fp8 needs an fp8 KV cache, native QK cannot be combined with a
static q scale (that path already casts q to the KV dtype), SPLIT operands
exist only in the DECODE kernel, and a STATIC PV cannot take per-token V
scales.
"""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, flash_attention

_ROW, _SPLIT, _STATIC = (
    configs.Fp8Operand.ROW,
    configs.Fp8Operand.SPLIT,
    configs.Fp8Operand.STATIC,
)


def _serve(dtype_kv, native_fp8=configs.NATIVE_FP8_OFF, **kwargs):
    return configs.ServingConfigs(
        num_seqs=8,
        page_size=128,
        total_q_tokens=16,
        num_page_indices=64,
        dtype_q=jnp.bfloat16,
        dtype_kv=dtype_kv,
        dtype_out=jnp.bfloat16,
        native_fp8=native_fp8,
        **kwargs,
    )


@pytest.mark.parametrize(
    "native_fp8",
    [
        configs.NativeFp8(pv=_ROW),
        configs.NativeFp8(qk=_ROW),
        configs.NativeFp8(qk=_SPLIT, pv=_SPLIT),
        configs.NativeFp8(pv=_STATIC),
    ],
)
def test_native_fp8_needs_an_fp8_kv_cache(native_fp8):
    with pytest.raises(ValueError, match="fp8 KV"):
        _serve(jnp.bfloat16, native_fp8)
    # SPLIT has no room in e5m2 (test_batched_rpa_native_fp8_admission.py).
    split = configs.Fp8Operand.SPLIT in (native_fp8.qk, native_fp8.pv)
    for dtype_kv in (jnp.float8_e4m3fn,) + (() if split else (jnp.float8_e5m2,)):
        assert _serve(dtype_kv, native_fp8).native_fp8 == native_fp8


def test_prefill_modes_need_an_fp8_kv_cache_too():
    with pytest.raises(ValueError, match="native_fp8_prefill=pvs .* fp8 KV"):
        _serve(jnp.bfloat16, native_fp8_prefill=configs.NativeFp8(pv=_STATIC))


@pytest.mark.parametrize("name", ["native_fp8", "native_fp8_prefill"])
def test_native_qk_rejects_a_static_q_scale(name):
    with pytest.raises(ValueError, match="static q scale"):
        _serve(jnp.float8_e4m3fn, **{name: configs.NativeFp8(qk=_ROW)}, scale_q=2.0)
    _serve(jnp.float8_e4m3fn, **{name: configs.NativeFp8(pv=_ROW)}, scale_q=2.0)


def test_split_operands_are_decode_only():
    split = configs.NativeFp8(qk=_SPLIT)
    assert _serve(jnp.float8_e4m3fn, split).native_fp8 == split
    with pytest.raises(ValueError, match="DECODE only"):
        _serve(jnp.float8_e4m3fn, native_fp8_prefill=split)


@pytest.mark.parametrize("name", ["native_fp8", "native_fp8_prefill"])
def test_static_pv_rejects_per_token_v_scales(name):
    with pytest.raises(ValueError, match=f"{name}.pv STATIC with per-token V scales"):
        _serve(
            jnp.float8_e4m3fn,
            **{name: configs.NativeFp8(pv=_STATIC)},
            per_token_scale=True,
            per_token_scale_dtype=jnp.float32,
        )


@pytest.mark.parametrize(
    ("modes", "short"),
    [
        (configs.NativeFp8(), "off"),
        (configs.NativeFp8(pv=_STATIC), "pvs"),
        (configs.NativeFp8(qk=_SPLIT, pv=_SPLIT), "qk2pv2"),
        (configs.NativeFp8(qk=_ROW, pv=_ROW), "qkpv"),
    ],
)
def test_short_form_for_logs(modes, short):
    assert str(modes) == short


def test_off_is_the_default_and_runs_on_any_kv_cache():
    serve = _serve(jnp.bfloat16, configs.NATIVE_FP8_OFF)
    assert serve.native_fp8 == configs.NativeFp8()
    assert not serve.native_fp8.any()


@pytest.mark.parametrize("mode", [configs.RpaCase.DECODE, configs.RpaCase.MIXED])
def test_each_kernel_takes_its_own_modes(mode):
    native_fp8 = configs.NativeFp8(qk=_SPLIT, pv=_ROW)
    native_fp8_prefill = configs.NativeFp8(pv=_STATIC)
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=16, bq_c_sz=16, bkv_sz=256, batch_size=1),
        model=configs.ModelConfigs(
            num_q_heads=1, num_kv_heads=1, head_dim=256, mask_value=-1e9
        ),
        serve=_serve(
            jnp.float8_e4m3fn, native_fp8, native_fp8_prefill=native_fp8_prefill
        ),
        mode=mode,
        vmem_limit_bytes=1 << 26,
    )
    decode = mode == configs.RpaCase.DECODE
    expected = native_fp8 if decode else native_fp8_prefill
    assert flash_attention._native_fp8(cfgs) == expected
    assert flash_attention.static_pv(cfgs) is not decode
