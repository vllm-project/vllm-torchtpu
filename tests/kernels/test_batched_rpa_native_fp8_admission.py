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
"""Where the batched RPA kernel admits native fp8 matmuls, what the decode
VMEM model counts for them, and that the kernels run what they admit.

SPLIT keeps a residual half scaled by `_SPLIT_LO`, which only e4m3 has the
range for, so SPLIT with an e5m2 KV cache is refused. Native fp8 needs fp8
matmuls on the MXU, so a chip without them is refused. A SPLIT operand
doubles the decode matmul's LHS rows and score tile, which the decode VMEM
model counts. Each operand an admitted mode turns on enters its decode matmul
in the KV dtype, e4m3 or e5m2, and each operand a prefill mode turns on enters
its MIXED kernel matmul in the KV dtype.

The STATIC operands and the prefill modes are tested on e4m3 only, so they
refuse an e5m2 KV cache; e5m2 runs only the DECODE kernel's ROW operands.
"""

from types import SimpleNamespace

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import pytest

from tests.kernels.experimental.batched_rpa.rpa_test_utils import rpa_kernel_dots
from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_ROW, _SPLIT = configs.Fp8Operand.ROW, configs.Fp8Operand.SPLIT
_STATIC = configs.Fp8Operand.STATIC


def _serve(dtype_kv, native_fp8, **kwargs):
    return configs.ServingConfigs(
        num_seqs=16,
        page_size=128,
        total_q_tokens=64,
        num_page_indices=16 * 800,
        dtype_q=jnp.bfloat16,
        dtype_kv=dtype_kv,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_query_size=4,
        native_fp8=native_fp8,
        **kwargs,
    )


@pytest.mark.parametrize(
    "native_fp8",
    [
        configs.NativeFp8(qk=_SPLIT),
        configs.NativeFp8(pv=_SPLIT),
        configs.NativeFp8(qk=_SPLIT, pv=_SPLIT),
    ],
)
def test_split_refuses_an_e5m2_kv_cache(native_fp8):
    with pytest.raises(ValueError, match="e5m2"):
        _serve(jnp.float8_e5m2, native_fp8)
    assert _serve(jnp.float8_e4m3fn, native_fp8).native_fp8 == native_fp8


def test_row_operands_run_on_an_e5m2_kv_cache():
    modes = configs.NativeFp8(qk=_ROW, pv=_ROW)
    assert _serve(jnp.float8_e5m2, modes).native_fp8 == modes


@pytest.mark.parametrize(
    ("name", "modes"),
    [
        ("native_fp8", configs.NativeFp8(pv=_STATIC)),
        ("native_fp8", configs.NativeFp8(qk=_STATIC)),
        ("native_fp8", configs.NativeFp8(qk=_STATIC, pv=_ROW)),
        ("native_fp8", configs.NativeFp8(qk=_STATIC, pv=_STATIC)),
        ("native_fp8_prefill", configs.NativeFp8(pv=_ROW)),
        ("native_fp8_prefill", configs.NativeFp8(qk=_ROW)),
        ("native_fp8_prefill", configs.NativeFp8(qk=_ROW, pv=_ROW)),
        ("native_fp8_prefill", configs.NativeFp8(pv=_STATIC)),
        ("native_fp8_prefill", configs.NativeFp8(qk=_STATIC)),
        ("native_fp8_prefill", configs.NativeFp8(qk=_STATIC, pv=_STATIC)),
    ],
    ids=str,
)
def test_static_and_prefill_operands_refuse_an_e5m2_kv_cache(name, modes):
    def serve(dtype_kv):
        if name == "native_fp8":
            return _serve(dtype_kv, modes)
        return _serve(dtype_kv, configs.NATIVE_FP8_OFF, native_fp8_prefill=modes)

    with pytest.raises(ValueError, match=f"{name}={modes} with an fp8_e5m2"):
        serve(jnp.float8_e5m2)
    assert getattr(serve(jnp.float8_e4m3fn), name) == modes


@pytest.mark.parametrize("name", ["native_fp8", "native_fp8_prefill"])
@pytest.mark.parametrize(
    ("chip", "admitted"),
    [(pltpu.ChipVersion.TPU_7X, True), (pltpu.ChipVersion.TPU_V5E, False)],
)
def test_native_fp8_needs_fp8_matmuls_on_the_mxu(monkeypatch, chip, admitted, name):
    info = pltpu.get_tpu_info_for_chip(chip, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)
    monkeypatch.setattr(
        wrapper, "logger", SimpleNamespace(warning_once=lambda *args: None)
    )
    tokens, num_seqs, pages_per_seq = 16, 8, 16
    kv_shape = wrapper.get_kv_cache_shape(
        num_seqs * pages_per_seq,
        128,
        1,
        256,
        jnp.float8_e4m3fn,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )

    def spec(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype)

    def trace():
        # The wrapper is jitted: trace afresh for this chip's info.
        jax.clear_caches()
        jax.make_jaxpr(
            lambda *a: wrapper.ragged_paged_attention(
                *a,
                sm_scale=1 / 16,
                k_scale=1.0,
                v_scale=1.0,
                kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                decode_query_size=4,
                **{name: configs.NativeFp8(pv=_ROW)},
            )
        )(
            spec((tokens, 4, 256), jnp.bfloat16),
            spec((tokens, 1, 256), jnp.bfloat16),
            spec((tokens, 1, 256), jnp.bfloat16),
            spec(kv_shape, jnp.float8_e4m3fn),
            spec((num_seqs,), jnp.int32),
            spec((num_seqs * pages_per_seq,), jnp.int32),
            spec((num_seqs + 1,), jnp.int32),
            spec((3,), jnp.int32),
        )

    if admitted:
        trace()
    else:
        with pytest.raises(ValueError, match=f"{name}=.*fp8 on the MXU"):
            trace()


def test_split_counts_its_doubled_lhs_in_the_decode_vmem_model(monkeypatch):
    """16 q / 1 KV head of 256, fp8 KV, the full 64 MiB of TPU7x VMEM: with
    SPLIT operands the decode KV block is one page step smaller; the prefill
    tiles, which never use SPLIT, do not change."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)
    model = configs.ModelConfigs(
        num_q_heads=16, num_kv_heads=1, head_dim=256, mask_value=-1e30
    )

    def blocks(native_fp8):
        return wrapper.calculate_block_sizes(
            model,
            _serve(jnp.float8_e4m3fn, native_fp8),
            64 * 1024 * 1024,
            decode_query_size=4,
        )

    off_decode, off_prefill = blocks(configs.NATIVE_FP8_OFF)
    split_decode, split_prefill = blocks(configs.NativeFp8(qk=_SPLIT, pv=_SPLIT))
    row_decode, _ = blocks(configs.NativeFp8(qk=_ROW, pv=_ROW))
    assert split_decode.bkv_sz < off_decode.bkv_sz
    assert split_prefill == off_prefill
    assert row_decode == off_decode


def _matmul_lhs(monkeypatch, dtype_kv, prefix, **modes):
    """The (dtype, rows) of the LHS of the QK and PV matmuls of the RPA kernels
    whose names start with `prefix` ("RPAd" for DECODE, "RPAm" for MIXED) on
    TPU7x at the Qwen3.5 TP8 geometry (4 q / 1 KV head of 256, 4 query tokens
    a decode request, the tuned tiles), with the wrapper's native fp8 `modes`.
    In this layout QK contracts axis 1 of K ([., head_dim, S]) and its LHS is
    q; PV contracts axis 2 of V and its LHS is the softmax weights."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)
    monkeypatch.setattr(
        wrapper, "logger", SimpleNamespace(warning_once=lambda *args: None)
    )
    tokens, num_seqs, pages_per_seq = 16, 8, 32
    kv_shape = wrapper.get_kv_cache_shape(
        num_seqs * pages_per_seq,
        128,
        1,
        256,
        dtype_kv,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    shapes = [
        ((tokens, 4, 256), jnp.bfloat16),
        ((tokens, 1, 256), jnp.bfloat16),
        ((tokens, 1, 256), jnp.bfloat16),
        (kv_shape, dtype_kv),
        ((num_seqs,), jnp.int32),
        ((num_seqs * pages_per_seq,), jnp.int32),
        ((num_seqs + 1,), jnp.int32),
        ((3,), jnp.int32),
    ]
    # The wrapper is jitted: trace afresh with this chip's info.
    jax.clear_caches()
    dots = rpa_kernel_dots(
        lambda *a: wrapper.ragged_paged_attention(
            *a,
            sm_scale=1 / 16,
            k_scale=1.0,
            v_scale=1.0,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            decode_query_size=4,
            tuned_tiles=True,
            skip_empty_mixed=True,
            **modes,
        ),
        *(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in shapes),
        prefix=prefix,
    )
    lhs = {"qk": set(), "pv": set()}
    for eqn in dots:
        (_, rhs_contracting), _ = eqn.params["dimension_numbers"]
        aval = eqn.invars[0].aval
        kind = "qk" if rhs_contracting[0] == 1 else "pv"
        lhs[kind].add((jnp.dtype(aval.dtype), aval.shape[-2]))
    return lhs


_E4M3, _E5M2 = jnp.float8_e4m3fn, jnp.float8_e5m2


@pytest.mark.parametrize(
    ("dtype_kv", "native_fp8", "decode_exp2"),
    [
        pytest.param(_E4M3, configs.NativeFp8(pv=_ROW), False, id="e4m3-pv"),
        pytest.param(_E4M3, configs.NativeFp8(qk=_ROW), False, id="e4m3-qk"),
        pytest.param(
            _E4M3, configs.NativeFp8(qk=_ROW, pv=_ROW), False, id="e4m3-qk_pv"
        ),
        pytest.param(_E4M3, configs.NativeFp8(pv=_SPLIT), False, id="e4m3-pv_split"),
        pytest.param(
            _E4M3,
            configs.NativeFp8(qk=_SPLIT, pv=_SPLIT),
            False,
            id="e4m3-qk_split_pv_split",
        ),
        pytest.param(_E4M3, configs.NativeFp8(pv=_STATIC), False, id="e4m3-pv_static"),
        # The weights leave the base-2 exponential already times P_SCALE.
        pytest.param(
            _E4M3, configs.NativeFp8(pv=_STATIC), True, id="e4m3-pv_static_exp2"
        ),
        pytest.param(
            _E4M3,
            configs.NativeFp8(qk=_STATIC, pv=_STATIC),
            False,
            id="e4m3-qk_static_pv_static",
        ),
        pytest.param(_E5M2, configs.NativeFp8(pv=_ROW), False, id="e5m2-pv"),
        pytest.param(
            _E5M2, configs.NativeFp8(qk=_ROW, pv=_ROW), False, id="e5m2-qk_pv"
        ),
    ],
)
def test_the_decode_kernel_runs_each_native_operand_in_fp8(
    monkeypatch, dtype_kv, native_fp8, decode_exp2
):
    """Each operand the modes turn on enters its DECODE matmul in the KV dtype
    (q for QK, the softmax weights for PV), with twice the rows for SPLIT; an
    operand left off enters as in the dequantizing kernel. The TPU tests'
    error bounds alone do not show this: the dequantizing kernel is within
    them too."""
    exp2 = {"decode_exp2": decode_exp2}
    off = _matmul_lhs(monkeypatch, dtype_kv, "RPAd", **exp2)
    got = _matmul_lhs(monkeypatch, dtype_kv, "RPAd", native_fp8=native_fp8, **exp2)
    for kind, operand in (("qk", native_fp8.qk), ("pv", native_fp8.pv)):
        ((off_dtype, off_rows),) = off[kind]
        assert off_dtype != jnp.dtype(dtype_kv)
        if operand is configs.Fp8Operand.OFF:
            expected = {(off_dtype, off_rows)}
        else:
            rows = 2 * off_rows if operand is _SPLIT else off_rows
            expected = {(jnp.dtype(dtype_kv), rows)}
        assert got[kind] == expected, (kind, got)


_PREFILL_MODES = {
    "pv": configs.NativeFp8(pv=_ROW),
    "pv_static": configs.NativeFp8(pv=_STATIC),
    "qk": configs.NativeFp8(qk=_ROW),
    "qk_pv": configs.NativeFp8(qk=_ROW, pv=_ROW),
    "qk_static": configs.NativeFp8(qk=_STATIC),
    "qk_static_pv_static": configs.NativeFp8(qk=_STATIC, pv=_STATIC),
}


@pytest.mark.parametrize("mode", list(_PREFILL_MODES))
def test_the_mixed_kernel_runs_each_native_operand_in_fp8(monkeypatch, mode):
    """The same for the MIXED kernel and `native_fp8_prefill` on an e4m3 KV
    cache, in the prefill TPU test's modes. The kernel can hold several QK
    and PV matmuls with different row counts; an operand turned on keeps each
    one's rows and enters it in the KV dtype."""
    modes = _PREFILL_MODES[mode]
    off = _matmul_lhs(monkeypatch, _E4M3, "RPAm")
    got = _matmul_lhs(monkeypatch, _E4M3, "RPAm", native_fp8_prefill=modes)
    for kind, operand in (("qk", modes.qk), ("pv", modes.pv)):
        assert off[kind]
        assert all(dtype != jnp.dtype(_E4M3) for dtype, _ in off[kind])
        if operand is configs.Fp8Operand.OFF:
            expected = off[kind]
        else:
            expected = {(jnp.dtype(_E4M3), rows) for _, rows in off[kind]}
        assert got[kind] == expected, (kind, got)
