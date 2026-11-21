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
"""Where the batched RPA kernel admits native fp8 matmuls, and what the decode
VMEM model counts for them.

SPLIT keeps a residual half scaled by `_SPLIT_LO`, which only e4m3 has the
range for, so SPLIT with an e5m2 KV cache is refused. Native fp8 needs fp8
matmuls on the MXU, so a chip without them is refused. A SPLIT operand
doubles the decode matmul's LHS rows and score tile, which the decode VMEM
model counts.
"""

from types import SimpleNamespace

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_ROW, _SPLIT = configs.Fp8Operand.ROW, configs.Fp8Operand.SPLIT


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
