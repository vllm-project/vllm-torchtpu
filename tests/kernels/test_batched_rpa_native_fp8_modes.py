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

A mode that cannot run raises instead of falling back to the dequantizing
"""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, flash_attention



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
    ],
)
def test_native_fp8_needs_an_fp8_kv_cache(native_fp8):
    with pytest.raises(ValueError, match="fp8 KV"):
        _serve(jnp.bfloat16, native_fp8)
    # SPLIT has no room in e5m2 (test_batched_rpa_native_fp8_admission.py).
    split = configs.Fp8Operand.SPLIT in (native_fp8.qk, native_fp8.pv)
    for dtype_kv in (jnp.float8_e4m3fn,) + (() if split else (jnp.float8_e5m2,)):
        assert _serve(dtype_kv, native_fp8).native_fp8 == native_fp8


    with pytest.raises(ValueError, match="static q scale"):


def test_off_is_the_default_and_runs_on_any_kv_cache():
    serve = _serve(jnp.bfloat16, configs.NATIVE_FP8_OFF)
    assert serve.native_fp8 == configs.NativeFp8()
    assert not serve.native_fp8.any()


@pytest.mark.parametrize("mode", [configs.RpaCase.DECODE, configs.RpaCase.MIXED])
    native_fp8 = configs.NativeFp8(qk=_SPLIT, pv=_ROW)
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=16, bq_c_sz=16, bkv_sz=256, batch_size=1),
        model=configs.ModelConfigs(
            num_q_heads=1, num_kv_heads=1, head_dim=256, mask_value=-1e9
        ),
        mode=mode,
        vmem_limit_bytes=1 << 26,
    )
    assert flash_attention._native_fp8(cfgs) == expected
