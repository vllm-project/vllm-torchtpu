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
"""Tests for FP8 KV clamp fast-path, k_eq_v deduplication, and batched RPA packing."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper
from vllm_torchtpu.layers.core.quantization import (
    quantize_kv,
    static_per_tensor_quantize_tensor,
)


def test_static_per_tensor_quantize_unit_scale_bit_exact():
    torch.manual_seed(7)
    x = torch.randn(64, 256, dtype=torch.bfloat16) * 20.0
    x[0] = 0.0
    x[1, :4] = torch.tensor([448.0, -448.0, 900.0, -900.0], dtype=torch.bfloat16)

    q_fast = static_per_tensor_quantize_tensor(x, torch.float8_e4m3fn, scale=1.0)
    finfo = torch.finfo(torch.float8_e4m3fn)
    q_ref = torch.clamp(x.to(torch.float32) / 1.0, finfo.min, finfo.max).to(
        torch.float8_e4m3fn
    )
    assert torch.equal(q_fast.view(torch.uint8), q_ref.view(torch.uint8))


def test_quantize_kv_deduplicates_identical_key_and_value():
    k = torch.randn(16, 64, dtype=torch.bfloat16)
    qk, qv = quantize_kv(torch.float8_e4m3fn, k, k, k_scale=1.0, v_scale=1.0)
    assert qv is qk

    qk2, qv2 = quantize_kv(torch.float8_e4m3fn, k, k, k_scale=1.0, v_scale=0.5)
    assert qv2 is not None
    assert qv2 is not qk2


def test_batched_rpa_prepare_inputs_stack_and_skip_kv_update():
    cpu = jax.devices("cpu")[0]
    fake_tpu_info = SimpleNamespace(num_lanes=128, num_sublanes=8)
    with (
        jax.default_device(cpu),
        patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=fake_tpu_info),
    ):
        rng = np.random.default_rng(42)
        q = jnp.asarray(
            rng.standard_normal((16, 8, 256), dtype=np.float32), dtype=jnp.bfloat16
        )
        k = jnp.asarray(
            rng.standard_normal((16, 2, 256), dtype=np.float32), dtype=jnp.bfloat16
        )
        v = jnp.asarray(
            rng.standard_normal((16, 2, 256), dtype=np.float32), dtype=jnp.bfloat16
        )

        for layout in (
            configs.KVLayout.HEAD_ALONG_SUBLANE,
            configs.KVLayout.SEQ_ALONG_LANE,
        ):
            q_out, kv_out = wrapper.prepare_inputs(
                q,
                k,
                v,
                jnp.bfloat16,
                jnp.bfloat16,
                kv_layout=layout,
                page_size=128,
                skip_kv_update=False,
            )
            q_skip, kv_skip = wrapper.prepare_inputs(
                q,
                k,
                v,
                jnp.bfloat16,
                jnp.bfloat16,
                kv_layout=layout,
                page_size=128,
                skip_kv_update=True,
            )
            np.testing.assert_array_equal(
                np.asarray(q_out.astype(jnp.float32)),
                np.asarray(q_skip.astype(jnp.float32)),
            )
            assert kv_skip.shape == kv_out.shape
            assert kv_skip.dtype == kv_out.dtype
            assert float(jnp.max(jnp.abs(kv_skip.astype(jnp.float32)))) == 0.0

        # Verify jnp.stack([k, v], axis=2).reshape(...) is bit-identical to
        # jnp.concatenate([k, v], axis=-1).reshape(...) across arbitrary shapes.
        stacked = jnp.stack([k, v], axis=2).reshape(16, 4, 256)
        concat = jnp.concatenate([k, v], axis=-1).reshape(16, 4, 256)
        np.testing.assert_array_equal(
            np.asarray(stacked.astype(jnp.float32)),
            np.asarray(concat.astype(jnp.float32)),
        )
