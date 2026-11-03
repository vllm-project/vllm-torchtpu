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
"""The fused query preparation retains BF16 rounding and inactive-row semantics."""

import jax
import numpy as np
import pytest
import torch


@pytest.mark.tpu_test
@torch.inference_mode()
def test_query_norm_rope_matches_reference_bits():
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.qwen4_exp.qsa import gemma_norm, main_rope
    from vllm_torchtpu.kernels.qwen4_exp.query_prepare import prepare_query_values

    def build(fused):
        def core(
            qgate: jax.Array, norm: jax.Array, positions: jax.Array, trig: jax.Array
        ) -> jax.Array:
            shaped = qgate.reshape(16, 24, 512)
            if fused:
                return prepare_query_values(shaped, norm, positions, trig, 1e-6, 64)
            return main_rope(
                gemma_norm(shaped[..., :256], norm, 1e-6), positions, trig, 64
            )

        op = pallas.jax_op(f"qsa_query_prepare_test::bits_{fused}", core)

        @op.register_fake
        def fake(qgate, norm, positions, trig):
            return qgate.new_empty((16, 24, 256))

        return torch.compile(op, backend="tpu", fullgraph=True, dynamic=False)

    reference, fused = build(False), build(True)
    generator = torch.Generator().manual_seed(5943)
    angles = torch.outer(
        torch.arange(67584).float(), 1e7 ** (-torch.arange(32).float() / 32)
    )
    trig = torch.nn.functional.pad(
        torch.cat((angles.cos(), angles.sin()), 1).bfloat16(), (0, 64)
    ).to("tpu")
    positions = torch.tensor(
        [
            0,
            1,
            3,
            4,
            31,
            32,
            127,
            128,
            1023,
            1024,
            32767,
            32768,
            65535,
            65536,
            65537,
            67583,
        ],
        dtype=torch.int32,
    ).to("tpu")
    for case in ("ordinary", "zero", "large", "nan_padding", "holes"):
        q = torch.randn((16, 12288), generator=generator).bfloat16()
        norm = (torch.randn((256,), generator=generator) * 0.2).bfloat16()
        if case == "zero":
            q.zero_()
        if case == "large":
            q.mul_(1e8)
        if case == "nan_padding":
            q[8:] = float("nan")
        if case == "holes":
            q[::2] = float("nan")
        args = (q.to("tpu"), norm.to("tpu"), positions, trig)
        expected = reference(*args).cpu().view(torch.uint16).numpy()
        actual = fused(*args).cpu().view(torch.uint16).numpy()
        np.testing.assert_array_equal(actual, expected, err_msg=case)
