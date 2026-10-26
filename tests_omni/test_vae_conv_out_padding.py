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

import torch


def _assert_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    # A 128-channel conv reduces in a different order than a 3-channel conv.
    scale = ref.abs().max().item()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5 * scale)


def test_vae_conv_out_padding():
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    orig_forward = WanCausalConv3d.forward

    conv = WanCausalConv3d(in_channels=96, out_channels=3, kernel_size=3, padding=1)

    from vllm_torchtpu.omni.patches import _patch_wan_vae_conv_out_lane_padding

    _patch_wan_vae_conv_out_lane_padding()

    assert WanCausalConv3d.forward is not orig_forward

    x_even = torch.randn(1, 96, 2, 8, 8)
    x_odd = torch.randn(1, 96, 3, 8, 8)

    _assert_close(conv(x_even), orig_forward(conv, x_even))
    _assert_close(conv(x_odd), orig_forward(conv, x_odd))

    # The padded weight is built once per weight tensor, not per call.
    first_ptr = conv._padded_weight.data_ptr()
    conv(x_even)
    assert conv._padded_weight.data_ptr() == first_ptr

    # Rebuild the padded weight when the source weight changes.
    conv.weight = torch.nn.Parameter(torch.randn_like(conv.weight))
    out_patched_even2 = conv(x_even)
    assert conv._padded_weight.data_ptr() != first_ptr
    _assert_close(out_patched_even2, orig_forward(conv, x_even))
