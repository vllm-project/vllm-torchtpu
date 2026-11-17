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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.autoencoders import autoencoder_kl_ltx2

from vllm_torchtpu.omni import ltx2_vae

pytestmark = pytest.mark.omni

_UPSTREAM_RMS_NORM_FORWARD = autoencoder_kl_ltx2.PerChannelRMSNorm.forward
_UPSTREAM_UPSAMPLER_FORWARD = autoencoder_kl_ltx2.LTX2VideoUpsampler3d.forward
_UPSTREAM_DECODER_FORWARD = autoencoder_kl_ltx2.LTX2VideoDecoder3d.forward


@pytest.fixture
def patched_vae(monkeypatch):
    monkeypatch.setattr(
        autoencoder_kl_ltx2.PerChannelRMSNorm, "forward", _UPSTREAM_RMS_NORM_FORWARD
    )
    monkeypatch.setattr(
        autoencoder_kl_ltx2.LTX2VideoUpsampler3d, "forward", _UPSTREAM_UPSAMPLER_FORWARD
    )
    monkeypatch.setattr(
        autoencoder_kl_ltx2.LTX2VideoDecoder3d, "forward", _UPSTREAM_DECODER_FORWARD
    )
    ltx2_vae.apply_ltx2_video_vae_patches()
    ltx2_vae.apply_ltx2_video_vae_patches()  # idempotent
    return autoencoder_kl_ltx2


def test_per_channel_rms_norm_matches_upstream(patched_vae):
    torch.manual_seed(0)
    norm = patched_vae.PerChannelRMSNorm()
    x = torch.randn(2, 16, 3, 4, 4)
    got = norm(x)
    want = _UPSTREAM_RMS_NORM_FORWARD(norm, x)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("residual", [True, False])
def test_upsampler_pre_permute_residual_matches_upstream(patched_vae, residual):
    torch.manual_seed(0)
    up = patched_vae.LTX2VideoUpsampler3d(
        in_channels=16,
        out_channels=16,
        stride=(2, 2, 2),
        residual=residual,
        upscale_factor=2,
    )
    x = torch.randn(1, 16, 3, 4, 4)
    got = up(x, causal=True)
    want = _UPSTREAM_UPSAMPLER_FORWARD(up, x, causal=True)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_decoder_unpatchify_and_tpu_compile_delegation(patched_vae, monkeypatch):
    torch.manual_seed(0)
    dec = patched_vae.LTX2VideoDecoder3d(
        in_channels=8,
        out_channels=3,
        block_out_channels=(16, 16),
        spatio_temporal_scaling=(True, False),
        layers_per_block=(1, 1, 1),
        upsample_type=("spatiotemporal", "spatiotemporal"),
        patch_size=2,
        patch_size_t=1,
        resnet_norm_eps=1e-6,
        is_causal=True,
        inject_noise=(False, False, False),
        timestep_conditioning=False,
        upsample_residual=(True, False),
        upsample_factor=(2, 1),
    )
    x = torch.randn(1, 8, 3, 4, 4)
    got = dec(x, temb=None, causal=True)
    want = _UPSTREAM_DECODER_FORWARD(dec, x, temb=None, causal=True)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    # Verify TPU device inputs trigger torch.compile(backend="tpu", dynamic=False) once.
    compile_calls = []

    def fake_compile(fn, *, backend, dynamic):
        compile_calls.append((backend, dynamic))
        return lambda self_arg, hs, t, c: ("compiled_ok", hs.shape)

    monkeypatch.setattr(torch, "compile", fake_compile)
    fake_tpu_tensor = MagicMock(spec=torch.Tensor)
    fake_tpu_tensor.device = SimpleNamespace(type="tpu")
    fake_tpu_tensor.shape = (1, 8, 3, 4, 4)

    res1 = patched_vae.LTX2VideoDecoder3d.forward(dec, fake_tpu_tensor, None, True)
    res2 = patched_vae.LTX2VideoDecoder3d.forward(dec, fake_tpu_tensor, None, True)
    assert res1 == ("compiled_ok", (1, 8, 3, 4, 4))
    assert res2 == res1
    assert compile_calls == [("tpu", False)]
