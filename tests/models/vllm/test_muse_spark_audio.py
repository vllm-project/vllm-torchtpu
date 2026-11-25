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
"""Bucketed waveform encoding for the Muse Spark audio tower."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.models.muse_spark.common import audio as upstream_audio

from vllm_torchtpu.models.vllm.muse_spark.audio import AutoEncoder, audio_bucket_tokens
from vllm_torchtpu.models.vllm.muse_spark.layers import MuseRMSNorm
from vllm_torchtpu.models.vllm.muse_spark.multimodal import (
    MuseSparkForConditionalGeneration,
)

PATCH_SIZES = (2, 3)
PATCH_SIZE = 6
LATENT_CHANNELS = 8


@pytest.fixture
def encoder(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(upstream_audio, "RMSNorm", nn.RMSNorm)
    torch.manual_seed(5)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        model = AutoEncoder(
            latent_channels=LATENT_CHANNELS,
            patch_size=PATCH_SIZE,
            kernel_size=3,
            patchify_sizes=PATCH_SIZES,
            d_models=(4, 8),
            n_layers=(2, 2),
            prefix="model.audio_tower",
        ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    # This fixture bypasses the loader that installs CPU GEMM dispatch.
    for module in model.modules():
        if isinstance(
            module,
            upstream_audio.ColumnParallelLinear | upstream_audio.RowParallelLinear,
        ):
            module.cpu_linear = F.linear
    return model


def upstream_encode(encoder: AutoEncoder, wav: torch.Tensor) -> torch.Tensor:
    return upstream_audio.AutoEncoder.forward(encoder, [wav])[0]


def waveforms(*lengths: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    return [torch.randn(n, generator=generator) for n in lengths]


@pytest.mark.parametrize(
    ("num_tokens", "bucket"),
    [(1, 16), (16, 16), (17, 32), (33, 64), (512, 512), (513, 1024)],
)
def test_audio_bucket_tokens_rounds_up_to_power_of_two(num_tokens, bucket):
    assert audio_bucket_tokens(num_tokens) == bucket


@torch.inference_mode()
def test_bucketed_forward_matches_upstream_per_waveform(encoder):
    wavs = waveforms(1, 6, 7, 95, 96, 97, 200)
    expected = [upstream_encode(encoder, wav) for wav in wavs]
    shapes = []
    encode = encoder._encode_prefill

    def record(x):
        shapes.append(tuple(x.shape))
        return encode(x)

    encoder._encode_prefill = record

    actual = encoder(wavs)

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # Five waveforms share the 16-token bucket and pad to a batch of eight.
    assert shapes == [
        (8, 16 * PATCH_SIZE, 1),
        (1, 32 * PATCH_SIZE, 1),
        (1, 64 * PATCH_SIZE, 1),
    ]


def omni_with_audio(encoder: AutoEncoder) -> MuseSparkForConditionalGeneration:
    model = MuseSparkForConditionalGeneration.__new__(MuseSparkForConditionalGeneration)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.tok_embeddings_norm = MuseRMSNorm(
        LATENT_CHANNELS, 1e-6, has_weight=False
    )
    model.model.audio_tower = encoder
    return model


@torch.inference_mode()
@pytest.mark.parametrize("layout", ["list", "stacked", "flat"])
def test_encode_audio_normalizes_each_waveform_in_input_order(encoder, layout):
    model = omni_with_audio(encoder)
    wavs = waveforms(40, 40, 40) if layout == "stacked" else waveforms(130, 5, 40)
    norm = model.model.language_model.tok_embeddings_norm
    expected = [norm(upstream_encode(encoder, wav)) for wav in wavs]

    if layout == "list":
        actual = model._encode_audio(wavs)
    elif layout == "stacked":
        actual = model._encode_audio(torch.stack(wavs))
    else:
        counts = torch.tensor([wav.numel() for wav in wavs])
        actual = model._encode_audio(torch.cat(wavs), counts)

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_encode_audio_rejects_flat_samples_that_disagree_with_counts(encoder):
    model = omni_with_audio(encoder)
    with pytest.raises(ValueError, match="do not match audio_sample_counts"):
        model._encode_audio(torch.zeros(10), torch.tensor([4, 5]))


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@torch.inference_mode()
def test_tpu_bucketed_forward_matches_cpu_upstream(encoder, compiled):
    wavs = waveforms(1, 7, 95, 200)
    expected = [upstream_encode(encoder, wav) for wav in wavs]

    encoder.to("tpu")
    if compiled:
        # As MuseSparkForConditionalGeneration compiles it with compile_mm_encoder.
        encoder._encode_prefill = torch.compile(
            encoder._encode_prefill, backend="tpu", fullgraph=True, dynamic=False
        )
    actual = encoder([wav.to("tpu") for wav in wavs])

    for got, want in zip(actual, expected):
        assert got.shape == want.shape
        torch.testing.assert_close(got.cpu(), want, rtol=2e-2, atol=2e-2)
