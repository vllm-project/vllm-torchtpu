# SPDX-License-Identifier: Apache-2.0
"""The encoder graph protocol must match ordinary image normalization."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration as Model
from vllm.model_executor.models.vision import FusedInputNorm

from vllm_torchtpu.models.vllm.qwen2_5_vl_patch import maybe_patch_qwen2_5_vl

pytestmark = pytest.mark.cpu_test


class RecordingVisionEncoder:
    dtype = torch.float32
    spatial_merge_size = 1

    def __call__(self, pixels, *args, **kwargs):
        return pixels


@pytest.mark.parametrize("device_normalization", [True, False])
@pytest.mark.parametrize("encoder_path", ["graph", "image", "video"])
def test_encoder_protocol_normalizes_like_ordinary_image_path(
        monkeypatch, device_normalization, encoder_path):
    # Restore class methods afterwards; patch installation is otherwise owned
    # by the worker's patch registry and lasts for the process lifetime.
    monkeypatch.setattr(Model, "encoder_cudagraph_forward",
                        Model.encoder_cudagraph_forward)
    monkeypatch.setattr(Model, "encoder_eager_forward",
                        Model.encoder_eager_forward)
    maybe_patch_qwen2_5_vl(
        SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen2_5_vl")))

    normalizer = (FusedInputNorm([0.5] * 3, [0.25] * 3, 1 / 255)
                  if device_normalization else FusedInputNorm.identity())
    model = SimpleNamespace(visual=RecordingVisionEncoder(),
                            input_norm=normalizer,
                            use_data_parallel=False)
    for name in ("get_input_modality", "_get_pixel_values_by_modality",
                 "_get_grid_thw_by_modality"):
        setattr(model, name, MethodType(getattr(Model, name), model))
    pixels = torch.tensor([[0., 64., 255.], [255., 128., 0.]])
    if not device_normalization:
        pixels = pixels / 127.5 - 1
    original = pixels.clone()
    grid = torch.tensor([[1, 1, 2]])
    expected = Model._process_image_input(model, {
        "type": "pixel_values",
        "pixel_values": pixels,
        "image_grid_thw": grid,
    })[0]

    if encoder_path == "graph":
        values = {"pixel_values": pixels}
        actual = Model.encoder_cudagraph_forward(model, values)
        assert values["pixel_values"] is pixels
    else:
        pixel_key = ("pixel_values"
                     if encoder_path == "image" else "pixel_values_videos")
        values = {pixel_key: pixels, f"{encoder_path}_grid_thw": grid}
        actual = Model.encoder_eager_forward(model, values)
        assert values[pixel_key] is pixels
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pixels, original)
