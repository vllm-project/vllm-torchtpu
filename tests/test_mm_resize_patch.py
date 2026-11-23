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
"""The strip resize gives the HF processors torchvision's bytes."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF
from transformers import image_processing_backends as backends
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)

from vllm_torchtpu import mm_resize_patch as patch_mod


def test_patch_keeps_another_upstream(monkeypatch):
    """Any other upstream source keeps upstream's resize."""
    tv = backends.TorchvisionBackend
    monkeypatch.setattr(tv, "resize", patch_mod.UPSTREAM_RESIZE)
    monkeypatch.setattr(patch_mod, "UPSTREAM_SHA256", "0" * 64)
    patch_mod.patch_mm_resize_strips()
    assert tv.resize is patch_mod.UPSTREAM_RESIZE


def test_processor_pixels_unchanged(monkeypatch):
    """The patch installs over the installed transformers, and a
    torchvision-backend processor gives the same pixels through the strips."""
    tv = backends.TorchvisionBackend
    monkeypatch.setattr(patch_mod, "host_threads", lambda: 8)  # strips on any host
    bounds, strips = patch_mod._bounds, []
    monkeypatch.setattr(
        patch_mod, "_bounds", lambda n, parts: strips.append(n) or bounds(n, parts)
    )
    hwc = np.random.default_rng(1).integers(0, 256, (1754, 3034, 3), dtype=np.uint8)
    proc = Qwen2VLImageProcessor()  # resizes to 756x1316: 7 strips
    kwargs = dict(images=[Image.fromarray(hwc)], return_tensors="pt")
    monkeypatch.setattr(tv, "resize", patch_mod.UPSTREAM_RESIZE)
    want = proc(**kwargs)["pixel_values"]
    patch_mod.patch_mm_resize_strips()
    assert tv.resize is patch_mod.resize, (
        "Upstream TorchvisionBackend.resize has drifted; re-sync "
        "src/vllm_torchtpu/mm_resize_patch.py."
    )
    assert torch.equal(proc(**kwargs)["pixel_values"], want)
    assert strips  # the resize ran in strips


BICUBIC, BILINEAR = tvF.InterpolationMode.BICUBIC, tvF.InterpolationMode.BILINEAR


@pytest.mark.parametrize(
    ("shape", "size"),
    [
        ((1754, 3034, 3), (1760, 3040)),  # Cosmos3's 3034x1754 JPEG, up
        ((1754, 3034, 3), (877, 1517)),  # down, odd sizes
        ((999, 1201, 4), (999, 640)),  # one pass only, 4 channels
        ((641, 1003, 3), (1300, 1003)),
        ((2, 600, 700, 3), (513, 1025)),  # a batch of images
        ((1, 2000, 3), (5, 70000)),  # fewer rows than strips
    ],
)
@pytest.mark.parametrize("interpolation", [BICUBIC, BILINEAR])
@pytest.mark.parametrize("antialias", [True, False])
def test_strips_match_torchvision(monkeypatch, shape, size, interpolation, antialias):
    """Every output byte, and the layout, equal one tvF.resize call on the
    channels-last view a PIL image converts to."""
    monkeypatch.setattr(patch_mod, "host_threads", lambda: 8)  # strips on any host
    hwc = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    image = torch.from_numpy(hwc).movedim(-1, -3)
    want = tvF.resize(
        image, list(size), interpolation=interpolation, antialias=antialias
    )
    got = patch_mod.resize_in_strips(image, size, interpolation, antialias)
    assert torch.equal(got, want)
    assert got.stride() == want.stride()
