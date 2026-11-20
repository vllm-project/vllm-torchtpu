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

import hashlib
import inspect

import numpy as np
import pytest
import torch
import transformers
from torchvision.transforms.v2 import functional as tvF

from vllm_torchtpu import mm_resize_patch as patch_mod

PINNED_TRANSFORMERS_VERSION = "5.20.2"
PINNED_SOURCE_HASH = (
    "3f84c32c17b2f8d659e41a2e0bf6f2bb6cae383d2c52a92b9cbf9d98ade07041"
)


@pytest.mark.skipif(
    transformers.__version__ != PINNED_TRANSFORMERS_VERSION,
    reason=f"resize reproduces transformers {PINNED_TRANSFORMERS_VERSION}",
)
def test_upstream_source_unchanged():
    """Fail when upstream changes the resize this patch reproduces."""
    source = inspect.getsource(patch_mod.UPSTREAM_RESIZE)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert digest == PINNED_SOURCE_HASH, (
        "Upstream TorchvisionBackend.resize has drifted; re-sync "
        "src/vllm_torchtpu/mm_resize_patch.py."
    )


BICUBIC, BILINEAR = tvF.InterpolationMode.BICUBIC, tvF.InterpolationMode.BILINEAR


@pytest.mark.parametrize(
    ("shape", "size"),
    [
        ((1754, 3034, 3), (1760, 3040)),  # Cosmos3's 3034x1754 JPEG, up
        ((1754, 3034, 3), (877, 1517)),  # down, odd sizes
        ((999, 1201, 4), (999, 640)),  # one pass only, 4 channels
        ((641, 1003, 3), (1300, 1003)),
        ((2, 600, 700, 3), (513, 1025)),  # a batch of images
    ],
)
@pytest.mark.parametrize("interpolation", [BICUBIC, BILINEAR])
@pytest.mark.parametrize("antialias", [True, False])
def test_strips_match_torchvision(shape, size, interpolation, antialias):
    """Every output byte, and the layout, equal one tvF.resize call on the
    channels-last view a PIL image converts to."""
    hwc = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    image = torch.from_numpy(hwc).movedim(-1, -3)
    want = tvF.resize(image, list(size), interpolation=interpolation, antialias=antialias)
    got = patch_mod.resize_in_strips(image, size, interpolation, antialias)
    assert torch.equal(got, want)
    assert got.stride() == want.stride()


def test_processor_pixels_unchanged(monkeypatch):
    """Qwen2-VL's image processor gives the same pixels with the patched
    resize, for a PIL image as the renderer hands it."""
    from PIL import Image
    from transformers import image_processing_backends
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    hwc = np.random.default_rng(1).integers(0, 256, (1754, 3034, 3), dtype=np.uint8)
    proc = Qwen2VLImageProcessor(
        size={"shortest_edge": 65536, "longest_edge": 16777216},
        patch_size=16,
        merge_size=2,
    )
    kwargs = dict(images=[Image.fromarray(hwc)], return_tensors="pt")
    want = proc(**kwargs)["pixel_values"]
    monkeypatch.setattr(
        image_processing_backends.TorchvisionBackend, "resize", patch_mod.resize
    )
    assert torch.equal(proc(**kwargs)["pixel_values"], want)
